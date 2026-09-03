/*
 * portal-capture.c
 *
 * Standalone Wayland capture daemon driven by xdg-desktop-portal + PipeWire.
 *
 * Designed for "any game": it captures whatever window the portal hands it and
 * AUTO-DETECTS the window's native resolution from the first frame, so no
 * per-game size config is needed.  --width/--height are optional overrides
 * that force a scaled output (e.g. to match a trained model's input).
 *
 * Control flow (portal-pw-fd execs us):
 *   1. Build a GStreamer pipeline on the inherited PipeWire fd:
 *        pipewiresrc fd=N path=M ! videoconvert [! videoscale] !
 *            video/x-raw,format=RGB[,width=W,height=H] ! appsink
 *      (no size constraint in auto mode -> native window size; videoscale only
 *       when a size override is given)
 *   2. Pull the first frame to learn the real WxH (and confirm frames flow).
 *   3. Allocate the shm ring at that size.
 *   4. Print "READY <shm> <sock>" on stdout (only once the size is known).
 *   5. Stream: a pull thread reads frames and publishes each to the shm ring.
 *
 * The Python consumer (native_capture.py) is unchanged: it reads the shm ring
 * header for WxH/slot layout and uses the control socket for quit/stats.
 *
 * The inherited PipeWire fd (PW_FD) is held open for the whole lifetime -- it
 * is the portal session keepalive; closing it ends the share.
 *
 * Build: see native/build.sh (links GStreamer).
 *
 * --test MODE (no live window needed): uses videotestsrc instead of
 * pipewiresrc so the detect + shm + publish path can be validated in CI.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#define CAPTURE_MAGIC   0x49445321u  /* "!SDI" */
#define CAPTURE_VERSION 1u
#define MAX_SLOTS       64
#define MAX_RETRY       20
#define DETECT_TIMEOUT_MS 10000

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t slot_bytes;
    uint32_t n_slots;
    uint32_t alive;
    uint32_t pad;
} CapHeader32;

typedef struct {
    uint64_t idx;
    uint64_t frames;
    uint64_t drops;
} CapHeader64;

typedef struct {
    uint32_t seq;
    uint32_t pad;
    uint64_t frame_id;
} CapSlotHead;

typedef struct Capture {
    int pw_fd;
    unsigned node_id;

    /* requested override (0 = auto-detect) */
    int force_w;
    int force_h;

    /* effective (detected or forced) size, known after detect */
    int out_w;
    int out_h;

    int slots;
    char shm_path[256];   /* canonical /dev/shm/<name> path (reported in READY) */
    char shm_name[256];   /* shm_open name (no slashes) */
    char sock_path[256];

    volatile int running;
    volatile uint64_t frame_id;
    volatile uint64_t drops;

    GstElement *pipeline;
    GstElement *sink;
    int pull_rc;      /* set by pull_thread: 0 ok/quit, 1 stream ended w/ 0 frames */

    void *shm;
    size_t shm_size;
    size_t slot_stride;
    int test;         /* videotestsrc mode */
} Capture;

static Capture pc;

static void on_signal(int sig) {
    (void)sig;
    pc.running = 0;
}

/* ------------------------------------------------------------------ */
/* shm ring                                                            */
/* ------------------------------------------------------------------ */

static int make_shm(void) {
    int w = pc.out_w, h = pc.out_h, n = pc.slots;
    if (w <= 0 || h <= 0 || n <= 0)
        return -1;

    size_t slot_bytes = (size_t)w * h * 3;
    size_t slot_stride = (32 + slot_bytes);
    size_t total = 64 + (size_t)n * slot_stride;

    /* use the caller-provided path, or a per-pid default */
    if (pc.shm_path[0] == '\0')
        snprintf(pc.shm_path, sizeof pc.shm_path, "/dev/shm/anyplay_pc_%d.bin",
                 (int)getpid());

    int fd = shm_open(pc.shm_name, O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (fd < 0) {
        fprintf(stderr, "[portal-capture] shm_open %s: %s\n", pc.shm_name,
                strerror(errno));
        return -1;
    }

    if (ftruncate(fd, (off_t)total) < 0) {
        fprintf(stderr, "[portal-capture] ftruncate: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    void *m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        fprintf(stderr, "[portal-capture] mmap: %s\n", strerror(errno));
        return -1;
    }

    pc.shm = m;
    pc.shm_size = total;
    pc.slot_stride = slot_stride;

    memset(m, 0, total);

    CapHeader32 *h32 = (CapHeader32 *)m;
    h32->magic = CAPTURE_MAGIC;
    h32->version = CAPTURE_VERSION;
    h32->width = (uint32_t)w;
    h32->height = (uint32_t)h;
    h32->slot_bytes = (uint32_t)slot_bytes;
    h32->n_slots = (uint32_t)n;
    h32->alive = 1;

    fprintf(stderr,
            "[portal-capture] shm %s: %zu bytes, %dx%d RGB, %d slots, "
            "slot stride %zu\n", pc.shm_path, total, w, h, n, slot_stride);

    return 0;
}

static void publish_frame(const void *data, size_t size, size_t src_stride,
                          int w, int h) {
    if (!pc.shm || !pc.running)
        return;

    uint64_t slot_idx = (uint64_t)(pc.frame_id % (uint64_t)pc.slots);
    uint64_t fid = pc.frame_id;
    size_t slot_bytes = (size_t)w * h * 3;
    size_t slot_off = 64 + slot_idx * pc.slot_stride;
    uint8_t *slot = (uint8_t *)pc.shm + slot_off;

    CapSlotHead *sh = (CapSlotHead *)slot;
    __sync_lock_test_and_set(&sh->seq, 1);  /* odd: writing */
    __sync_synchronize();

    if (src_stride == (size_t)w * 3 && size >= slot_bytes) {
        memcpy(slot + 32, data, slot_bytes);
    } else {
        const uint8_t *src = (const uint8_t *)data;
        size_t row = (size_t)w * 3;
        for (int y = 0; y < h; y++) {
            size_t rstride = src_stride < row ? src_stride : row;
            if (rstride > row)
                rstride = row;
            memcpy(slot + 32 + (size_t)y * row, src + (size_t)y * src_stride,
                   rstride);
        }
    }

    sh->frame_id = fid;
    __sync_synchronize();
    __sync_lock_test_and_set(&sh->seq, (uint32_t)(fid * 2 + 2));  /* even: done */

    CapHeader64 *h64 = (CapHeader64 *)((uint8_t *)pc.shm + 32);
    __sync_lock_test_and_set(&h64->idx, slot_idx);
    __sync_lock_test_and_set(&h64->frames, fid + 1);

    pc.frame_id++;
}

/* ------------------------------------------------------------------ */
/* GStreamer pipeline                                                  */
/* ------------------------------------------------------------------ */

static int build_and_play(void) {
    char src[256];
    char mid[256];

    if (pc.test) {
        snprintf(src, sizeof src,
                 "videotestsrc is-live=true pattern=smpte");
    } else {
        snprintf(src, sizeof src, "pipewiresrc fd=%d path=%u", pc.pw_fd,
                 (unsigned)pc.node_id);
    }

    if (pc.force_w > 0 && pc.force_h > 0)
        snprintf(mid, sizeof mid,
                 "videoconvert ! videoscale ! "
                 "video/x-raw,format=RGB,width=%d,height=%d",
                 pc.force_w, pc.force_h);
    else
        snprintf(mid, sizeof mid, "videoconvert ! video/x-raw,format=RGB");

    char tmpl[768];
    snprintf(tmpl, sizeof tmpl,
             "%s ! %s ! appsink name=sink max-buffers=8 drop=true sync=false",
             src, mid);

    fprintf(stderr, "[portal-capture] pipeline: %s\n", tmpl);

    GError *err = NULL;
    GstElement *pipe = gst_parse_launch(tmpl, &err);
    if (!pipe) {
        fprintf(stderr, "[portal-capture] gst_parse_launch: %s\n",
                err ? err->message : "unknown");
        if (err)
            g_error_free(err);
        return -1;
    }

    GstElement *sink = gst_bin_get_by_name(GST_BIN(pipe), "sink");
    if (!sink) {
        fprintf(stderr, "[portal-capture] appsink not found\n");
        gst_object_unref(pipe);
        return -1;
    }

    pc.pipeline = pipe;
    pc.sink = sink;
    pc.running = 1;

    if (gst_element_set_state(pipe, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[portal-capture] set_state(PLAYING) failed\n");
    }

    GstStateChangeReturn scr =
        gst_element_get_state(pipe, NULL, NULL, 5 * GST_SECOND);
    if (scr == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[portal-capture] pipeline FAILED to reach PLAYING\n");
        gst_object_unref(sink);
        gst_object_unref(pipe);
        pc.pipeline = NULL;
        pc.sink = NULL;
        return -1;
    }

    fprintf(stderr, "[portal-capture] pipeline PLAYING\n");
    return 0;
}

static void teardown(void) {
    if (pc.pipeline) {
        gst_element_set_state(pc.pipeline, GST_STATE_NULL);
        gst_object_unref(pc.sink);
        gst_object_unref(pc.pipeline);
        pc.pipeline = NULL;
        pc.sink = NULL;
    }
}

/*
 * Pull frames until the first one arrives; read its caps to learn the real
 * output WxH.  Returns:
 *    0  -> got a frame; pc.out_w/pc.out_h set
 *   -1  -> pipeline EOS/error or quit (stream is over / not retryable)
 *   -2  -> no frame within DETECT_TIMEOUT_MS but pipeline still alive (retry)
 */
static int detect_first_frame(void) {
    GstAppSink *sink = GST_APP_SINK(pc.sink);
    GstBus *bus = gst_element_get_bus(pc.pipeline);

    GTimeVal t0;
    g_get_current_time(&t0);

    for (;;) {
        if (!pc.running)
            return -1;

        GstSample *sample =
            gst_app_sink_try_pull_sample(sink, 500 * GST_MSECOND);
        if (sample) {
            GstCaps *caps = gst_sample_get_caps(sample);
            int w = 0, h = 0;
            if (caps) {
                GstStructure *s = gst_caps_get_structure(caps, 0);
                int sv;
                if (s) {
                    if (gst_structure_get_int(s, "width", &sv))
                        w = sv;
                    if (gst_structure_get_int(s, "height", &sv))
                        h = sv;
                }
            }
            gst_sample_unref(sample);
            if (w > 0 && h > 0) {
                pc.out_w = w;
                pc.out_h = h;
                fprintf(stderr, "[portal-capture] detected %dx%d RGB\n", w, h);
                return 0;
            }
        }

        GstMessage *msg =
            gst_bus_pop_filtered(bus, GST_MESSAGE_EOS | GST_MESSAGE_ERROR);
        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *e = NULL;
                gchar *dbg = NULL;
                gst_message_parse_error(msg, &e, &dbg);
                fprintf(stderr, "[portal-capture] detect: pipeline error: %s\n",
                        e ? e->message : "?");
                if (e)
                    g_error_free(e);
                if (dbg)
                    g_free(dbg);
            }
            gst_message_unref(msg);
            return -1;
        }

        GTimeVal now;
        g_get_current_time(&now);
        long elapsed = (now.tv_sec - t0.tv_sec) * 1000 +
                       (now.tv_usec - t0.tv_usec) / 1000;
        if (elapsed > DETECT_TIMEOUT_MS) {
            fprintf(stderr,
                    "[portal-capture] detect: no frame after %ld ms\n",
                    elapsed);
            return -2;
        }
    }
}

/* ------------------------------------------------------------------ */
/* pull thread                                                         */
/* ------------------------------------------------------------------ */

static void *pull_thread(void *arg) {
    Capture *p = (Capture *)arg;
    GstAppSink *sink = GST_APP_SINK(p->sink);
    GstBus *bus = gst_element_get_bus(p->pipeline);
    const GstClockTime poll_to = 150 * GST_MSECOND;
    uint64_t frames = 0;

    for (;;) {
        if (!p->running)
            break;

        GstSample *sample = gst_app_sink_try_pull_sample(sink, poll_to);
        if (sample) {
            GstBuffer *buf = gst_sample_get_buffer(sample);
            GstMapInfo info;
            if (buf && gst_buffer_map(buf, &info, GST_MAP_READ)) {
                size_t stride = (size_t)p->out_w * 3;
                int w = p->out_w, h = p->out_h;
                GstCaps *caps = gst_sample_get_caps(sample);
                if (caps) {
                    GstStructure *s = gst_caps_get_structure(caps, 0);
                    int sv;
                    if (s) {
                        if (gst_structure_get_int(s, "width", &sv))
                            w = sv;
                        if (gst_structure_get_int(s, "height", &sv))
                            h = sv;
                        if (gst_structure_get_int(s, "stride", &sv))
                            stride = (size_t)sv;
                    }
                }
                if (w != p->out_w || h != p->out_h) {
                    /* window resized mid-session; shm can't follow */
                    fprintf(stderr, "[portal-capture] geometry changed to "
                            "%dx%d (expected %dx%d); dropping\n",
                            w, h, p->out_w, p->out_h);
                } else if (info.size >= (size_t)h * stride) {
                    publish_frame(info.data, info.size, stride, w, h);
                    frames++;
                }
                gst_buffer_unmap(buf, &info);
            }
            gst_sample_unref(sample);
        } else {
            GstMessage *msg =
                gst_bus_pop_filtered(bus, GST_MESSAGE_EOS | GST_MESSAGE_ERROR);
            if (msg) {
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                    GError *e = NULL;
                    gchar *dbg = NULL;
                    gst_message_parse_error(msg, &e, &dbg);
                    fprintf(stderr,
                            "[portal-capture] pipeline error (stream ending): "
                            "%s\n", e ? e->message : "?");
                    if (e)
                        g_error_free(e);
                    if (dbg)
                        g_free(dbg);
                }
                gst_message_unref(msg);
                break;
            }
        }
    }

    p->pull_rc = (frames > 0 || !p->running) ? 0 : 1;
    fprintf(stderr,
            "[portal-capture] pull thread exited: %llu frames, %llu drops\n",
            (unsigned long long)frames, (unsigned long long)p->drops);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* control socket                                                      */
/* ------------------------------------------------------------------ */

static void *sock_thread(void *arg) {
    (void)arg;
    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0)
        return NULL;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, pc.sock_path, sizeof addr.sun_path - 1);

    unlink(pc.sock_path);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        fprintf(stderr, "[portal-capture] bind %s: %s\n", pc.sock_path,
                strerror(errno));
        close(lfd);
        return NULL;
    }
    if (listen(lfd, 4) < 0) {
        close(lfd);
        return NULL;
    }
    fprintf(stderr, "[portal-capture] listening on %s\n", pc.sock_path);

    struct pollfd pf = { .fd = lfd, .events = POLLIN };
    while (pc.running) {
        int pr = poll(&pf, 1, 200);
        if (pr <= 0)
            continue;
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0)
            continue;

        uint8_t b[1];
        if (recv(cfd, b, 1, 0) > 0) {
            if (b[0] == 'q') {
                fprintf(stderr, "[portal-capture] quit requested\n");
                pc.running = 0;
            } else if (b[0] == 's') {
                CapHeader64 *h64 = (CapHeader64 *)((uint8_t *)pc.shm + 32);
                uint64_t f = h64->frames;
                char msg[128];
                snprintf(msg, sizeof msg, "frames=%llu fps=%.1f drops=%llu\n",
                         (unsigned long long)f, (double)f / 1.0,
                         (unsigned long long)pc.drops);
                send(cfd, msg, strlen(msg), 0);
            }
        }
        close(cfd);
    }

    close(lfd);
    unlink(pc.sock_path);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  --width N        force output width (0 = auto-detect, default)\n"
            "  --height N       force output height (0 = auto-detect, default)\n"
            "  --slots N        shm ring slots (default 8, max 64)\n"
            "  --shm PATH       shm path (default /dev/shm/anyplay_pc_<pid>.bin)\n"
            "  --sock PATH      control socket (default /tmp/anyplay_pc_<pid>.sock)\n"
            "  --test           use videotestsrc instead of pipewiresrc (no fd)\n"
            "  -h, --help       this help\n",
            prog);
}

static int parse_args(int argc, char **argv) {
    pc.pw_fd = -1;
    pc.node_id = 0;
    pc.force_w = 0;
    pc.force_h = 0;
    pc.slots = 8;
    pc.out_w = 0;
    pc.out_h = 0;
    pc.running = 1;
    pc.test = 0;

    static struct option opts[] = {
        { "width", required_argument, 0, 'w' },
        { "height", required_argument, 0, 'H' },
        { "slots", required_argument, 0, 's' },
        { "shm", required_argument, 0, 'm' },
        { "sock", required_argument, 0, 'k' },
        { "test", no_argument, 0, 't' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 }
    };

    int c;
    int opt_index = 0;
    while ((c = getopt_long(argc, argv, "w:H:s:m:k:th", opts, &opt_index)) !=
           -1) {
        switch (c) {
        case 'w':
            pc.force_w = atoi(optarg);
            break;
        case 'H':
            pc.force_h = atoi(optarg);
            break;
        case 's':
            pc.slots = atoi(optarg);
            break;
        case 'm':
            snprintf(pc.shm_path, sizeof pc.shm_path, "%s", optarg);
            break;
        case 'k':
            snprintf(pc.sock_path, sizeof pc.sock_path, "%s", optarg);
            break;
        case 't':
            pc.test = 1;
            break;
        case 'h':
            usage(argv[0]);
            exit(0);
        default:
            usage(argv[0]);
            return -1;
        }
    }

    if (pc.slots < 1 || pc.slots > MAX_SLOTS)
        pc.slots = 8;

    if (pc.shm_path[0] == '\0')
        snprintf(pc.shm_path, sizeof pc.shm_path, "/dev/shm/anyplay_pc_%d.bin",
                 (int)getpid());
    if (pc.sock_path[0] == '\0')
        snprintf(pc.sock_path, sizeof pc.sock_path, "/tmp/anyplay_pc_%d.sock",
                 (int)getpid());

    /* shm_open names must contain no slashes; the file lives at
     * /dev/shm/<basename>.  Normalise the reported path to that. */
    {
        const char *base = strrchr(pc.shm_path, '/');
        snprintf(pc.shm_name, sizeof pc.shm_name, "%s",
                 base ? base + 1 : pc.shm_path);
        snprintf(pc.shm_path, sizeof pc.shm_path, "/dev/shm/%s",
                 pc.shm_name);
    }

    if (!pc.test) {
        const char *fd_s = getenv("PW_FD");
        const char *nid_s = getenv("PW_NODE_ID");
        if (!fd_s || !nid_s) {
            fprintf(stderr,
                    "[portal-capture] PW_FD / PW_NODE_ID env not set "
                    "(use --test for a no-fd dry run)\n");
            return -1;
        }
        pc.pw_fd = atoi(fd_s);
        pc.node_id = (unsigned)atoi(nid_s);
    }

    return 0;
}

int main(int argc, char **argv) {
    if (parse_args(argc, argv) < 0)
        return 1;

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    fprintf(stderr,
            "[portal-capture] %s mode; node id %u; %s; %d slots\n",
            pc.test ? "TEST(videotestsrc)" : "portal", pc.node_id,
            (pc.force_w > 0 && pc.force_h > 0)
                ? "forced output"
                : "auto-detect size",
            pc.slots);
    if (!pc.test)
        fprintf(stderr,
                "[portal-capture] portal fd %d held open (session keepalive)\n",
                pc.pw_fd);

    gst_init(NULL, NULL);

    pthread_t sock_tid;
    if (pthread_create(&sock_tid, NULL, sock_thread, NULL) != 0) {
        fprintf(stderr, "[portal-capture] sock thread failed\n");
        return 1;
    }

    /* --- detect phase: build, learn size, retry on transient failure --- */
    int size_known = 0;
    for (int attempt = 1; attempt <= MAX_RETRY; attempt++) {
        if (!pc.running)
            break;

        if (build_and_play() < 0) {
            fprintf(stderr, "[portal-capture] build failed (attempt %d)\n",
                    attempt);
            teardown();
            g_usleep(1000000);
            continue;
        }

        int r = detect_first_frame();
        if (r == 0) {
            size_known = 1;
            break;
        }

        teardown();
        if (r == -1) {
            fprintf(stderr,
                    "[portal-capture] stream ended before first frame; exiting\n");
            break;
        }
        /* r == -2: no frame yet (node not ready / static window) */
        fprintf(stderr,
                "[portal-capture] no frame (attempt %d/%d), retrying\n",
                attempt, MAX_RETRY);
        g_usleep(1000000);
    }

    if (!size_known) {
        if (!pc.running) {
            fprintf(stderr, "[portal-capture] quit during detect\n");
        } else {
            fprintf(stderr, "[portal-capture] could not capture a frame; "
                    "giving up\n");
        }
        return 1;
    }

    /* --- shm at the detected size, then advertise readiness --- */
    if (make_shm() < 0) {
        fprintf(stderr, "[portal-capture] shm allocation failed\n");
        teardown();
        return 1;
    }

    printf("READY %s %s\n", pc.shm_path, pc.sock_path);
    fflush(stdout);

    /* --- stream phase: publish frames; reconnect if the stream ends --- */
    for (int s = 0; ; s++) {
        pthread_t pull_tid;
        if (pthread_create(&pull_tid, NULL, pull_thread, &pc) != 0) {
            fprintf(stderr, "[portal-capture] pull thread failed\n");
            break;
        }
        pthread_join(pull_tid, NULL);

        if (!pc.running)
            break;      /* quit requested */
        if (pc.pull_rc == 0)
            break;      /* frames flowed then stream ended -> session over */

        /* pull_rc == 1: 0 frames, stream ended -> try to reconnect */
        if (s >= MAX_RETRY) {
            fprintf(stderr, "[portal-capture] stream kept ending; giving up\n");
            break;
        }
        fprintf(stderr, "[portal-capture] stream ended; reconnecting\n");
        teardown();
        if (build_and_play() < 0) {
            g_usleep(1000000);
            continue;
        }
    }

    /* --- cleanup --- */
    pc.running = 0;
    teardown();

    if (pc.shm) {
        CapHeader32 *h32 = (CapHeader32 *)pc.shm;
        h32->alive = 0;
        munmap(pc.shm, pc.shm_size);
        shm_unlink(pc.shm_name);
    }

    pthread_join(sock_tid, NULL);

    fprintf(stderr, "[portal-capture] exited cleanly\n");
    return 0;
}
