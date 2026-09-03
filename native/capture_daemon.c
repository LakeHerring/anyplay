/*
 * capture_daemon.c — native GStreamer screen-capture daemon.
 *
 * The single GStreamer implementation of screen capture for the Shadow
 * Dungeon agent. Python never touches GStreamer: this daemon owns the
 * pipeline
 *
 *   ximagesrc -> videorate -> videoscale -> videoconvert -> appsink
 *
 * and publishes every frame (RGB, already scaled to the training size)
 * into a POSIX shared-memory ring. A dedicated pull thread does the
 * producer side; a Python consumer maps the same region and reads the
 * newest frame zero-copy through a per-slot seqlock.
 *
 * Shared-memory layout (little-endian):
 *
 *   offset 0   CapHeader (64 B)
 *      u32 magic, version, width, height, slot_bytes, n_slots, alive, pad
 *      u64 idx     (atomic, release: newest frame id + 1)
 *      u64 frames  (total frames published)
 *      u64 drops   (atomic, overrun drops)
 *
 *   offset 64  CapSlot x n_slots, stride = 32 + slot_bytes
 *      u32 seq (atomic; even = stable, odd = being written)
 *      u32 pad
 *      u64 frame_id (atomic)
 *      u8  data[slot_bytes]
 *
 * Consumer protocol: read hdr.idx (acquire), s = (idx-1) % n_slots,
 * then retry the slot read while seq is odd or changes between the
 * pre/post reads.
 *
 * Control socket (AF_UNIX): one byte per command.
 *   'q'  quit cleanly        -> replies 'Q'
 *   's'  stats               -> replies "frames=N fps=X drops=N\n"
 *
 * Build:  bash native/build.sh      (-> native/capture-daemon)
 * Run:    native/capture-daemon --region x,y,w,h [--width W --height H]
 *                              [--fps 60] [--slots 8] [--display :0.0]
 *                              [--shm PATH] [--sock PATH]
 *
 * Prints "READY <shm> <sock>" on stdout once the pipeline is PLAYING
 * and the shm is mapped.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <inttypes.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <errno.h>
#include <glib.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#define SDAI_MAGIC   0x49445321u
#define SDAI_VERSION 1u
#define HDR_SIZE     64u
#define SLOT_META    32u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t slot_bytes;
    uint32_t n_slots;
    _Atomic uint32_t alive;
    _Atomic uint32_t pad;
    _Atomic uint64_t idx;     /* newest published frame id + 1 */
    _Atomic uint64_t frames;  /* total frames published */
    _Atomic uint64_t drops;   /* overrun drops */
} CapHeader;

typedef struct {
    _Atomic uint32_t seq;     /* even = stable, odd = writing */
    _Atomic uint32_t pad;
    _Atomic uint64_t frame_id;
} CapSlotMeta;

typedef struct {
    char display[32];
    int rx, ry, rw, rh;       /* capture region */
    int out_w, out_h;         /* output (training) size */
    int fps;
    int n_slots;
    char shm[256];
    char sock[256];

    int shm_fd;
    void *shm_mem;
    size_t shm_size;
    size_t slot_stride;
    CapHeader *hdr;

    GstElement *pipe;
    GstAppSink *sink;
    GMainLoop *loop;
    volatile sig_atomic_t running;
    double t0;                /* real-time seconds at start */
    pthread_t pull_th, sock_th;
} Daemon;

static Daemon *g_daemon;

static void on_signal(int sig)
{
    (void)sig;
    if (g_daemon)
        g_daemon->running = 0;
}

static gboolean check_running(gpointer user_data)
{
    (void)user_data;
    if (!g_daemon->running)
        g_main_loop_quit(g_daemon->loop);
    return G_SOURCE_CONTINUE;
}

/* ------------------------------------------------------------------ */
/* producer: pull the latest frame and publish it into the ring       */

static void *pull_thread(void *arg)
{
    Daemon *d = arg;
    uint64_t id = 0;

    while (d->running) {
        /* Blocking pull of the oldest buffered sample; with
         * max-buffers=2 drop=true that is always the newest frame.
         * Returns NULL once the pipeline is torn down. */
        GstSample *sample = gst_app_sink_pull_sample(d->sink);
        if (!sample)
            continue;

        GstBuffer *buf = gst_sample_get_buffer(sample);
        GstMapInfo info;
        if (buf && gst_buffer_map(buf, &info, GST_MAP_READ) &&
            info.size >= d->hdr->slot_bytes) {
            size_t s = id % d->hdr->n_slots;
            uint8_t *base = (uint8_t *)d->shm_mem + HDR_SIZE + s * d->slot_stride;
            CapSlotMeta *slot = (CapSlotMeta *)base;

            uint32_t seq = atomic_load_explicit(&slot->seq, memory_order_relaxed);
            atomic_store_explicit(&slot->seq, seq + 1, memory_order_relaxed); /* odd */
            atomic_store_explicit(&slot->frame_id, id, memory_order_relaxed);
            memcpy(base + SLOT_META, info.data, d->hdr->slot_bytes);
            atomic_store_explicit(&slot->seq, seq + 2, memory_order_relaxed); /* even */

            atomic_store_explicit(&d->hdr->idx, id + 1, memory_order_release);
            atomic_store_explicit(&d->hdr->frames, id + 1, memory_order_relaxed);
            id++;
            gst_buffer_unmap(buf, &info);
        } else {
            atomic_fetch_add_explicit(&d->hdr->drops, 1, memory_order_relaxed);
        }
        gst_sample_unref(sample);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* control socket                                                      */

static void xwrite(int fd, const char *p, size_t n)
{
    size_t off = 0;
    while (off < n) {
        ssize_t k = write(fd, p + off, n - off);
        if (k <= 0)
            return;
        off += (size_t)k;
    }
}

static void *sock_thread(void *arg)
{
    Daemon *d = arg;
    int lfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (lfd < 0) {
        fprintf(stderr, "socket(): %s\n", strerror(errno));
        return NULL;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof addr);
    if (strlen(d->sock) >= sizeof addr.sun_path) {
        fprintf(stderr, "socket path too long: %s\n", d->sock);
        close(lfd);
        return NULL;
    }
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, d->sock, sizeof addr.sun_path - 1);
    unlink(d->sock);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof addr) < 0 ||
        listen(lfd, 4) < 0) {
        fprintf(stderr, "bind/listen %s: %s\n", d->sock, strerror(errno));
        close(lfd);
        return NULL;
    }

    while (d->running) {
        fd_set set;
        struct timeval tmo = { 0, 100 * 1000 };
        FD_ZERO(&set);
        FD_SET(lfd, &set);
        if (select(lfd + 1, &set, NULL, NULL, &tmo) <= 0)
            continue;

        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0)
            continue;
        for (;;) {
            char c = 0;
            ssize_t n = read(cfd, &c, 1);
            if (n <= 0)
                break;
            if (c == 'q') {
                xwrite(cfd, "Q", 1);
                d->running = 0;
                if (d->loop)
                    g_main_loop_quit(d->loop);
                /* unblock the pull thread */
                gst_element_set_state(GST_ELEMENT(d->pipe), GST_STATE_NULL);
                close(cfd);
                break;
            } else if (c == 's') {
                double dt = g_get_real_time() / 1e6 - d->t0;
                double fps = dt > 0 ? (double)atomic_load(&d->hdr->frames) / dt : 0.0;
                char buf[160];
                int len = snprintf(buf, sizeof buf,
                                   "frames=%" PRIu64 " fps=%.1f drops=%" PRIu64 "\n",
                                   atomic_load(&d->hdr->frames), fps,
                                   atomic_load(&d->hdr->drops));
                xwrite(cfd, buf, (size_t)len);
            }
        }
        close(cfd);
    }
    close(lfd);
    return NULL;
}

/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --region x,y,w,h [--width W --height H] [--fps N]\n"
            "                [--slots N] [--display :0.0] [--shm PATH] [--sock PATH]\n",
            prog);
}

static int parse_args(Daemon *d, int argc, char **argv)
{
    memset(d, 0, sizeof *d);
    strncpy(d->display, ":0.0", sizeof d->display - 1);
    d->fps = 60;
    d->n_slots = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--region") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%d,%d,%d", &d->rx, &d->ry, &d->rw, &d->rh) != 4) {
                fprintf(stderr, "bad --region (want x,y,w,h)\n");
                return -1;
            }
        } else if (!strcmp(argv[i], "--width") && i + 1 < argc) {
            d->out_w = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--height") && i + 1 < argc) {
            d->out_h = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fps") && i + 1 < argc) {
            d->fps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--slots") && i + 1 < argc) {
            d->n_slots = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--display") && i + 1 < argc) {
            strncpy(d->display, argv[++i], sizeof d->display - 1);
        } else if (!strcmp(argv[i], "--shm") && i + 1 < argc) {
            strncpy(d->shm, argv[++i], sizeof d->shm - 1);
        } else if (!strcmp(argv[i], "--sock") && i + 1 < argc) {
            strncpy(d->sock, argv[++i], sizeof d->sock - 1);
        } else {
            usage(argv[0]);
            return -1;
        }
    }
    if (d->rw <= 0 || d->rh <= 0) {
        fprintf(stderr, "--region x,y,w,h is required (all > 0)\n");
        return -1;
    }
    if (d->out_w <= 0 || d->out_h <= 0) {
        d->out_w = d->rw;
        d->out_h = d->rh;
    }
    if (d->n_slots < 2 || d->n_slots > 64) {
        fprintf(stderr, "--slots must be 2..64\n");
        return -1;
    }
    if (!d->shm[0])
        snprintf(d->shm, sizeof d->shm, "/dev/shm/anyplay_cap_%d.bin", (int)getpid());
    if (!d->sock[0])
        snprintf(d->sock, sizeof d->sock, "/tmp/anyplay_cap_%d.sock", (int)getpid());
    return 0;
}

static int make_shm(Daemon *d)
{
    size_t slot_bytes = (size_t)d->out_w * d->out_h * 3;
    d->slot_stride = SLOT_META + slot_bytes;
    d->shm_size = HDR_SIZE + (size_t)d->n_slots * d->slot_stride;

    unlink(d->shm);
    d->shm_fd = open(d->shm, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (d->shm_fd < 0) {
        fprintf(stderr, "open %s: %s\n", d->shm, strerror(errno));
        return -1;
    }
    if (ftruncate(d->shm_fd, (off_t)d->shm_size) < 0) {
        fprintf(stderr, "ftruncate: %s\n", strerror(errno));
        return -1;
    }
    d->shm_mem = mmap(NULL, d->shm_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                  d->shm_fd, 0);
    if (d->shm_mem == MAP_FAILED) {
        fprintf(stderr, "mmap: %s\n", strerror(errno));
        return -1;
    }
    memset(d->shm_mem, 0, d->shm_size);

    CapHeader *h = (CapHeader *)d->shm_mem;
    h->magic = SDAI_MAGIC;
    h->version = SDAI_VERSION;
    h->width = (uint32_t)d->out_w;
    h->height = (uint32_t)d->out_h;
    h->slot_bytes = (uint32_t)slot_bytes;
    h->n_slots = (uint32_t)d->n_slots;
    atomic_store_explicit(&h->alive, 1, memory_order_relaxed);
    d->hdr = h;
    return 0;
}

static void shm_destroy(Daemon *d)
{
    if (d->hdr)
        atomic_store_explicit(&d->hdr->alive, 0, memory_order_relaxed);
    if (d->shm_mem && d->shm_mem != MAP_FAILED)
        munmap(d->shm_mem, d->shm_size);
    if (d->shm_fd >= 0)
        close(d->shm_fd);
    unlink(d->shm);
}

int main(int argc, char **argv)
{
    Daemon d;
    if (parse_args(&d, argc, argv) < 0)
        return 2;

    struct sigaction sa = { .sa_handler = on_signal };
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    gst_init(&argc, &argv);

    if (make_shm(&d) < 0)
        return 1;

    char desc[1024];
    snprintf(desc, sizeof desc,
             "ximagesrc display-name=%s startx=%d starty=%d endx=%d endy=%d "
             "do-timestamp=true use-damage=false "
             "! video/x-raw,framerate=%d/1 ! videorate ! videoscale ! videoconvert "
             "! video/x-raw,width=%d,height=%d,format=RGB "
             "! appsink name=sink max-buffers=2 drop=true",
             d.display, d.rx, d.ry, d.rx + d.rw - 1, d.ry + d.rh - 1,
             d.fps, d.out_w, d.out_h);

    GError *err = NULL;
    d.pipe = gst_parse_launch(desc, &err);
    if (!d.pipe) {
        fprintf(stderr, "parse: %s\n", err ? err->message : "?");
        if (err) g_error_free(err);
        shm_destroy(&d);
        return 1;
    }

    d.sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(d.pipe), "sink"));
    if (!GST_IS_APP_SINK(d.sink)) {
        fprintf(stderr, "appsink not found\n");
        shm_destroy(&d);
        return 1;
    }

    d.loop = g_main_loop_new(NULL, FALSE);
    d.running = 1;
    d.t0 = g_get_real_time() / 1e6;
    g_daemon = &d;

    if (gst_element_set_state(GST_ELEMENT(d.pipe), GST_STATE_PLAYING)
        == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "set_state(PLAYING) failed\n");
        shm_destroy(&d);
        return 1;
    }
    GstStateChangeReturn res;
    GstState state;
    res = gst_element_get_state(GST_ELEMENT(d.pipe), &state, 0, 5 * GST_SECOND);
    GstBus *bus = gst_element_get_bus(GST_ELEMENT(d.pipe));
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, 0, GST_MESSAGE_ERROR);
    if (msg) {
        GError *e2 = NULL;
        gst_message_parse_error(msg, &e2, NULL);
        fprintf(stderr, "pipeline error: %s\n", e2 ? e2->message : "?");
        if (e2) g_error_free(e2);
        gst_message_unref(msg);
        shm_destroy(&d);
        return 1;
    }
    if (res == GST_STATE_CHANGE_FAILURE ||
        (state != GST_STATE_PLAYING && state != GST_STATE_PAUSED)) {
        fprintf(stderr, "pipeline did not reach PLAYING (res=%d state=%d)\n",
                (int)res, (int)state);
        shm_destroy(&d);
        return 1;
    }

    pthread_create(&d.pull_th, NULL, pull_thread, &d);
    pthread_create(&d.sock_th, NULL, sock_thread, &d);
    g_timeout_add(100, check_running, NULL);

    printf("READY %s %s\n", d.shm, d.sock);
    fflush(stdout);

    g_main_loop_run(d.loop);

    d.running = 0;
    gst_element_set_state(GST_ELEMENT(d.pipe), GST_STATE_NULL); /* unblocks pull */
    pthread_join(d.pull_th, NULL);
    pthread_join(d.sock_th, NULL);
    g_main_loop_unref(d.loop);
    gst_object_unref(d.sink);
    gst_object_unref(d.pipe);
    unlink(d.sock);
    shm_destroy(&d);
    return 0;
}
