/*
 * capture_check.c — boring-but-reliable capture diagnostic (TEST 2-4).
 *
 * Goal: prove, in-process, that a GStreamer video source delivers valid
 * RGB frames through an appsink — no ML, no shared memory, no fancy I/O.
 * It is deliberately a standalone check, not the production daemon.
 *
 * Two source modes:
 *
 *   --testsrc
 *     A synthetic source (testsrc, SMPTE bars). Always emits at the
 *     requested rate, so it validates the GStreamer->appsink->RGB-parse
 *     mechanism WITHOUT a portal session or a running game. Use this to
 *     prove the capture plumbing itself is sound.
 *
 *   --portal   (or just set PW_FD/PW_NODE_ID)
 *     A real screencast:
 *         pipewiresrc fd=$PW_FD path=$PW_NODE_ID
 *     The portal handshake (TEST 1) is done by tools/portal-pw-fd, which
 *     exports PW_FD/PW_NODE_ID and execs this binary. Run it as:
 *         native/portal-pw-fd --types 1 --cursor 1 -- \
 *             native/capture-check --portal --width 1920 --height 1080 --duration 5
 *     NOTE: screencast delivery is damage-driven. A fully static screen
 *     yields no frames; the game must be actively rendering.
 *
 * Pipeline (both modes), built in-process:
 *     <source> ! videoconvert ! videoscale
 *              ! video/x-raw,width=W,height=H,format=RGB
 *              ! appsink name=sink max-buffers=4 drop=true sync=false
 *
 * A dedicated pull thread consumes frames with gst_app_sink_pull_sample
 * (the same robust pattern as capture_daemon.c, not a gst-launch+stdout
 * subprocess). For each frame it records: PTS (CLOCK_MONOTONIC domain),
 * negotiated caps (first frame), buffer-size validity, and a sampled
 * min/mean/max pixel histogram so a black/dead frame is distinguishable
 * from a real one.
 *
 * Output (stdout) is a machine-readable block:
 *     CAPTURE OK
 *     source:      testsrc | portal(fd=N,node=ID)
 *     resolution:  WxH
 *     format:      RGB
 *     fps:         <measured>
 *     frames:      <count>
 *     dropped:     <estimated from PTS gaps>
 *     first_frame_ms: <PLAYING -> first frame>
 *     latency_ms:  <mean (consume_time - pts)>
 *     pixels:      min=A mean=B max=C   (sampled)
 *     state:       PLAYING
 *   ... or:  CAPTURE FAIL (<reason>)
 *
 * Build:  bash native/build.sh          (-> native/capture-check)
 * Run:    native/capture-check --testsrc --width 1280 --height 720 --duration 5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <glib.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

typedef struct {
    int  mode;           /* 0 = testsrc, 1 = portal */
    int  pw_fd;
    int  pw_node;
    int  out_w, out_h;
    int  fps;
    double duration_s;   /* seconds to run */
    int  max_frames;     /* stop after N frames (0 = until duration) */

    /* pipeline */
    GstElement *pipe;
    GstAppSink *sink;

    /* runtime */
    volatile sig_atomic_t running;
    double t_playing;    /* real seconds when PLAYING confirmed */
    double t_first;      /* real seconds when first frame consumed */
    double t0;           /* real seconds at start */
    pthread_t pull_th;

    /* collected stats (atomics where the pull thread writes them) */
    _Atomic uint64_t frames;
    _Atomic uint64_t size_mismatch;
    _Atomic uint64_t est_dropped;   /* estimated from PTS gaps (single writer) */
    _Atomic int64_t  last_pts_ns;   /* -1 until first frame */
    _Atomic int      first_seen;
    /* pixel stats accumulated in pull thread (single writer) */
    uint64_t px_samples;
    double   px_sum;
    double   px_min;
    double   px_max;
    /* latency accumulator (single writer) */
    uint64_t lat_samples;
    double   lat_sum_ns;
    char     fmt[32];
    int      fmt_w, fmt_h;
    char     source_desc[128];

    /* optional frame dump (--save PATH): pull thread keeps the latest RGB */
    char     save_path[256];
    uint8_t *last_frame;   /* malloc'd in pull thread */
    int      save_w, save_h;
} Check;

static Check *g_chk;

static void on_signal(int sig) { (void)sig; if (g_chk) g_chk->running = 0; }

static double now_real(void) { return g_get_real_time() / 1e6; }

static double mono_now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------ */
/* pull thread                                                         */

static void *pull_thread(void *arg)
{
    Check *c = arg;
    double interval = 1.0 / (c->fps > 0 ? c->fps : 60.0);

    while (c->running) {
        GstSample *sample = gst_app_sink_pull_sample(c->sink);
        if (!sample)
            continue;

        GstBuffer *buf = gst_sample_get_buffer(sample);
        if (buf) {
            /* negotiated caps from the first real frame (TEST 3) */
            if (!c->first_seen) {
                c->first_seen = 1;
                c->t_first = now_real();
                GstCaps *caps = gst_sample_get_caps(sample);
                if (caps) {
                    GstStructure *s = gst_caps_get_structure(caps, 0);
                    const char *f = gst_structure_get_string(s, "format");
                    if (f) snprintf(c->fmt, sizeof c->fmt, "%s", f);
                    gst_structure_get_int(s, "width", &c->fmt_w);
                    gst_structure_get_int(s, "height", &c->fmt_h);
                }
            }

            int64_t pts = (int64_t)GST_BUFFER_PTS(buf);
            GstMapInfo info;
            if (gst_buffer_map(buf, &info, GST_MAP_READ)) {
                size_t need = (size_t)c->out_w * c->out_h * 3;
                if (info.size < need) {
                    atomic_fetch_add_explicit(&c->size_mismatch, 1, memory_order_relaxed);
                } else {
                    /* lightweight pixel stats: ~4096 points spread across */
                    const uint8_t *d = info.data;
                    size_t stride = info.size / 4096 + 1;
                    double mn = 255.0, mx = 0.0, sum = 0.0;
                    uint64_t n = 0;
                    for (size_t i = 0; i < info.size; i += stride) {
                        double v = d[i];
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                        sum += v;
                        n++;
                    }
                    if (n) {
                        c->px_samples += n;
                        c->px_sum += sum;
                        if (c->px_min > mn) c->px_min = mn;
                        if (c->px_max < mx) c->px_max = mx;
                    }

                    /* keep the latest frame for --save (overwrite) */
                    if (c->save_path[0]) {
                        size_t n2 = (size_t)c->out_w * c->out_h * 3;
                        if (info.size >= n2) {
                            if (!c->last_frame) c->last_frame = malloc(n2);
                            if (c->last_frame) memcpy(c->last_frame, info.data, n2);
                            if (!c->save_w) { c->save_w = c->fmt_w ? c->fmt_w : c->out_w; c->save_h = c->fmt_h ? c->fmt_h : c->out_h; }
                        }
                    }
                }

                /* PTS gap -> estimate frames the sink dropped (drop=true) */
                if (GST_BUFFER_PTS_IS_VALID(buf)) {
                    int64_t last = atomic_load_explicit(&c->last_pts_ns, memory_order_relaxed);
                    if (last >= 0) {
                        int64_t gap = pts - last;
                        if (gap > (int64_t)(2.5 * interval * 1e9)) {
                            atomic_fetch_add_explicit(&c->est_dropped,
                                (uint64_t)((gap / (int64_t)(interval * 1e9)) - 1),
                                memory_order_relaxed);
                        }
                        /* latency: consume time - frame PTS (same CLOCK_MONOTONIC domain) */
                        double lat = (mono_now_s() - (double)pts / 1e9) * 1e3; /* ms */
                        if (lat > 0 && lat < 5000) {
                            c->lat_samples++;
                            c->lat_sum_ns += lat; /* ms */
                        }
                    }
                    atomic_store_explicit(&c->last_pts_ns, pts, memory_order_release);
                }
                atomic_fetch_add_explicit(&c->frames, 1, memory_order_relaxed);
                gst_buffer_unmap(buf, &info);
            }
        }
        gst_sample_unref(sample);

        if (c->max_frames > 0 &&
            atomic_load_explicit(&c->frames, memory_order_relaxed) >= (uint64_t)c->max_frames)
            c->running = 0;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */

static int parse_args(Check *c, int argc, char **argv)
{
    memset(c, 0, sizeof *c);
    c->out_w = 1920;
    c->out_h = 1080;
    c->fps = 60;
    c->duration_s = 5.0;
    c->max_frames = 0;
    c->last_pts_ns = -1;
    c->px_min = 255.0;
    c->running = 1;
    /* pick default mode from env */
    const char *e_fd = getenv("PW_FD");
    const char *e_node = getenv("PW_NODE_ID");
    if (e_fd && e_node) {
        c->mode = 1;
        c->pw_fd = atoi(e_fd);
        c->pw_node = atoi(e_node);
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--testsrc")) { c->mode = 0; }
        else if (!strcmp(argv[i], "--portal")) { c->mode = 1; }
        else if (!strcmp(argv[i], "--fd") && i + 1 < argc) c->pw_fd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--node") && i + 1 < argc) c->pw_node = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) c->out_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) c->out_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) c->fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) c->duration_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) c->max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) snprintf(c->save_path, sizeof c->save_path, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            fprintf(stderr,
                "usage: %s [--testsrc | --portal [--fd N --node ID]]\n"
                "                [--width W --height H --fps N --duration S --frames N]\n"
                "                [--save OUT.ppm]\n",
                argv[0]);
            return -2;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    if (c->mode == 1 && c->pw_fd < 0) {
        fprintf(stderr, "--portal requires PW_FD/--fd and PW_NODE_ID/--node\n");
        return -1;
    }
    return 0;
}

static void build_pipeline(Check *c, GError **err)
{
    char desc[1024];
    if (c->mode == 0) {
        /* this videotestsrc build has no size/rate properties; drive the
         * format+size+rate from the caps filter instead. */
        snprintf(desc, sizeof desc,
                 "videotestsrc is-live=true pattern=smpte "
                 "! videoconvert ! videoscale "
                 "! video/x-raw,width=%d,height=%d,framerate=%d/1,format=RGB "
                 "! appsink name=sink max-buffers=4 drop=true sync=false",
                 c->out_w, c->out_h, c->fps, c->out_w, c->out_h);
        snprintf(c->source_desc, sizeof c->source_desc,
                 "videotestsrc(SMPTE %dx%d @ %dfps)", c->out_w, c->out_h, c->fps);
    } else {
        snprintf(desc, sizeof desc,
                 "pipewiresrc fd=%d path=%d "
                 "! videoconvert ! videoscale "
                 "! video/x-raw,width=%d,height=%d,format=RGB "
                 "! appsink name=sink max-buffers=4 drop=true sync=false",
                 c->pw_fd, c->pw_node, c->out_w, c->out_h);
        snprintf(c->source_desc, sizeof c->source_desc,
                 "portal(fd=%d,node=%d)", c->pw_fd, c->pw_node);
    }
    c->pipe = gst_parse_launch(desc, err);
}

int main(int argc, char **argv)
{
    Check c;
    int rc = parse_args(&c, argc, argv);
    if (rc == -2) return 0;
    if (rc != 0) return 2;
    g_chk = &c;

    struct sigaction sa = { .sa_handler = on_signal };
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    gst_init(&argc, &argv);

    GError *err = NULL;
    build_pipeline(&c, &err);
    if (!c.pipe) {
        printf("CAPTURE FAIL (parse: %s)\n", err ? err->message : "?");
        if (err) g_error_free(err);
        return 1;
    }
    c.sink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(c.pipe), "sink"));
    if (!GST_IS_APP_SINK(c.sink)) {
        printf("CAPTURE FAIL (appsink not found)\n");
        return 1;
    }

    c.t0 = now_real();
    c.running = 1;

    GstStateChangeReturn res = gst_element_set_state(GST_ELEMENT(c.pipe), GST_STATE_PLAYING);
    GstState state = GST_STATE_VOID_PENDING;
    res = gst_element_get_state(GST_ELEMENT(c.pipe), &state, 0, 5 * GST_SECOND);
    GstBus *bus = gst_element_get_bus(GST_ELEMENT(c.pipe));
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, 0, GST_MESSAGE_ERROR);
    if (msg) {
        GError *e2 = NULL;
        gst_message_parse_error(msg, &e2, NULL);
        printf("CAPTURE FAIL (pipeline error: %s)\n", e2 ? e2->message : "?");
        if (e2) g_error_free(e2);
        gst_message_unref(msg);
        return 1;
    }
    if (res == GST_STATE_CHANGE_FAILURE ||
        (state != GST_STATE_PLAYING && state != GST_STATE_PAUSED)) {
        /* surface async errors that arrive just after the state query */
        GstMessage *emsg = gst_bus_timed_pop_filtered(bus, 300 * GST_MSECOND,
                                                       GST_MESSAGE_ERROR);
        if (emsg) {
            GError *e3 = NULL;
            gst_message_parse_error(emsg, &e3, NULL);
            printf("CAPTURE FAIL (pipeline error: %s)\n", e3 ? e3->message : "?");
            if (e3) g_error_free(e3);
            gst_message_unref(emsg);
        } else {
            printf("CAPTURE FAIL (state res=%d state=%d)\n", (int)res, (int)state);
        }
        return 1;
    }
    c.t_playing = now_real();

    pthread_create(&c.pull_th, NULL, pull_thread, &c);

    /* run for the requested window */
    double deadline = c.t0 + c.duration_s;
    while (c.running && now_real() < deadline)
        g_usleep(50 * 1000);
    c.running = 0;

    double run_s = now_real() - c.t_playing;
    if (run_s <= 0) run_s = c.duration_s;

    uint64_t frames = atomic_load_explicit(&c.frames, memory_order_relaxed);
    uint64_t mismatch = atomic_load_explicit(&c.size_mismatch, memory_order_relaxed);
    uint64_t dropped = atomic_load_explicit(&c.est_dropped, memory_order_relaxed);

    c.running = 0;
    gst_element_set_state(GST_ELEMENT(c.pipe), GST_STATE_NULL); /* unblock pull */
    pthread_join(c.pull_th, NULL);

    double fps = (double)frames / run_s;
    double first_ms = (c.t_first > 0) ? (c.t_first - c.t_playing) * 1e3 : -1.0;
    double lat_ms = c.lat_samples ? c.lat_sum_ns / (double)c.lat_samples : -1.0;
    double px_mean = c.px_samples ? c.px_sum / (double)c.px_samples : -1.0;

    /* ---- report ---- */
    if (frames == 0) {
        printf("CAPTURE FAIL (0 frames in %.1fs)\n", run_s);
        printf("source:      %s\n", c.source_desc);
        printf("state:       %s\n", state == GST_STATE_PLAYING ? "PLAYING" : "not-PLAYING");
        if (c.mode == 1)
            printf("hint:        screencast is damage-driven; the surface must be "
                   "actively rendering (a static screen yields no frames)\n");
    } else {
        printf("CAPTURE OK\n");
        printf("source:      %s\n", c.source_desc);
        printf("resolution:  %dx%d\n", c.out_w, c.out_h);
        printf("format:      %s\n", c.fmt[0] ? c.fmt : "RGB");
        printf("fps:         %.1f\n", fps);
        printf("frames:      %" PRIu64 "\n", frames);
        printf("dropped:     %" PRIu64 " (est. from PTS gaps)\n", dropped);
        printf("first_frame_ms: %.1f\n", first_ms);
        if (lat_ms >= 0) printf("latency_ms:  %.1f\n", lat_ms);
        else printf("latency_ms:  n/a (no absolute-clock PTS)\n");
        printf("pixels:      min=%.0f mean=%.1f max=%.0f\n", c.px_min, px_mean, c.px_max);
        if (mismatch)
            printf("warn:        %" PRIu64 " frame(s) had unexpected buffer size\n", mismatch);
        printf("state:       PLAYING\n");
        if (c.save_path[0] && c.last_frame && c.save_w > 0) {
            FILE *pf = fopen(c.save_path, "wb");
            if (pf) {
                fprintf(pf, "P6\n%d %d\n255\n", c.save_w, c.save_h);
                size_t n3 = (size_t)c.save_w * c.save_h * 3;
                fwrite(c.last_frame, 1, n3, pf);
                fclose(pf);
                printf("saved:       %s (%dx%d)\n", c.save_path, c.save_w, c.save_h);
            } else {
                printf("warn:        could not open %s for writing\n", c.save_path);
            }
        }
    }

    if (c.last_frame) free(c.last_frame);
    gst_object_unref(c.sink);
    gst_object_unref(c.pipe);
    return frames == 0 ? 1 : 0;
}
