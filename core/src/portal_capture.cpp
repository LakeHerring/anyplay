// portal_capture.cpp — PipeWire (xdg-desktop-portal) capture source.
//
// Port of native/portal_capture.c. Pipeline layout and caps match the C
// daemon exactly:
//
//   production:
//     pipewiresrc fd=<PW_FD> path=<PW_NODE_ID>
//       ! videoconvert [! videoscale ! video/x-raw,RGB,WxH]
//       ! video/x-raw,format=RGB[,width=W,height=H]
//       ! appsink name=sink max-buffers=8 drop=true sync=false
//
//   --test (no live window, CI):
//     videotestsrc is-live=true pattern=smpte ! videoconvert
//       ! video/x-raw,format=RGB ! appsink name=sink max-buffers=8
//       drop=true sync=false
//
// The inherited PipeWire fd (PW_FD) is held open for the whole process
// lifetime — that keeps the portal capture session alive.

#include "portal_capture.h"

#include <anyplay/shm_protocol.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>

namespace anyplay {

namespace {

// Pull-loop poll timeout (ms) — matches the C daemon.
constexpr auto kPollTimeout = std::chrono::milliseconds(150);
// Detect: max time (ms) to wait for the first frame before retrying.
constexpr int kDetectTimeoutMs = 10000;
// State-change wait when bringing the pipeline to PLAYING (ns).
constexpr GstClockTime kStateTimeout = 5 * GST_SECOND;
// Bus message mask meaning "the stream is ending".
constexpr GstMessageType kStreamEndMsgs = static_cast<GstMessageType>(
    GST_MESSAGE_EOS | GST_MESSAGE_ERROR);

std::string build_desc(const PortalCapture::Config &cfg) {
    std::ostringstream oss;
    if (cfg.test) {
        oss << "videotestsrc is-live=true pattern=smpte";
    } else {
        // fd + node id go in the launch string exactly like the C daemon
        // (`pipewiresrc fd=%d path=%u`).
        oss << "pipewiresrc fd=" << cfg.pw_fd << " path=" << cfg.node_id;
    }

    if (cfg.force_w > 0 && cfg.force_h > 0) {
        oss << " ! videoconvert ! videoscale"
            << " ! video/x-raw,format=RGB,width=" << cfg.force_w
            << ",height=" << cfg.force_h;
    } else {
        oss << " ! videoconvert ! video/x-raw,format=RGB";
    }
    oss << " ! appsink name=sink max-buffers=8 drop=true sync=false";
    return oss.str();
}

}  // namespace

struct PortalCapture::Private {
    Config cfg;
    GstElement *pipeline = nullptr;
    GstElement *sink = nullptr;
    std::atomic<bool> stop_flag{false};
    // Effective output size (forced if set, else detected); known after
    // detect(). Used by stream() to drop mid-session geometry changes.
    std::uint32_t out_w = 0;
    std::uint32_t out_h = 0;

    void teardown() {
        if (pipeline != nullptr) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            if (sink != nullptr) {
                gst_object_unref(sink);
                sink = nullptr;
            }
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
    }
};

PortalCapture::PortalCapture(Config cfg)
    : p_(std::make_unique<Private>()) {
    p_->cfg = std::move(cfg);
}

PortalCapture::~PortalCapture() { stop(); }

std::string PortalCapture::build() {
    gst_init(nullptr, nullptr);  // idempotent

    // Idempotent: tear down any previous pipeline first (reconnect path).
    // NOTE: out_w/out_h are NOT reset here — they are learned in detect() and
    // must persist across a reconnect (death/respawn), which rebuilds without
    // re-detecting. The C daemon keeps pc.out_w/pc.out_h alive the same way.
    p_->teardown();
    p_->stop_flag.store(false, std::memory_order_relaxed);

    const std::string desc = build_desc(p_->cfg);
    std::fprintf(stderr, "[portal-capture] pipeline: %s\n", desc.c_str());

    GError *err = nullptr;
    GstElement *pipeline = gst_parse_launch(desc.c_str(), &err);
    if (pipeline == nullptr) {
        std::string msg = "gst_parse_launch: ";
        msg += err != nullptr ? err->message : "unknown";
        if (err != nullptr) g_error_free(err);
        return msg;
    }

    GstElement *sink = static_cast<GstElement *>(
        gst_bin_get_by_name(GST_BIN(pipeline), "sink"));
    if (sink == nullptr) {
        std::fprintf(stderr, "[portal-capture] appsink not found\n");
        gst_object_unref(pipeline);
        return "appsink 'sink' not found in pipeline";
    }

    p_->pipeline = pipeline;
    p_->sink = sink;

    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE) {
        std::fprintf(stderr, "[portal-capture] set_state(PLAYING) failed\n");
    }

    GstStateChangeReturn scr =
        gst_element_get_state(pipeline, nullptr, nullptr, kStateTimeout);
    if (scr == GST_STATE_CHANGE_FAILURE) {
        std::fprintf(stderr,
                     "[portal-capture] pipeline FAILED to reach PLAYING\n");
        p_->teardown();
        return "pipeline failed to reach PLAYING";
    }
    std::fprintf(stderr, "[portal-capture] pipeline PLAYING\n");
    return "";
}

int PortalCapture::detect(std::uint32_t &w, std::uint32_t &h) {
    if (p_->sink == nullptr) return -1;
    if (p_->cfg.force_w > 0 && p_->cfg.force_h > 0) {
        // Forced: the pipeline already scales to this; no need to pull.
        p_->out_w = p_->cfg.force_w;
        p_->out_h = p_->cfg.force_h;
        w = p_->out_w;
        h = p_->out_h;
        std::fprintf(stderr, "[portal-capture] forced %ux%u RGB\n", w, h);
        return 0;
    }

    auto *sink = GST_APP_SINK(p_->sink);
    auto *bus = gst_element_get_bus(p_->pipeline);
    const auto t0 = std::chrono::steady_clock::now();

    for (;;) {
        if (p_->stop_flag.load(std::memory_order_acquire)) return -1;

        GstSample *sample = gst_app_sink_try_pull_sample(
            sink, static_cast<GstClockTime>(500 * GST_MSECOND));
        if (sample != nullptr) {
            GstCaps *caps = gst_sample_get_caps(sample);
            int cw = 0, ch = 0;
            if (caps != nullptr) {
                GstStructure *s = gst_caps_get_structure(caps, 0);
                if (s != nullptr) {
                    int sv = 0;
                    if (gst_structure_get_int(s, "width", &sv)) cw = sv;
                    if (gst_structure_get_int(s, "height", &sv)) ch = sv;
                }
            }
            gst_sample_unref(sample);
            if (cw > 0 && ch > 0) {
                p_->out_w = static_cast<std::uint32_t>(cw);
                p_->out_h = static_cast<std::uint32_t>(ch);
                std::fprintf(stderr,
                             "[portal-capture] detected %dx%d RGB\n", cw, ch);
                w = p_->out_w;
                h = p_->out_h;
                return 0;
            }
        }

        // Stream ended (EOS/ERROR) before a frame — not retryable.
        GstMessage *msg = gst_bus_pop_filtered(bus, kStreamEndMsgs);
        if (msg != nullptr) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError *e = nullptr;
                gchar *dbg = nullptr;
                gst_message_parse_error(msg, &e, &dbg);
                std::fprintf(stderr,
                             "[portal-capture] detect: pipeline error: %s\n",
                             e != nullptr ? e->message : "?");
                if (e != nullptr) g_error_free(e);
                if (dbg != nullptr) g_free(dbg);
            }
            gst_message_unref(msg);
            return -1;
        }

        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (elapsed_ms > kDetectTimeoutMs) {
            std::fprintf(stderr,
                         "[portal-capture] detect: no frame after %ld ms\n",
                         static_cast<long>(elapsed_ms));
            return -2;
        }
    }
}

int PortalCapture::stream(FrameCallback cb, DropCallback drop_cb,
                          const std::function<bool()> &should_stop) {
    if (p_->sink == nullptr) return 0;

    auto *sink = GST_APP_SINK(p_->sink);
    auto *bus = gst_element_get_bus(p_->pipeline);
    std::uint64_t frames = 0;
    std::uint64_t drops = 0;

    for (;;) {
        if (p_->stop_flag.load(std::memory_order_acquire)) break;
        if (should_stop && should_stop()) break;

        const auto t0 = std::chrono::steady_clock::now();
        GstSample *sample = gst_app_sink_try_pull_sample(
            sink, static_cast<GstClockTime>(kPollTimeout.count() * GST_MSECOND));
        const std::uint64_t wait_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count());

        if (sample != nullptr) {
            GstBuffer *buf = gst_sample_get_buffer(sample);
            GstMapInfo info;
            const bool have =
                buf != nullptr && gst_buffer_map(buf, &info, GST_MAP_READ);
            if (have) {
                // Defaults: the effective size, tightly-packed stride.
                int cw = static_cast<int>(p_->out_w);
                int ch = static_cast<int>(p_->out_h);
                std::size_t stride = static_cast<std::size_t>(p_->out_w) * 3;
                GstCaps *caps = gst_sample_get_caps(sample);
                if (caps != nullptr) {
                    GstStructure *s = gst_caps_get_structure(caps, 0);
                    if (s != nullptr) {
                        int sv = 0;
                        if (gst_structure_get_int(s, "width", &sv)) cw = sv;
                        if (gst_structure_get_int(s, "height", &sv)) ch = sv;
                        if (gst_structure_get_int(s, "stride", &sv))
                            stride = static_cast<std::size_t>(sv);
                    }
                }

                if (static_cast<std::uint32_t>(cw) != p_->out_w ||
                    static_cast<std::uint32_t>(ch) != p_->out_h) {
                    // Window resized mid-session; the fixed-size ring can't
                    // follow — drop, like the C daemon.
                    std::fprintf(stderr,
                                 "[portal-capture] geometry changed to "
                                 "%dx%d (expected %ux%u); dropping\n",
                                 cw, ch, p_->out_w, p_->out_h);
                    if (drop_cb) drop_cb();
                } else if (info.size >=
                           static_cast<std::size_t>(ch) * stride) {
                    if (cb) cb(info.data, stride, wait_ns);
                    ++frames;
                } else if (drop_cb) {
                    drop_cb();
                }
                gst_buffer_unmap(buf, &info);
            } else if (drop_cb) {
                drop_cb();
            }
            gst_sample_unref(sample);
        } else {
            // Timeout: only treat as a hard stop if EOS/ERROR is queued.
            GstMessage *msg = gst_bus_pop_filtered(bus, kStreamEndMsgs);
            if (msg != nullptr) {
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                    GError *e = nullptr;
                    gchar *dbg = nullptr;
                    gst_message_parse_error(msg, &e, &dbg);
                    std::fprintf(stderr,
                                 "[portal-capture] pipeline error "
                                 "(stream ending): %s\n",
                                 e != nullptr ? e->message : "?");
                    if (e != nullptr) g_error_free(e);
                    if (dbg != nullptr) g_free(dbg);
                }
                gst_message_unref(msg);
                break;
            }
        }
    }

    const bool quit =
        p_->stop_flag.load(std::memory_order_acquire) ||
        (should_stop && should_stop());
    const int rc = (frames > 0 || quit) ? 0 : 1;
    std::fprintf(stderr,
                 "[portal-capture] stream exited: %llu frames, %llu drops\n",
                 static_cast<unsigned long long>(frames),
                 static_cast<unsigned long long>(drops));
    return rc;
}

void PortalCapture::stop() {
    if (p_ == nullptr) return;
    p_->stop_flag.store(true, std::memory_order_release);
    p_->teardown();
}

bool PortalCapture::running() const {
    return p_ != nullptr && p_->pipeline != nullptr &&
           !p_->stop_flag.load(std::memory_order_acquire);
}

}  // namespace anyplay
