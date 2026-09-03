// gst_capture.cpp — GStreamer X11 capture (port of native/capture_daemon.c).

#include "gst_capture.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace anyplay {

struct GstCapture::Private {
    Config cfg;
    FrameCallback cb;
    DropCallback drop_cb;
    GstElement *pipeline = nullptr;
    GstElement *sink = nullptr;
    std::thread thread;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> playing{false};
    std::once_flag init_flag;
};

// ---------------------------------------------------------------------------
// Pipeline construction — same element chain and caps as the C daemon:
//   ximagesrc ! video/x-raw,framerate=fps/1 ! videorate ! videoscale
//   ! videoconvert ! video/x-raw,width=W,height=H,format=RGB
//   ! appsink name=sink max-buffers=2 drop=true
// ---------------------------------------------------------------------------

namespace {

void parse_region(const std::string &region, int &x, int &y, int &w, int &h) {
    int n = std::sscanf(region.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h);
    if (n != 4 || w <= 0 || h <= 0) {
        throw std::invalid_argument("bad --region (want x,y,w,h): " + region);
    }
}

std::string build_pipeline(const GstCapture::Config &cfg, int &out_w,
                           int &out_h) {
    int x, y, w, h;
    parse_region(cfg.region, x, y, w, h);
    out_w = static_cast<int>(cfg.scale_w > 0 ? cfg.scale_w : w);
    out_h = static_cast<int>(cfg.scale_h > 0 ? cfg.scale_h : h);

    std::ostringstream oss;
    oss << "ximagesrc display-name=" << cfg.display
        << " startx=" << x << " starty=" << y
        << " endx=" << (x + w - 1) << " endy=" << (y + h - 1)
        << " do-timestamp=true use-damage=false"
        << " ! video/x-raw,framerate=" << cfg.fps << "/1"
        << " ! videorate ! videoscale ! videoconvert"
        << " ! video/x-raw,width=" << out_w << ",height=" << out_h
        << ",format=RGB"
        << " ! appsink name=sink max-buffers=2 drop=true";
    return oss.str();
}

}  // namespace

// ---------------------------------------------------------------------------

GstCapture::GstCapture(Config cfg, FrameCallback cb)
    : p_(std::make_unique<Private>()) {
    p_->cfg = std::move(cfg);
    p_->cb = std::move(cb);
}

GstCapture::~GstCapture() { stop(); }

void GstCapture::set_callbacks(FrameCallback cb, DropCallback drop_cb) {
    p_->cb = std::move(cb);
    p_->drop_cb = std::move(drop_cb);
}

std::string GstCapture::start() {
    const auto &cfg = p_->cfg;

    std::call_once(p_->init_flag, [] { gst_init(nullptr, nullptr); });

    int out_w = 0, out_h = 0;
    std::string desc;
    try {
        desc = build_pipeline(cfg, out_w, out_h);
    } catch (const std::exception &e) {
        return e.what();
    }

    GError *err = nullptr;
    p_->pipeline = gst_parse_launch(desc.c_str(), &err);
    if (err != nullptr) {
        std::string msg = "gst_parse_launch: ";
        msg += err->message;
        g_error_free(err);
        return msg;
    }
    if (p_->pipeline == nullptr) {
        return "gst_parse_launch: no pipeline";
    }

    p_->sink = static_cast<GstElement *>(
        gst_bin_get_by_name(GST_BIN(p_->pipeline), "sink"));
    if (p_->sink == nullptr) {
        gst_object_unref(p_->pipeline);
        p_->pipeline = nullptr;
        return "appsink 'sink' not found in pipeline";
    }

    GstStateChangeReturn ret =
        gst_element_set_state(p_->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        gst_object_unref(p_->sink);
        p_->sink = nullptr;
        gst_element_set_state(p_->pipeline, GST_STATE_NULL);
        gst_object_unref(p_->pipeline);
        p_->pipeline = nullptr;
        return "pipeline failed to go to PLAYING";
    }

    p_->playing.store(true, std::memory_order_release);
    p_->thread = std::thread([this] { this->pull_loop(); });
    return "";
}

void GstCapture::stop() {
    auto &st = p_;
    if (st == nullptr) return;

    st->stop_flag.store(true, std::memory_order_release);
    if (st->thread.joinable()) {
        st->thread.join();
    }
    if (st->pipeline != nullptr) {
        gst_element_set_state(st->pipeline, GST_STATE_NULL);
        if (st->sink != nullptr) {
            gst_object_unref(st->sink);
            st->sink = nullptr;
        }
        gst_object_unref(st->pipeline);
        st->pipeline = nullptr;
    }
    st->playing.store(false, std::memory_order_release);
}

bool GstCapture::running() const {
    return p_ != nullptr && p_->playing.load(std::memory_order_acquire);
}

void GstCapture::pull_loop() {
    auto &sink = p_->sink;

    while (!p_->stop_flag.load(std::memory_order_acquire)) {
        const auto t0 = std::chrono::steady_clock::now();
        GstSample *sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(sink),
            static_cast<GstClockTime>(500 * GST_MSECOND));
        const auto wait_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0)
                .count());

        if (sample == nullptr) continue;  // timeout or EOS

        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstMapInfo mi;
        const bool have_buffer =
            buffer != nullptr && gst_buffer_map(buffer, &mi, GST_MAP_READ);

        if (have_buffer) {
            // The pipeline caps guarantee the buffer size; an empty or
            // unmappable buffer counts as a drop, like the C daemon.
            if (p_->cb) p_->cb(mi.data, wait_ns);
            gst_buffer_unmap(buffer, &mi);
        } else if (p_->drop_cb) {
            p_->drop_cb();
        }
        gst_sample_unref(sample);
    }
}

}  // namespace anyplay
