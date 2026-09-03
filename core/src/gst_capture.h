// gst_capture.h — GStreamer X11 capture source (port of the pipeline
// half of native/capture_daemon.c). PIMPL keeps GStreamer types out of
// the header.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace anyplay {

class GstCapture {
  public:
    // Invoked on the pull thread for each valid frame.
    //   data: RGB pixel buffer (exactly slot_bytes of the ring)
    //   wait_ns: producer wait time for this sample (ns)
    using FrameCallback = std::function<void(const void *data,
                                             std::uint64_t wait_ns)>;
    // Invoked when a sample is dropped (undersized buffer), matching
    // the C daemon's drops counter.
    using DropCallback = std::function<void()>;

    struct Config {
        // Region "x,y,w,h" — captured area in display pixels.
        std::string region = "0,0,640,480";
        std::string display = ":0.0";
        std::uint32_t fps = 60;      // advisory (videorate passthrough)
        std::uint32_t scale_w = 0;   // output (ring) dimensions, 0 = region
        std::uint32_t scale_h = 0;
    };

    // cb may be empty (no frames delivered; the daemon still runs).
    explicit GstCapture(Config cfg, FrameCallback cb = nullptr);
    ~GstCapture();

    GstCapture(const GstCapture &) = delete;
    GstCapture &operator=(const GstCapture &) = delete;

    void set_callbacks(FrameCallback cb, DropCallback drop_cb);

    // Returns "" on success, error message on failure.
    std::string start();
    void stop();
    bool running() const;

  private:
    struct Private;
    std::unique_ptr<Private> p_;
    void pull_loop();
};

}  // namespace anyplay
