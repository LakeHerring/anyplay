// portal_capture.h — PipeWire (xdg-desktop-portal) capture source.
//
// Port of native/portal_capture.c. Unlike GstCapture (ximagesrc, known
// size up front), the portal path must first pull a single frame to learn
// the window's native size before the shm ring can be allocated. This
// class therefore exposes a two-phase API:
//
//   build()   -- construct the pipewiresrc / videotestsrc pipeline, PLAYING
//   detect()  -- pull the first frame, report its WxH
//   stream()  -- pull the remaining frames, invoking cb per frame
//
// The pipeline stays PLAYING across detect() and stream() (no teardown in
// between). If the PipeWire stream ends with 0 frames (the game
// disconnected, e.g. death/respawn) stream() returns 1 so the caller can
// rebuild via build(). If frames flowed and then the stream ends, it
// returns 0 (session over).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace anyplay {

class PortalCapture {
  public:
    struct Config {
        int pw_fd = -1;             // inherited PipeWire fd (PW_FD env)
        std::uint32_t node_id = 0;  // portal node id (PW_NODE_ID env)
        std::uint32_t force_w = 0;  // 0 = auto-detect
        std::uint32_t force_h = 0;  // 0 = auto-detect
        bool test = false;          // videotestsrc mode (no live window)
    };

    // Per-frame callback: `data` is the mapped sample, `stride` is the
    // source row stride in bytes (may exceed width*3), `wait_ns` is the
    // producer wait for the sample.
    using FrameCallback = std::function<void(const void *data,
                                             std::size_t stride,
                                             std::uint64_t wait_ns)>;
    using DropCallback = std::function<void()>;

    explicit PortalCapture(Config cfg);
    ~PortalCapture();

    PortalCapture(const PortalCapture &) = delete;
    PortalCapture &operator=(const PortalCapture &) = delete;

    // Build the pipeline and bring it to PLAYING (waits up to 5 s for the
    // state change). Any previous pipeline is torn down first, so this is
    // safe to call repeatedly (reconnect). Returns "" on success, else an
    // error string.
    std::string build();

    // Pull the first frame to learn the native size. The pipeline must
    // already be PLAYING (from build()). On success the effective output
    // size (forced if set, else detected) is stored and reported via w/h.
    // Returns:
    //    0  -- got a frame; w/h are set
    //   -1  -- stream ended / quit before a frame (not retryable)
    //   -2  -- no frame within the detect timeout (retry)
    int detect(std::uint32_t &w, std::uint32_t &h);

    // Stream frames on the PLAYING pipeline until should_stop() is true
    // or the stream ends. cb is invoked per frame. Returns:
    //    0 -- quit requested, or at least one frame flowed before the
    //         stream ended (session over)
    //    1 -- stream ended with 0 frames (caller may reconnect)
    int stream(FrameCallback cb, DropCallback drop_cb,
               const std::function<bool()> &should_stop);

    // Tear down the pipeline (set NULL, unref). Idempotent.
    void stop();

    // True if a pipeline is currently built and PLAYING.
    bool running() const;

  private:
    struct Private;
    std::unique_ptr<Private> p_;
};

}  // namespace anyplay
