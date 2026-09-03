// frame_ring.h — POSIX shared-memory frame ring (RAII).
//
// Port of the shm ring in native/capture_daemon.c. The byte layout is
// owned by anyplay/shm_protocol.h; this class owns the file, mapping,
// lifetime, and the seqlock publish protocol.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "anyplay/shm_protocol.h"

namespace anyplay {

class FrameRing {
  public:
    // Creates (or reuses) the shm file, maps it, and writes the header.
    // Throws std::runtime_error on failure.
    FrameRing(std::string path, std::uint32_t width, std::uint32_t height,
              std::uint32_t n_slots);

    FrameRing(const FrameRing &) = delete;
    FrameRing &operator=(const FrameRing &) = delete;
    FrameRing(FrameRing &&) = delete;
    FrameRing &operator=(FrameRing &&) = delete;

    ~FrameRing();

    // Publish one RGB frame using the seqlock protocol.
    //   data     : pointer to width*height*3 bytes of RGB888
    //   frame_id : monotonically increasing id assigned by the caller
    //   wait_ns  : how long the producer waited for this sample (ns)
    // Drops the frame if a consumer is not holding the ring open
    // (SIGBUS on header read) or if the slot is being written.
    void publish(const void *data, std::uint64_t frame_id,
                 std::uint64_t wait_ns);

    // Publish a frame whose source rows may be padded (src_stride bytes
    // per row, as produced by PipeWire/GStreamer RGB buffers) instead of
    // tightly packed. Rows are packed down to width*3 in the slot. Same
    // drop semantics as publish(). Mirrors the C daemon's publish_frame.
    void publish_strided(const void *data, std::size_t src_stride,
                         std::uint64_t frame_id, std::uint64_t wait_ns = 0);

    // Mark the ring as not-alive (graceful shutdown).
    void mark_dead();

    const std::string &path() const { return path_; }

    std::uint64_t frames_published() const {
        return hdr_->frames.load(std::memory_order_acquire);
    }
    std::uint64_t drops() const {
        return hdr_->drops.load(std::memory_order_acquire);
    }

  private:
    std::string path_;
    int fd_ = -1;
    void *mem_ = nullptr;
    std::size_t map_size_ = 0;
    std::size_t stride_ = 0;
    CapHeader *hdr_ = nullptr;

    // Returns slot pointer + metadata for slot index s (0-based).
    void *slot_data(std::size_t s) const;
    CapSlotMeta *slot_meta(std::size_t s) const;

    // Returns false if the consumer is not mapped (SIGBUS sentinel).
    bool consumer_present() const;

    // Core seqlock-guarded copy of `data` (src_stride bytes per source
    // row) into the target slot, packed to width*3. Shared by the flat
    // and strided entry points.
    void publish_impl(const void *data, std::size_t src_stride,
                      std::uint64_t frame_id, std::uint64_t wait_ns);
};

}  // namespace anyplay
