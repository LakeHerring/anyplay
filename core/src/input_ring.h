// input_ring.h — POSIX shared-memory input-event ring (RAII).
//
// Sibling of frame_ring.h: same shm-file lifecycle and seqlock publish
// protocol, but the payload is a batch of kernel input events (see
// anyplay/input_ring_protocol.h).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "anyplay/input_ring_protocol.h"

namespace anyplay {

class InputRing {
  public:
    // Creates (or reuses) the shm file, maps it, and writes the header.
    // Throws std::runtime_error on failure.
    InputRing(std::string path, std::uint32_t n_slots);

    InputRing(const InputRing &) = delete;
    InputRing &operator=(const InputRing &) = delete;
    InputRing(InputRing &&) = delete;
    InputRing &operator=(InputRing &&) = delete;

    ~InputRing();

    // Publish one batch of events from one device (the ring's own
    // InpEvent layout, which is byte-identical to kernel
    // struct input_event — pass a buffer of either). dev_id is the
    // device's index in CLI order (keyboard first, then pointer) and
    // applies to every event in the batch. Thread-safe: multiple
    // producers (one per evdev device) may call concurrently; the slot
    // index is claimed with an atomic CAS. Drops the batch if
    // count > INP_EVENTS_PER_SLOT or no consumer is present. Publishing is safe
    // even before a consumer has mapped the ring (the daemon's own
    // mapping stays valid while the file exists); if the file is
    // destroyed under the mapping (e.g. a new daemon instance reopened
    // the path with O_TRUNC), the one-shot SIGBUS sentinel terminates
    // this process — the same stale-instance protection as the frame
    // ring.
    void publish(std::uint32_t dev_id, const void *events,
                 std::uint32_t count);

    // Mark the ring as not-alive (graceful shutdown).
    void mark_dead();

    const std::string &path() const { return path_; }
    std::uint32_t n_slots() const {
        return hdr_->n_slots;
    }

    std::uint64_t slots_published() const {
        return hdr_->idx.load(std::memory_order_acquire);
    }
    std::uint64_t events_published() const {
        return hdr_->events.load(std::memory_order_acquire);
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
    InpHeader *hdr_ = nullptr;

    void *slot_data(std::size_t s) const;
    InpSlotMeta *slot_meta(std::size_t s) const;

    // False if the consumer is not mapped (SIGBUS sentinel).
    bool consumer_present() const;
};

}  // namespace anyplay
