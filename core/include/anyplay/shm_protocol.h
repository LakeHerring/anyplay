// anyplay/shm_protocol.h — shared-memory capture ring protocol.
//
// THIS IS THE CONTRACT between the C++ core (producer) and the Python
// client (anyplay/capture/native_capture.py, consumer).
// The byte layout must stay identical to the original C daemon
// (native/capture_daemon.c) so the existing Python client works
// unchanged against the C++ binary.
//
// Layout (little-endian), total = HDR_SIZE + n_slots * slot_stride:
//
//   offset 0   CapHeader (64 B; 56 B used, 8 B reserved padding)
//      u32 magic, version, width, height, slot_bytes, n_slots, alive, pad
//      u64 idx     (atomic, release: newest frame id + 1)
//      u64 frames  (total frames published)
//      u64 drops   (overrun / no-buffer drops)
//
//   offset 64  CapSlot x n_slots, stride = SLOT_META + slot_bytes
//      u32 seq          (atomic; even = stable, odd = being written)
//      u32 pad
//      u64 frame_id     (atomic)
//      u64 ts_ns        (CLOCK_MONOTONIC at publish; in previously
//                        unused bytes — old clients never read it)
//      u64 wait_ns      (producer wait for the sample, ns; same)
//      u8  data[slot_bytes]
//
// Consumer protocol: read hdr.idx (acquire), s = (idx-1) % n_slots,
// retry the slot read while seq is odd or changes pre/post read.
//
// Control socket (AF_UNIX), one byte per command:
//   'q'  quit cleanly  -> replies 'Q'
//   's'  stats         -> replies "frames=N fps=X drops=N\n"

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace anyplay {

constexpr std::uint32_t SDAI_MAGIC = 0x49445321u;  // "SDAI!"
constexpr std::uint32_t SDAI_VERSION = 1u;
constexpr std::size_t HDR_SIZE = 64;   // header region size (bytes)
constexpr std::size_t SLOT_META = 32;  // per-slot metadata size (bytes)

// Byte-identical to `struct CapHeader` in native/capture_daemon.c.
// The Python client reads the first 32 bytes as <8I and the u64
// triplets at offset 32 as <3Q.
struct CapHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t slot_bytes;
    std::uint32_t n_slots;
    std::atomic<std::uint32_t> alive;  // align 4, offset 24 (C layout)
    std::atomic<std::uint32_t> pad;    // offset 28
    std::atomic<std::uint64_t> idx;
    std::atomic<std::uint64_t> frames;
    std::atomic<std::uint64_t> drops;
};
static_assert(sizeof(CapHeader) == 56,
              "CapHeader must match the Python client layout");

// Byte-identical to `struct CapSlot`'s metadata region in the C daemon.
// Offsets 16..31 were unused padding in the C version; the C++ core
// publishes timing stamps there (ts_ns, wait_ns). Old clients ignore
// them; new Python code can read them for latency analysis.
struct CapSlotMeta {
    std::atomic<std::uint32_t> seq;     // offset 0
    std::atomic<std::uint32_t> pad;     // offset 4
    std::atomic<std::uint64_t> frame_id;  // offset 8
    std::atomic<std::uint64_t> ts_ns;     // offset 16 (C: unused padding)
    std::atomic<std::uint64_t> wait_ns;   // offset 24 (C: unused padding)
};
static_assert(sizeof(CapSlotMeta) == SLOT_META,
              "CapSlotMeta must match the Python client layout");

inline std::uint64_t now_mono_ns() {
    auto tp = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp).count());
}

}  // namespace anyplay
