// anyplay/input_ring_protocol.h — shared-memory input-event ring protocol.
//
// Contract between the C++ core (EvdevReader producer) and the Python
// client (anyplay/capture/input_ring.py consumer). This is a NEW ring,
// separate from the frame ring (shm_protocol.h): it carries normalized
// evdev input events (EV_KEY / EV_REL / EV_ABS), not video.
//
// Why a batched ring: input arrives in bursts (125 Hz mouse + keys). A
// per-event ring would need a very large slot count. Instead each slot
// holds up to INP_EVENTS_PER_SLOT events and the producer publishes one
// slot per device-read batch, giving low latency and few seqlock writes.
//
// Layout (little-endian), total = INP_HDR_SIZE + n_slots * slot_stride:
//
//   offset 0   InpHeader (64 B region; 56 B used, 8 B reserved)
//      u32 magic, version, n_slots, events_per_slot, alive, pad, pad2, pad3
//      u64 idx     (atomic; slots published so far, release store)
//      u64 events  (total events published)
//      u64 drops   (overrun drops)
//
//   offset 64  InpSlot x n_slots, stride = INP_SLOT_META + events_per_slot*24
//      u32 seq        (atomic; even = stable, odd = being written)
//      u32 dev_id     (device index in CLI order: keyboard first, then
//                      pointer; all events in a slot share one dev_id,
//                      because a slot is one batch from one device read)
//      u64 slot_id    (atomic; the idx value this slot was published with)
//      u64 ts_ns      (CLOCK_MONOTONIC at publish, ns)
//      u64 count      (number of valid events in data[0..count))
//      InpEvent data[events_per_slot]   (24 B each)
//
// InpEvent is byte-identical to kernel `struct input_event` (linux/input.h):
// struct timeval time (tv_sec 8 B, tv_usec 8 B), u16 type, u16 code, s32 value.
//
// Producer protocol (seqlock, multiple producers: one thread per evdev
// device). A producer claims its slot by CAS-advancing idx:
//   claimed = atomic_cas(idx, claimed -> claimed+1)   (acq_rel)
//   s = claimed % n_slots; wait until slot s's seq is even
//   seq = prev+1 (odd)  -> write data[0..count), ts_ns, count, slot_id
//   seq = prev+2 (even) -> events += count
// idx therefore advances BEFORE the slot write; a slot is only reused
// after a full lap, and the seqlock plus the slot_id field let the
// consumer detect in-progress or overwritten slots.
//
// Consumer protocol: keep a local `consumed` counter (next slot to read).
//   idx = hdr.idx (acquire)
//   while consumed < idx:
//     s = consumed % n_slots
//     read seq; if odd, wait and retry
//     read slot_id, ts_ns, count, data[0..count)
//     re-read seq; if changed, retry the slot
//     if slot_id != consumed, the slot was overwritten by a later lap:
//       its data is gone — skip it (consumed += 1) and continue
//     deliver the events, consumed += 1
// A slot with count == 0 is an empty (dropped) slot: skip it.
//
// The 8 x u32 header prefix is the same shape as the frame ring's
// CapHeader, so the Python client reuses its <8I / <3Q unpacking.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace anyplay {

// Distinct from SDAI_MAGIC (frame ring) so a client can tell the two ring
// types apart by reading offset 0.
constexpr std::uint32_t SDAI_INPUT_MAGIC = 0x49505449u;  // "IPTI"
constexpr std::uint32_t SDAI_INPUT_VERSION = 1u;
constexpr std::size_t INP_HDR_SIZE = 64;
constexpr std::size_t INP_SLOT_META = 32;

// One slot carries this many events. 32 covers a full device-read batch at
// 125 Hz mouse plus keyboard bursts.
constexpr std::uint32_t INP_EVENTS_PER_SLOT = 32;

// Byte-identical to kernel struct input_event (linux/input.h).
struct InpEvent {
    std::int64_t tv_sec;
    std::int64_t tv_usec;
    std::uint16_t type;
    std::uint16_t code;
    std::int32_t value;
};
static_assert(sizeof(InpEvent) == 24, "InpEvent must match struct input_event");

// Same 8 x u32 prefix shape as CapHeader, then the same u64 triple.
struct InpHeader {
    std::uint32_t magic;            // offset 0
    std::uint32_t version;          // offset 4
    std::uint32_t n_slots;          // offset 8
    std::uint32_t events_per_slot;  // offset 12
    std::atomic<std::uint32_t> alive;  // offset 16
    std::atomic<std::uint32_t> pad;    // offset 20
    std::atomic<std::uint32_t> pad2;   // offset 24
    std::atomic<std::uint32_t> pad3;   // offset 28
    std::atomic<std::uint64_t> idx;    // offset 32 (slots published)
    std::atomic<std::uint64_t> events; // offset 40 (events published)
    std::atomic<std::uint64_t> drops;  // offset 48 (overrun drops)
};
static_assert(sizeof(InpHeader) == 56, "InpHeader must be 56 bytes");

// Same 32 B metadata shape as CapSlotMeta; the wait_ns field position
// (offset 24) is reused here as the per-slot event count. The pad field
// (offset 4) is reused as the per-slot device id.
struct InpSlotMeta {
    std::atomic<std::uint32_t> seq;      // offset 0
    std::atomic<std::uint32_t> dev_id;   // offset 4 (CLI device order)
    std::atomic<std::uint64_t> slot_id;  // offset 8
    std::atomic<std::uint64_t> ts_ns;    // offset 16 (CLOCK_MONOTONIC)
    std::atomic<std::uint64_t> count;    // offset 24 (valid events in data)
};
static_assert(sizeof(InpSlotMeta) == INP_SLOT_META,
              "InpSlotMeta must be 32 bytes");

inline std::size_t input_slot_stride() {
    return INP_SLOT_META +
           static_cast<std::size_t>(INP_EVENTS_PER_SLOT) * sizeof(InpEvent);
}

}  // namespace anyplay
