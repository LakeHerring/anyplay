// input_ring.cpp — implementation. See input_ring.h and
// anyplay/input_ring_protocol.h for the contract.
//
// Deliberately mirrors frame_ring.cpp: same shm-file lifecycle, same
// one-shot SIGBUS sentinel, same seqlock publish protocol.
//
// The sentinel's real role: the daemon's own mapping of its own file
// is always valid while the file is intact, so a SIGBUS here means
// the file was destroyed under the mapping — in practice, a NEW
// daemon instance reopened the same path with O_TRUNC (stale
// instance from a crashed process, see NativeCapture's stale-path
// cleanup). The stale daemon must die rather than fault in a loop,
// so the handler re-raises with the default disposition. (A signal
// handler may never simply return from a persistent fault: the
// faulting instruction re-executes and faults again.)

#include "input_ring.h"

#include "anyplay/shm_protocol.h"  // now_mono_ns()

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace anyplay {

// ---------------------------------------------------------------------------
// Consumer-presence sentinel (SIGBUS handler, same as the frame ring).
// ---------------------------------------------------------------------------

namespace {

std::atomic<bool> g_consumer_gone{false};

void sigbus_handler(int) {
    g_consumer_gone.store(true, std::memory_order_relaxed);
    // Re-raise with default disposition so the process dies rather than
    // looping forever on the faulting instruction.
    signal(SIGBUS, SIG_DFL);
    raise(SIGBUS);
}

void install_sigbus_handler() {
    struct sigaction sa;
    sa.sa_handler = sigbus_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGBUS, &sa, nullptr);
}

}  // namespace

// ---------------------------------------------------------------------------
// InputRing
// ---------------------------------------------------------------------------

InputRing::InputRing(std::string path, std::uint32_t n_slots)
    : path_(std::move(path)),
      stride_(input_slot_stride()) {
    if (n_slots == 0 || n_slots > 4096) {
        throw std::runtime_error("InputRing: invalid n_slots");
    }

    map_size_ = INP_HDR_SIZE + static_cast<std::size_t>(n_slots) * stride_;

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("InputRing: open(" + path_ +
                                 "): " + std::strerror(errno));
    }
    if (::ftruncate(fd_, static_cast<off_t>(map_size_)) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("InputRing: ftruncate: " +
                                 std::string(std::strerror(errno)));
    }
    mem_ = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                  fd_, 0);
    if (mem_ == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("InputRing: mmap: " +
                                 std::string(std::strerror(errno)));
    }

    std::memset(mem_, 0, map_size_);

    hdr_ = static_cast<InpHeader *>(mem_);
    hdr_->magic = SDAI_INPUT_MAGIC;
    hdr_->version = SDAI_INPUT_VERSION;
    hdr_->n_slots = n_slots;
    hdr_->events_per_slot = INP_EVENTS_PER_SLOT;
    hdr_->alive.store(1, std::memory_order_release);
    hdr_->idx.store(0, std::memory_order_release);

    install_sigbus_handler();
}

InputRing::~InputRing() {
    mark_dead();
    if (mem_ != nullptr && mem_ != MAP_FAILED) {
        ::munmap(mem_, map_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    // Remove the shm file so /dev/shm does not accumulate stale entries.
    if (!path_.empty()) {
        ::unlink(path_.c_str());
    }
}

void InputRing::mark_dead() {
    if (hdr_ != nullptr) {
        hdr_->alive.store(0, std::memory_order_release);
    }
}

void *InputRing::slot_data(std::size_t s) const {
    return static_cast<unsigned char *>(mem_) + INP_HDR_SIZE +
           s * stride_ + INP_SLOT_META;
}

InpSlotMeta *InputRing::slot_meta(std::size_t s) const {
    return reinterpret_cast<InpSlotMeta *>(
        static_cast<unsigned char *>(mem_) + INP_HDR_SIZE + s * stride_);
}

bool InputRing::consumer_present() const {
    // Any read of the header faults if the file was destroyed under
    // the mapping (a new daemon instance took over the path).
    volatile auto *h = reinterpret_cast<const volatile InpHeader *>(hdr_);
    (void)h->magic;
    g_consumer_gone.store(false, std::memory_order_relaxed);
    return !g_consumer_gone.load(std::memory_order_relaxed);
}

void InputRing::publish(std::uint32_t dev_id, const void *events,
                        std::uint32_t count) {
    const auto h = hdr_;

    if (count == 0)
        return;
    if (count > INP_EVENTS_PER_SLOT) {
        // Overrun guard: never publish a partial batch.
        h->drops.fetch_add(count, std::memory_order_relaxed);
        return;
    }
    if (!consumer_present()) {
        h->drops.fetch_add(count, std::memory_order_relaxed);
        return;
    }

    const auto n_slots = h->n_slots;

    // Atomically claim the next slot index. Multiple producer threads
    // (one per evdev device) share this ring, so a read-then-store on
    // idx would let two threads claim the same slot and one publish
    // would silently overwrite the other (lost batch, idx advanced only
    // once). The CAS serializes the claim; the seqlock below still
    // guards the write window against the consumer. idx now advances
    // BEFORE the slot write; consumers handle the odd-seq window by
    // retrying, and a slot is never reused until a full lap.
    std::uint64_t claimed = h->idx.load(std::memory_order_relaxed);
    while (!h->idx.compare_exchange_weak(
               claimed, claimed + 1,
               std::memory_order_acq_rel,
               std::memory_order_relaxed)) {
    }
    const auto s = static_cast<std::size_t>(claimed % n_slots);
    auto *meta = slot_meta(s);

    // The claimed slot may still hold a prior lap's in-progress write
    // (consumer lagged a full lap). Wait for it to settle; the prior
    // writer always finishes its seq pair, so this terminates.
    for (int tries = 0; tries < 1000; tries++) {
        if ((meta->seq.load(std::memory_order_acquire) & 1u) == 0) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (meta->seq.load(std::memory_order_acquire) & 1u) {
        // Pathological: prior writer stuck mid-slot. Wait for the seq to
        // settle, then publish an EMPTY slot so the consumer never sees
        // the prior lap's stale data at this idx, and count the drop.
        while (meta->seq.load(std::memory_order_acquire) & 1u) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        const std::uint32_t p2 = meta->seq.load(std::memory_order_relaxed);
        meta->seq.store(p2 + 1, std::memory_order_release);
        meta->count.store(0, std::memory_order_relaxed);
        meta->seq.store(p2 + 2, std::memory_order_release);
        h->drops.fetch_add(count, std::memory_order_relaxed);
        return;
    }

    const std::uint32_t prev = meta->seq.load(std::memory_order_relaxed);
    meta->seq.store(prev + 1, std::memory_order_release);  // writing

    std::memcpy(slot_data(s), events,
                static_cast<std::size_t>(count) * sizeof(InpEvent));
    meta->dev_id.store(dev_id, std::memory_order_relaxed);
    meta->ts_ns.store(now_mono_ns(), std::memory_order_relaxed);
    meta->count.store(count, std::memory_order_relaxed);
    meta->slot_id.store(claimed, std::memory_order_relaxed);

    meta->seq.store(prev + 2, std::memory_order_release);  // done

    h->events.fetch_add(count, std::memory_order_relaxed);
}

}  // namespace anyplay
