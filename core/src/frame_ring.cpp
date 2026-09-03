// frame_ring.cpp — shm ring implementation (port of native/capture_daemon.c).

#include "frame_ring.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace anyplay {

// ---------------------------------------------------------------------------
// Consumer-presence sentinel (SIGBUS handler, same as the C daemon).
// ---------------------------------------------------------------------------

namespace {

std::atomic<bool> g_consumer_gone{false};

void sigbus_handler(int) {
    g_consumer_gone.store(true, std::memory_order_relaxed);
    // Re-raise with default disposition so the process dies rather than
    // looping forever.
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
// FrameRing
// ---------------------------------------------------------------------------

FrameRing::FrameRing(std::string path, std::uint32_t width,
                     std::uint32_t height, std::uint32_t n_slots)
    : path_(std::move(path)),
      stride_(SLOT_META + static_cast<std::size_t>(width) * height * 3) {
    if (width == 0 || height == 0 || n_slots == 0) {
        throw std::runtime_error("FrameRing: invalid dimensions");
    }

    map_size_ = HDR_SIZE + static_cast<std::size_t>(n_slots) * stride_;

    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd_ < 0) {
        throw std::runtime_error("FrameRing: open(" + path_ +
                                 "): " + std::strerror(errno));
    }
    if (::ftruncate(fd_, static_cast<off_t>(map_size_)) != 0) {
        throw std::runtime_error("FrameRing: ftruncate: " +
                                 std::string(std::strerror(errno)));
    }
    mem_ = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                  fd_, 0);
    if (mem_ == MAP_FAILED) {
        throw std::runtime_error("FrameRing: mmap: " +
                                 std::string(std::strerror(errno)));
    }

    std::memset(mem_, 0, map_size_);

    hdr_ = static_cast<CapHeader *>(mem_);
    hdr_->magic = SDAI_MAGIC;
    hdr_->version = SDAI_VERSION;
    hdr_->width = width;
    hdr_->height = height;
    hdr_->slot_bytes = static_cast<std::uint32_t>(stride_ - SLOT_META);
    hdr_->n_slots = n_slots;
    hdr_->alive.store(1, std::memory_order_release);
    hdr_->idx.store(0, std::memory_order_release);

    install_sigbus_handler();
}

FrameRing::~FrameRing() {
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

void FrameRing::mark_dead() {
    if (hdr_ != nullptr) {
        hdr_->alive.store(0, std::memory_order_release);
    }
}

void *FrameRing::slot_data(std::size_t s) const {
    return static_cast<unsigned char *>(mem_) + HDR_SIZE +
           s * stride_ + SLOT_META;
}

CapSlotMeta *FrameRing::slot_meta(std::size_t s) const {
    return reinterpret_cast<CapSlotMeta *>(static_cast<unsigned char *>(mem_) +
                                           HDR_SIZE + s * stride_);
}

bool FrameRing::consumer_present() const {
    // Any read of the header faults if the consumer unmapped the file.
    volatile auto *h = reinterpret_cast<const volatile CapHeader *>(hdr_);
    (void)h->magic;
    g_consumer_gone.store(false, std::memory_order_relaxed);
    return !g_consumer_gone.load(std::memory_order_relaxed);
}

void FrameRing::publish(const void *data, std::uint64_t frame_id,
                        std::uint64_t wait_ns) {
    // Tightly-packed RGB888: source row stride equals the destination.
    publish_impl(data, static_cast<std::size_t>(hdr_->width) * 3, frame_id,
                 wait_ns);
}

void FrameRing::publish_strided(const void *data, std::size_t src_stride,
                                std::uint64_t frame_id, std::uint64_t wait_ns) {
    publish_impl(data, src_stride, frame_id, wait_ns);
}

void FrameRing::publish_impl(const void *data, std::size_t src_stride,
                             std::uint64_t frame_id, std::uint64_t wait_ns) {
    const auto h = hdr_;

    if (!consumer_present()) {
        h->drops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const auto n_slots = h->n_slots;
    const auto s = static_cast<std::size_t>(frame_id % n_slots);
    auto *sl = slot_meta(s);

    // Wait for any in-progress write to settle.
    for (int tries = 0; tries < 100; tries++) {
        if ((sl->seq.load(std::memory_order_acquire) & 1u) == 0) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (sl->seq.load(std::memory_order_acquire) & 1u) {
        h->drops.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const std::uint32_t prev = sl->seq.load(std::memory_order_relaxed);
    sl->seq.store(prev + 1, std::memory_order_release);  // writing

    // Copy the frame into the slot. If the source rows are padded
    // (PipeWire/GStreamer), pack them down to width*3 per row; otherwise
    // a flat memcpy of the whole frame. Mirrors the C publish_frame.
    const std::size_t dest_row = static_cast<std::size_t>(h->width) * 3;
    const std::size_t frame_bytes = static_cast<std::size_t>(h->width) *
                                    static_cast<std::size_t>(h->height) * 3;
    unsigned char *dst = static_cast<unsigned char *>(slot_data(s));
    if (src_stride == dest_row) {
        std::memcpy(dst, data, frame_bytes);
    } else {
        const unsigned char *src = static_cast<const unsigned char *>(data);
        for (std::uint32_t y = 0; y < h->height; ++y) {
            const std::size_t copy = src_stride < dest_row ? src_stride
                                                           : dest_row;
            std::memcpy(dst + static_cast<std::size_t>(y) * dest_row,
                        src + static_cast<std::size_t>(y) * src_stride, copy);
        }
    }

    sl->frame_id.store(frame_id, std::memory_order_relaxed);
    // Timing stamps in previously-unused metadata bytes (P0 latency work).
    sl->ts_ns.store(now_mono_ns(), std::memory_order_relaxed);
    sl->wait_ns.store(wait_ns, std::memory_order_relaxed);
    sl->seq.store(prev + 2, std::memory_order_release);  // done

    h->frames.fetch_add(1, std::memory_order_relaxed);
    h->idx.store(frame_id + 1, std::memory_order_release);
}

}  // namespace anyplay
