// evdev_reader.cpp — see evdev_reader.h for the contract.
//
// Reads raw kernel input_event records (no libevdev dependency) and
// republishes them in the ring's InpEvent layout (byte-identical), so
// the Python client sees the same tuples py-evdev would have produced.

#include "evdev_reader.h"

#include <linux/input.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace anyplay {

namespace {

// Same watched set as Python's InputRecorder (input_recorder.py):
// EV_KEY / EV_ABS / EV_REL. EV_SYN and the rest carry no action state.
constexpr unsigned kWatchedTypes[] = {EV_KEY, EV_ABS, EV_REL};

bool watched(unsigned type) {
    for (unsigned t : kWatchedTypes)
        if (type == t) return true;
    return false;
}

}  // namespace

EvdevReader::~EvdevReader() { stop(); }

int EvdevReader::add_device(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        std::fprintf(stderr, "[evdev] open(%s) failed: %s\n",
                     path.c_str(), std::strerror(errno));
        return -1;
    }
    // Switch to blocking once opened (O_NONBLOCK keeps open() from
    // blocking on devices that wait for a handshake, e.g. some
    // Bluetooth nodes).
    if (::fcntl(fd, F_SETFL, 0) != 0) {
        std::fprintf(stderr, "[evdev] fcntl(%s) failed: %s\n", path.c_str(),
                     std::strerror(errno));
        ::close(fd);
        return -1;
    }

    Device d;
    d.fd = fd;
    d.path = path;
    char name[128] = {0};
    if (ioctl(fd, EVIOCGNAME(sizeof(name)), name) == 0)
        d.name = name;
    else
        d.name = path;

    const int id = static_cast<int>(devices_.size());
    devices_.push_back(std::move(d));
    std::fprintf(stderr, "[evdev] dev %d: %s (%s)\n", id,
                 devices_[id].path.c_str(), devices_[id].name.c_str());
    return id;
}

void EvdevReader::start(InputRing &ring) {
    if (running_ || devices_.empty())
        return;
    ring_ = &ring;
    // One wakeup fd per reader thread (exclusive — see header comment).
    wakeups_.resize(devices_.size(), -1);
    for (std::size_t i = 0; i < devices_.size(); i++) {
        wakeups_[i] = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wakeups_[i] < 0) {
            ring_ = nullptr;
            wakeups_.clear();
            return;
        }
    }
    running_ = true;
    threads_.reserve(devices_.size());
    for (std::size_t i = 0; i < devices_.size(); i++)
        threads_.emplace_back([this, i] { loop(i); });
}

void EvdevReader::stop() {
    if (!running_)
        return;
    running_ = false;
    // Wake every reader thread out of poll(). (Closing the evdev fds here
    // would NOT unblock a concurrent blocking read(): the file description
    // stays open for the duration of the syscall, so join() would hang.)
    for (int wf : wakeups_) {
        if (wf >= 0) {
            std::uint64_t one = 1;
            ssize_t r = ::write(wf, &one, sizeof one);
            (void)r;
        }
    }
    for (auto &t : threads_)
        if (t.joinable())
            t.join();
    threads_.clear();
    // Threads are gone: only now is it safe to close the fds.
    for (auto &d : devices_) {
        if (d.fd >= 0) {
            ::close(d.fd);
            d.fd = -1;
        }
    }
    for (int wf : wakeups_) {
        if (wf >= 0)
            ::close(wf);
    }
    wakeups_.clear();
    ring_ = nullptr;
}

void EvdevReader::loop(std::size_t dev_id) {
    Device &d = devices_[dev_id];
    if (d.fd < 0 || ring_ == nullptr)
        return;

    // One batch buffer: at most INP_EVENTS_PER_SLOT kernel events.
    std::vector<InpEvent> raw(INP_EVENTS_PER_SLOT);
    std::vector<InpEvent> out;
    out.reserve(INP_EVENTS_PER_SLOT);

    const int evfd = wakeups_[dev_id];  // this thread's exclusive wakeup fd
    while (running_) {
        // poll() on the device plus this thread's private wakeup fd.
        // Bounded-wakeup design: stop() writes the eventfd, which is the
        // only way out of poll() other than real input.
        struct pollfd pfds[2] = {
            {d.fd, POLLIN, 0},
            {evfd, POLLIN, 0},
        };
        const int pr = ::poll(pfds, 2, -1);
        if (pr < 0) {
            if (errno == EINTR)
                continue;
            // poll() failed on a still-open fd: ENOMEM or similar. Give up
            // on this device rather than spin.
            std::fprintf(stderr, "[evdev] dev %zu poll failed: %s\n", dev_id,
                         std::strerror(errno));
            break;
        }
        if (pfds[1].revents & POLLIN) {
            // Wakeup from stop(): drain the counter, re-check running_.
            std::uint64_t v = 0;
            while (::read(evfd, &v, sizeof v) > 0)
                ;
            continue;
        }
        if ((pfds[0].revents & POLLIN) == 0)
            continue;

        out.clear();
        const ssize_t n =
            ::read(d.fd, raw.data(), raw.size() * sizeof(InpEvent));
        if (n < 0) {
            if (errno == EINTR)
                continue;
            if (errno == EINVAL || errno == EBADF)
                break;  // fd closed by stop()
            // Transient error (e.g. ENODEV); give up on this device.
            std::fprintf(stderr, "[evdev] dev %zu read failed: %s\n", dev_id,
                         std::strerror(errno));
            break;
        }
        if (n == 0)
            break;  // device node removed

        const std::size_t count = static_cast<std::size_t>(n) /
                                  sizeof(struct input_event);
        const auto *evs = reinterpret_cast<const struct input_event *>(raw.data());
        for (std::size_t i = 0; i < count; i++) {
            if (watched(evs[i].type)) {
                InpEvent e;
                e.tv_sec = static_cast<std::int64_t>(evs[i].time.tv_sec);
                e.tv_usec = static_cast<std::int64_t>(evs[i].time.tv_usec);
                e.type = evs[i].type;
                e.code = evs[i].code;
                e.value = evs[i].value;
                out.push_back(e);
            }
        }
        if (!out.empty())
            ring_->publish(static_cast<std::uint32_t>(dev_id), out.data(),
                           static_cast<std::uint32_t>(out.size()));
    }
}

}  // namespace anyplay
