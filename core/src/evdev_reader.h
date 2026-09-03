// evdev_reader.h — read normalized evdev input events from /dev/input
// nodes and publish them to an InputRing.
//
// One thread per device. Each thread does a blocking read() of kernel
// input_event records, filters to EV_KEY / EV_REL / EV_ABS (the types the
// agent cares about), and publishes each device-read batch as one ring
// slot. EV_SYN and other types are dropped (they carry no action state).

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "input_ring.h"

namespace anyplay {

class EvdevReader {
  public:
    EvdevReader() = default;
    ~EvdevReader();

    EvdevReader(const EvdevReader &) = delete;
    EvdevReader &operator=(const EvdevReader &) = delete;

    // Open a /dev/input node read-only. Returns the assigned dev_id
    // (>= 0) on success, or -1 on failure (error logged to stderr).
    // dev_id is the index the device occupies in the daemon CLI, so the
    // Python client can map ring events back to a path.
    int add_device(const std::string &path);

    // Start one reader thread per opened device, publishing into ring.
    void start(InputRing &ring);

    // Signal all reader threads to stop and join them.
    void stop();

    std::size_t n_devices() const { return devices_.size(); }
    const std::string &device_path(std::size_t i) const {
        return devices_[i].path;
    }
    const std::string &device_name(std::size_t i) const {
        return devices_[i].name;
    }

  private:
    struct Device {
        int fd = -1;
        std::string path;
        std::string name;
    };

    std::vector<Device> devices_;
    InputRing *ring_ = nullptr;
    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
    // One wakeup eventfd PER reader thread. A single shared eventfd is
    // unsafe: stop() writes it once, and the first thread to drain the
    // counter leaves the others' level-triggered poll() re-evaluation
    // seeing cnt==0, so they sleep forever (wakeup steal). Per-thread
    // fds make each wakeup exclusive to its owner.
    std::vector<int> wakeups_;

    void loop(std::size_t dev_id);
};

}  // namespace anyplay
