// control_sock.h — AF_UNIX control socket (port of the C daemon's
// accept loop). Protocol: one byte per command.
//   'q'  quit cleanly  -> replies 'Q'
//   's'  stats         -> replies "frames=N fps=X drops=N\n"

#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace anyplay {

class ControlSocket {
  public:
    // Stats provider called on 's'.
    using StatsFn = std::function<std::string()>;

    ControlSocket(std::string path, StatsFn stats);
    ~ControlSocket();

    ControlSocket(const ControlSocket &) = delete;
    ControlSocket &operator=(const ControlSocket &) = delete;

    // Binds, listens, and starts the accept thread.
    // Returns false on bind/listen failure (error on stderr).
    bool start();

    // Stops the accept loop and joins the thread.
    void stop();

    bool quit_requested() const {
        return quit_.load(std::memory_order_acquire);
    }

  private:
    void run();

    std::string path_;
    StatsFn stats_;
    int fd_ = -1;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> quit_{false};
};

}  // namespace anyplay
