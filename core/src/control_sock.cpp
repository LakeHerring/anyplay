// control_sock.cpp — AF_UNIX control socket (port of the C daemon).

#include "control_sock.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace anyplay {

ControlSocket::ControlSocket(std::string path, StatsFn stats)
    : path_(std::move(path)), stats_(std::move(stats)) {}

ControlSocket::~ControlSocket() { stop(); }

bool ControlSocket::start() {
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));

    fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) {
        std::fprintf(stderr, "socket: %s\n", std::strerror(errno));
        return false;
    }

    // Let accept() return periodically so stop() can join the thread.
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 200 * 1000;
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    addr.sun_family = AF_UNIX;
    if (path_.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "socket path too long: %s\n", path_.c_str());
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);

    ::unlink(path_.c_str());
    if (::bind(fd_, reinterpret_cast<struct sockaddr *>(&addr),
               sizeof(addr)) != 0) {
        std::fprintf(stderr, "bind(%s): %s\n", path_.c_str(),
                     std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    if (::listen(fd_, 2) != 0) {
        std::fprintf(stderr, "listen: %s\n", std::strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] { this->run(); });
    return true;
}

void ControlSocket::stop() {
    if (!running_.exchange(false)) {
        // Never started; nothing to do.
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    ::unlink(path_.c_str());
}

void ControlSocket::run() {
    while (running_.load(std::memory_order_acquire)) {
        int cfd = ::accept(fd_, nullptr, nullptr);
        if (cfd < 0) continue;  // timeout or EINTR — recheck running_

        char cmd = 0;
        if (::recv(cfd, &cmd, 1, MSG_DONTWAIT) != 1) {
            ::close(cfd);
            continue;
        }

        if (cmd == 'q') {
            ::send(cfd, "Q", 1, 0);
            ::close(cfd);
            quit_.store(true, std::memory_order_release);
            running_.store(false, std::memory_order_release);
            return;
        }

        if (cmd == 's') {
            std::string out = stats_ ? stats_() : "";
            ::send(cfd, out.data(), out.size(), 0);
            ::close(cfd);
        } else {
            ::close(cfd);
        }
    }
}

}  // namespace anyplay
