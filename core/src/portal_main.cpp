// portal_main.cpp — PipeWire (xdg-desktop-portal) capture daemon.
//
// Port of native/portal_capture.c. Separate binary from anyplay-capture
// (the X11 daemon) so the working X11 regression path is untouched.
//
// Behavior and protocol match the C daemon:
//   CLI:   --slots N --shm PATH --sock PATH [--width W] [--height H] [--test]
//          [--keyboard PATH[,PATH...]] [--pointer PATH[,PATH...]]
//          [--input-shm PATH] [--input-slots N] [--input-only]
//   Env:   PW_FD, PW_NODE_ID (set by tools/portal-pw-fd; required unless
//          --test or --input-only)
//   Stdout: "READY <shm> <sock> [input=<input_shm>]\n"
//   Socket: 'q' -> 'Q' | 's' -> "frames=N fps=X drops=N [in_events=N
//          in_drops=N]\n"
//
// The inherited PW_FD is held open for the whole process lifetime — that is
// the portal session keepalive.
//
// P-core step 2: the evdev input ring (shared EvdevReader/InputRing code
// with the X11 daemon) is available here too, because the game runs under
// Wayland and the agent captures via this daemon.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "control_sock.h"
#include "evdev_reader.h"
#include "frame_ring.h"
#include "input_ring.h"
#include "portal_capture.h"

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

// Detect / reconnect retry budget — matches the C daemon.
constexpr int kMaxRetry = 20;
constexpr unsigned kMaxSlots = 64;

struct Options {
    int pw_fd = -1;
    unsigned node_id = 0;
    unsigned force_w = 0;
    unsigned force_h = 0;
    unsigned slots = 8;
    bool test = false;
    std::string shm;
    std::string sock;
    // Input ring (P-core step 2).
    std::string keyboard = "";
    std::string pointer = "";
    std::string input_shm = "/dev/shm/anyplay_input.bin";
    unsigned input_slots = 32;
    bool input_only = false;
};

void usage() {
    std::fprintf(
        stderr,
        "Usage: anyplay-portal-capture --slots N --shm PATH --sock PATH "
        "[--width W] [--height H] [--test]\n"
        "                           [--keyboard PATH[,PATH...]] "
        "[--pointer PATH[,PATH...]]\n"
        "                           [--input-shm PATH] [--input-slots N] "
        "[--input-only]\n"
        "  Env: PW_FD, PW_NODE_ID (set by tools/portal-pw-fd; required "
        "unless --test or --input-only)\n");
}

bool parse_args(int argc, char **argv, Options &opt) {
    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--width") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.force_w = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--height") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.force_h = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--slots") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.slots = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--shm") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.shm = v;
        } else if (std::strcmp(argv[i], "--sock") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.sock = v;
        } else if (std::strcmp(argv[i], "--test") == 0) {
            opt.test = true;
        } else if (std::strcmp(argv[i], "--keyboard") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.keyboard = v;
        } else if (std::strcmp(argv[i], "--pointer") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.pointer = v;
        } else if (std::strcmp(argv[i], "--input-shm") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.input_shm = v;
        } else if (std::strcmp(argv[i], "--input-slots") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.input_slots = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--input-only") == 0) {
            opt.input_only = true;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            usage();
            std::exit(0);
        } else {
            return false;
        }
    }
    return true;
}

std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        usage();
        return 1;
    }

    const auto pid = static_cast<long>(getpid());
    if (opt.shm.empty())
        opt.shm = "/dev/shm/anyplay_pc_" + std::to_string(pid) + ".bin";
    if (opt.sock.empty())
        opt.sock = "/tmp/anyplay_pc_" + std::to_string(pid) + ".sock";
    if (opt.slots < 1 || opt.slots > kMaxSlots) opt.slots = 8;

    if (!opt.test && !opt.input_only) {
        const char *fd_s = std::getenv("PW_FD");
        const char *nid_s = std::getenv("PW_NODE_ID");
        if (!fd_s || !nid_s) {
            std::fprintf(stderr,
                         "[portal-capture] PW_FD / PW_NODE_ID env not set "
                         "(use --test for a no-fd dry run)\n");
            return 1;
        }
        opt.pw_fd = std::atoi(fd_s);
        opt.node_id = static_cast<unsigned>(std::atoi(nid_s));
    }

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // ------------------------------------------------------------------
    // Input ring + evdev reader (P-core step 2). Set up first so it is
    // alive before READY; the video pipeline may be absent (--input-only).
    // ------------------------------------------------------------------
    const bool have_input = !opt.keyboard.empty() || !opt.pointer.empty();
    std::unique_ptr<anyplay::InputRing> iring;
    anyplay::EvdevReader reader;
    if (have_input) {
        try {
            iring = std::make_unique<anyplay::InputRing>(
                opt.input_shm, opt.input_slots ? opt.input_slots : 32);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
        std::vector<std::string> devs =
            split_csv(opt.keyboard + "," + opt.pointer);
        // Order matters: dev_id == position in this list (CLI order).
        for (const auto &d : devs)
            if (reader.add_device(d) < 0)
                return 1;
        if (reader.n_devices() == 0) {
            std::fprintf(stderr, "no input devices could be opened\n");
            return 1;
        }
        reader.start(*iring);
    }

    if (!opt.input_only) {
        std::fprintf(stderr,
                     "[portal-capture] %s mode; node id %u; %s; %u slots\n",
                     opt.test ? "TEST(videotestsrc)" : "portal",
                     opt.node_id,
                     (opt.force_w > 0 && opt.force_h > 0) ? "forced output"
                                                          : "auto-detect size",
                     opt.slots);
        if (!opt.test)
            std::fprintf(stderr,
                         "[portal-capture] portal fd %d held open "
                         "(session keepalive)\n",
                         opt.pw_fd);

        // ------------------------------------------------------------
        // PipeWire capture source.
        // ------------------------------------------------------------
        anyplay::PortalCapture::Config cfg;
        cfg.pw_fd = opt.pw_fd;
        cfg.node_id = opt.node_id;
        cfg.force_w = opt.force_w;
        cfg.force_h = opt.force_h;
        cfg.test = opt.test;
        anyplay::PortalCapture capture(cfg);

        // ------------------------------------------------------------
        // Detect phase: build, learn the size, retry on transient
        // failure.
        // ------------------------------------------------------------
        std::uint32_t w = 0, h = 0;
        bool size_known = false;
        for (int attempt = 1; attempt <= kMaxRetry; attempt++) {
            if (!g_running.load(std::memory_order_acquire)) break;

            const std::string err = capture.build();
            if (!err.empty()) {
                std::fprintf(stderr,
                             "[portal-capture] build failed (attempt %d): %s\n",
                             attempt, err.c_str());
                capture.stop();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }

            const int r = capture.detect(w, h);
            if (r == 0) {
                size_known = true;
                break;
            }
            capture.stop();
            if (r == -1) {
                std::fprintf(stderr,
                             "[portal-capture] stream ended before first "
                             "frame; exiting\n");
                break;
            }
            // r == -2: no frame yet (node not ready / static window).
            std::fprintf(stderr,
                         "[portal-capture] no frame (attempt %d/%d), "
                         "retrying\n",
                         attempt, kMaxRetry);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        if (!size_known) {
            reader.stop();
            if (!g_running.load(std::memory_order_acquire))
                std::fprintf(stderr,
                             "[portal-capture] quit during detect\n");
            else
                std::fprintf(stderr,
                             "[portal-capture] could not capture a frame; "
                             "giving up\n");
            return 1;
        }

        // ------------------------------------------------------------
        // Shm ring at the detected/forced size.
        // ------------------------------------------------------------
        std::unique_ptr<anyplay::FrameRing> ring;
        try {
            ring = std::make_unique<anyplay::FrameRing>(
                opt.shm, w, h, opt.slots);
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                         "[portal-capture] shm allocation failed: %s\n",
                         e.what());
            capture.stop();
            reader.stop();
            return 1;
        }

        // ------------------------------------------------------------
        // Control socket (ring now exists, so stats are always valid).
        // ------------------------------------------------------------
        auto frame_id = std::make_shared<std::uint64_t>(0);
        auto *ring_ptr = ring.get();
        const auto t_start = std::chrono::steady_clock::now();
        anyplay::ControlSocket sock(
            opt.sock,
            [ring_ptr, &iring, t_start]() {
                const auto up =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start)
                        .count();
                char buf[256];
                std::snprintf(
                    buf, sizeof(buf),
                    "frames=%llu fps=%.1f drops=%llu",
                    static_cast<unsigned long long>(
                        ring_ptr->frames_published()),
                    up > 0.0
                        ? static_cast<double>(ring_ptr->frames_published()) /
                              up
                        : 0.0,
                    static_cast<unsigned long long>(ring_ptr->drops()));
                if (iring)
                    std::snprintf(
                        buf + strlen(buf), sizeof(buf) - strlen(buf),
                        " in_events=%llu in_drops=%llu\n",
                        static_cast<unsigned long long>(
                            iring->events_published()),
                        static_cast<unsigned long long>(iring->drops()));
                else
                    std::strcat(buf, "\n");
                return std::string(buf);
            });

        if (!sock.start()) {
            capture.stop();
            reader.stop();
            return 1;
        }

        std::printf("READY %s %s", opt.shm.c_str(), opt.sock.c_str());
        if (iring)
            std::printf(" input=%s", opt.input_shm.c_str());
        std::printf("\n");
        std::fflush(stdout);

        // ------------------------------------------------------------
        // Stream phase: publish frames; reconnect if the stream ends
        // with 0 frames (the game disconnected, e.g. death/respawn).
        // ------------------------------------------------------------
        for (int s = 0;; s++) {
            const int rc = capture.stream(
                [ring_ptr, frame_id](const void *data, std::size_t stride,
                                     std::uint64_t wait_ns) {
                    ring_ptr->publish_strided(data, stride, (*frame_id)++,
                                              wait_ns);
                },
                // Geometry changed mid-session (resize): the fixed-size
                // ring can't follow, so the frame is dropped. The C
                // daemon just logs and continues (it does not count
                // these), so a no-op is the faithful behaviour.
                [] {},
                [&sock] {
                    return !g_running.load(std::memory_order_relaxed) ||
                           sock.quit_requested();
                });

            if (!g_running.load(std::memory_order_acquire)) break;
            if (rc == 0) break;  // frames flowed then stream ended -> session over

            // rc == 1: 0 frames, stream ended -> try to reconnect.
            if (s >= kMaxRetry) {
                std::fprintf(stderr,
                             "[portal-capture] stream kept ending; giving "
                             "up\n");
                break;
            }
            std::fprintf(stderr,
                         "[portal-capture] stream ended; reconnecting\n");
            capture.stop();
            const std::string err = capture.build();
            if (!err.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
        }

        // ------------------------------------------------------------
        // Cleanup.
        // ------------------------------------------------------------
        g_running.store(false, std::memory_order_relaxed);
        capture.stop();
        ring->mark_dead();
        sock.stop();
    } else {
        // --input-only: no video, no PipeWire; just the input ring and
        // the control socket (headless regression / step-3 testing).
        const auto t_start = std::chrono::steady_clock::now();
        (void)t_start;
        anyplay::ControlSocket sock(
            opt.sock,
            [&iring]() {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "frames=0 fps=0.0 drops=0");
                if (iring)
                    std::snprintf(
                        buf + strlen(buf), sizeof(buf) - strlen(buf),
                        " in_events=%llu in_drops=%llu\n",
                        static_cast<unsigned long long>(
                            iring->events_published()),
                        static_cast<unsigned long long>(iring->drops()));
                else
                    std::strcat(buf, "\n");
                return std::string(buf);
            });
        if (!sock.start()) {
            reader.stop();
            return 1;
        }

        std::printf("READY - %s", opt.sock.c_str());
        if (iring)
            std::printf(" input=%s", opt.input_shm.c_str());
        std::printf("\n");
        std::fflush(stdout);

        while (g_running.load(std::memory_order_acquire) &&
               !sock.quit_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        g_running.store(false, std::memory_order_relaxed);
        sock.stop();
    }

    reader.stop();
    if (iring)
        iring->mark_dead();
    std::fprintf(stderr, "[portal-capture] exited cleanly\n");
    return 0;
}
