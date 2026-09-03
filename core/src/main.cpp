// main.cpp — C++ capture daemon (port of native/capture_daemon.c).
//
// Behavior and protocol are byte-identical to the C daemon:
//   CLI:  --region X,Y,W,H [--width W] [--height H] [--fps N]
//          [--slots N] [--display D] [--shm PATH] [--sock PATH]
//   Stdout: "READY <shm> <sock> [input=<input_shm>]\n"
//   Socket: 'q' -> 'Q' | 's' -> "frames=N fps=X drops=N [in_events=N
//          in_drops=N]\n"
//
// Differences (intentional, additive):
//   * C++ RAII classes (FrameRing, GstCapture, ControlSocket)
//   * per-frame timing stamps in previously-unused slot metadata bytes
//     (ts_ns, wait_ns — see anyplay/shm_protocol.h)
//   * optional evdev input ring (P-core step 2): --keyboard / --pointer
//     open /dev/input nodes and publish normalized events to a second shm
//     ring; --input-only runs the input pipeline with no video (no X
//     needed), which is what the regression tests use.

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

#include "control_sock.h"
#include "evdev_reader.h"
#include "frame_ring.h"
#include "gst_capture.h"
#include "input_ring.h"

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int) { g_running.store(false, std::memory_order_relaxed); }

struct Options {
    std::string region = "0,0,640,480";
    unsigned width = 0;
    unsigned height = 0;
    unsigned fps = 60;
    unsigned slots = 8;
    std::string display = ":0.0";
    std::string shm = "/dev/shm/anyplay_capture.bin";
    std::string sock = "/tmp/anyplay_capture.sock";
    // Input ring (P-core step 2).
    std::string keyboard = "";  // comma-separated /dev/input paths
    std::string pointer = "";   // comma-separated /dev/input paths
    std::string input_shm = "/dev/shm/anyplay_input.bin";
    unsigned input_slots = 32;
    bool input_only = false;    // no video pipeline at all
};

void usage() {
    std::fprintf(
        stderr,
        "Usage: anyplay-capture --region X,Y,W,H [--width W] [--height H]\n"
        "                    [--fps N] [--slots N] [--display D]\n"
        "                    [--shm PATH] [--sock PATH]\n"
        "                    [--keyboard PATH[,PATH...]]\n"
        "                    [--pointer PATH[,PATH...]]\n"
        "                    [--input-shm PATH] [--input-slots N]\n"
        "                    [--input-only]\n");
}

bool parse_args(int argc, char **argv, Options &opt) {
    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (std::strcmp(argv[i], "--region") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.region = v;
        } else if (std::strcmp(argv[i], "--width") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.width = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--height") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.height = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--fps") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.fps = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--slots") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.slots = std::strtoul(v, nullptr, 10);
        } else if (std::strcmp(argv[i], "--display") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.display = v;
        } else if (std::strcmp(argv[i], "--shm") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.shm = v;
        } else if (std::strcmp(argv[i], "--sock") == 0) {
            const char *v = next();
            if (!v) return false;
            opt.sock = v;
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

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // ------------------------------------------------------------------
    // Input ring + evdev reader (P-core step 2). Set up first so it is
    // alive before READY; the frame ring may be absent (--input-only).
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

    // ------------------------------------------------------------------
    // Video pipeline (skipped in --input-only mode: no X required).
    // ------------------------------------------------------------------
    std::unique_ptr<anyplay::FrameRing> ring;
    auto frame_id = std::make_shared<std::uint64_t>(0);
    std::unique_ptr<anyplay::GstCapture> capture;
    if (!opt.input_only) {
        try {
            ring = std::make_unique<anyplay::FrameRing>(
                opt.shm, opt.width ? opt.width : 640,
                opt.height ? opt.height : 480, opt.slots);
        } catch (const std::exception &e) {
            std::fprintf(stderr, "%s\n", e.what());
            reader.stop();
            return 1;
        }

        anyplay::GstCapture::Config cfg;
        cfg.region = opt.region;
        cfg.display = opt.display;
        cfg.fps = opt.fps;
        cfg.scale_w = opt.width;
        cfg.scale_h = opt.height;

        capture = std::make_unique<anyplay::GstCapture>(
            cfg,
            [&ring, frame_id](const void *data, std::uint64_t wait_ns) {
                // The ring header carries the actual width/height; the
                // publisher copies exactly slot_bytes, matching the C daemon.
                ring->publish(data, (*frame_id)++, wait_ns);
            });

        std::string err = capture->start();
        if (!err.empty()) {
            std::fprintf(stderr, "%s\n", err.c_str());
            reader.stop();
            return 1;
        }
    }

    // ------------------------------------------------------------------
    // Control socket.
    // ------------------------------------------------------------------
    const auto t_start = std::chrono::steady_clock::now();
    anyplay::ControlSocket sock(
        opt.sock,
        [&ring, &iring, t_start]() {
            char buf[256];
            if (ring) {
                const auto up = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t_start)
                                    .count();
                std::snprintf(
                    buf, sizeof(buf),
                    "frames=%llu fps=%.1f drops=%llu",
                    static_cast<unsigned long long>(ring->frames_published()),
                    up > 0.0
                        ? static_cast<double>(ring->frames_published()) / up
                        : 0.0,
                    static_cast<unsigned long long>(ring->drops()));
            } else {
                std::snprintf(buf, sizeof(buf), "frames=0 fps=0.0 drops=0");
            }
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

    std::printf("READY %s %s", opt.shm.c_str(), opt.sock.c_str());
    if (iring)
        std::printf(" input=%s", opt.input_shm.c_str());
    std::printf("\n");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Main loop.
    // ------------------------------------------------------------------
    while (g_running.load(std::memory_order_acquire) &&
           !sock.quit_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::fprintf(stderr, "stopping capture daemon\n");
    if (capture)
        capture->stop();
    reader.stop();
    sock.stop();
    if (iring)
        iring->mark_dead();

    return 0;
}
