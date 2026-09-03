# core/Makefile — C++ game core (capture daemons first; input/timing/IPC
# to follow as P-core progresses).
#
# Builds:
#   build/anyplay-capture          — X11 daemon (port of native/capture-daemon)
#   build/anyplay-portal-capture   — PipeWire/portal daemon (port of
#                                    native/portal-capture)
#
# Both are drop-in replacements for the C daemons (same CLI, same shm
# protocol, same READY line, same control-socket protocol). Regression
# gates:
#
#   python core/test/regression_daemon.py    (X11; needs a display)
#   python core/test/regression_portal.py    (portal; --test is headless)

CXX      ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++20 -pthread
LDFLAGS  ?= -pthread

GST_CFLAGS := $(shell pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0)
GST_LIBS   := $(shell pkg-config --libs   gstreamer-1.0 gstreamer-app-1.0 glib-2.0)

BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj

# Shared by both daemons.
COMMON_SRCS := src/frame_ring.cpp src/control_sock.cpp \
               src/input_ring.cpp src/evdev_reader.cpp
# X11 daemon (ximagesrc, known size up front).
X11_SRCS    := src/main.cpp src/gst_capture.cpp $(COMMON_SRCS)
# Portal daemon (pipewiresrc, size learned from the first frame).
PORTAL_SRCS := src/portal_main.cpp src/portal_capture.cpp $(COMMON_SRCS)

X11_OBJS    := $(X11_SRCS:%.cpp=$(OBJ_DIR)/%.o)
PORTAL_OBJS := $(PORTAL_SRCS:%.cpp=$(OBJ_DIR)/%.o)

HDRS := src/frame_ring.h src/gst_capture.h src/control_sock.h \
        src/portal_capture.h src/input_ring.h src/evdev_reader.h \
        include/anyplay/shm_protocol.h \
        include/anyplay/input_ring_protocol.h

X11_BIN    := $(BUILD_DIR)/anyplay-capture
PORTAL_BIN := $(BUILD_DIR)/anyplay-portal-capture

.PHONY: all clean test

all: $(X11_BIN) $(PORTAL_BIN)

$(OBJ_DIR)/%.o: %.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(GST_CFLAGS) -Iinclude -Isrc -c $< -o $@

$(X11_BIN): $(X11_OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(GST_LIBS)

$(PORTAL_BIN): $(PORTAL_OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(GST_LIBS)

test: $(X11_BIN) $(PORTAL_BIN)
	python3 test/regression_daemon.py
	python3 test/regression_portal.py
	python3 test/regression_input.py

clean:
	rm -rf $(BUILD_DIR)
