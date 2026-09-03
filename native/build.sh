#!/usr/bin/env bash
# Build the native capture daemons + portal handshake tool.
set -uo pipefail
cd "$(dirname "$0")"

rc=0

gcc -O2 -Wall -Wextra -o capture-daemon capture_daemon.c \
    $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0) \
    $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0) \
    -lpthread \
    && echo "built $(pwd)/capture-daemon" || rc=1

# portal-pw-fd (D-Bus/portal handshake; lives in ../tools)
TOOLS_DIR="$(cd .. && pwd)/tools"
if [ -f "$TOOLS_DIR/portal_pw_fd.c" ]; then
    gcc -O2 -Wall -Wextra -o "$TOOLS_DIR/portal-pw-fd" "$TOOLS_DIR/portal_pw_fd.c" \
        $(pkg-config --cflags --libs gio-2.0) \
        && echo "built $TOOLS_DIR/portal-pw-fd" || rc=1
fi

# portal-capture (in-process GStreamer DmaBuf->RGB -> shm ring)
# Uses pipewiresrc via gst_parse_launch, so it links GStreamer, not the
# pipewire C API directly.
gcc -O2 -Wall -Wextra -o portal-capture portal_capture.c \
    $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0) \
    $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0) \
    -lpthread \
    && echo "built $(pwd)/portal-capture" || rc=1

# capture-check (TEST 2-4 diagnostic: in-process GStreamer, testsrc + portal)
gcc -O2 -Wall -Wextra -pthread -o capture-check capture_check.c \
    $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 glib-2.0) \
    $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 glib-2.0) \
    && echo "built $(pwd)/capture-check" || rc=1

exit $rc
