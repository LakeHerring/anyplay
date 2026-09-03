"""Capture X11 screen video to MP4 with an in-process GStreamer pipeline.

Pipeline:  ximagesrc ! videoconvert ! x264enc ! mp4mux ! filesink

Same output contract as :class:`VideoCapturer` (h264/yuv420p in an mp4
container) and the same public interface (``start_time``, ``start()``,
``stop()``, ``running()``), so ``capture.py`` can swap backends.

Notes:
* The X11 source is ``ximagesrc`` (gstreamer1.0-plugins-good); the
  ffmpeg-style ``x11grab`` element was removed from GStreamer in 1.26.
  Regions are captured natively via ``startx/starty/endx/endy``
  (inclusive end coordinates; ``videoscale`` normalizes the result to the
  requested even dimensions).
* ``use-damage=false`` makes the source poll at the negotiated framerate
  (the caps filter below), giving a steady 60 FPS master even when the
  screen is idle; damage-driven capture would emit frames only on change
  and never complete preroll on a still screen.
* ``x264enc`` rate-controls by target bitrate, not CRF. The ``crf`` argument
  is mapped to an approximate bitrate (0.1 bit/pixel at crf 18, halving per
  6 CRF steps); pass ``bitrate_kbps`` to override explicitly.
"""

import threading
import time

import gi
from gi.repository import GLib
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def crf_to_bitrate_kbps(crf, width, height, fps):
    """Approximate x264 CRF quality as a target bitrate in kbit/s."""
    bpp = 0.10 * (2.0 ** (-(int(crf) - 18) / 6.0))
    return max(500, int(width * height * fps * bpp // 1000))


class GstVideoCapturer:
    def __init__(self, out_path, display=":0.0", region="", fps=60, crf=18,
                 preset="veryfast", duration=0.0, bitrate_kbps=0):
        Gst.init(None)
        self.out_path = str(out_path)
        self.display = display
        self.region = region
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.duration = duration
        self.bitrate_kbps = bitrate_kbps
        self.start_time = None
        self.pipeline = None
        self._loop = None
        self._running = False

    @staticmethod
    def _screen_size(display):
        """Root window size (floored to even) of the display, or a fallback."""
        try:
            from Xlib.display import Display
            disp = Display(display)
            _x, _y, w, h = disp.screen().root.get_geometry()
            disp.close()
        except Exception:
            w, h = 1920, 1080
        return w & ~1, h & ~1

    def _pipeline_desc(self):
        if self.region:
            x, y, w, h = (int(v) for v in self.region.split(","))
            # endx/endy are INCLUSIVE; videoscale then normalizes the
            # off-by-one (and any odd dimensions) to the requested size.
            src = (f"ximagesrc display-name={self.display} "
                   f"startx={x} starty={y} endx={x + w - 1} endy={y + h - 1} "
                   f"do-timestamp=true use-damage=false")
            width, height = w & ~1, h & ~1
        else:
            src = (f"ximagesrc display-name={self.display} "
                   f"do-timestamp=true use-damage=false")
            width, height = self._screen_size(self.display)
        if self.bitrate_kbps:
            bitrate = self.bitrate_kbps
        else:
            bitrate = crf_to_bitrate_kbps(self.crf, width, height, self.fps)
        return (
            f"{src} ! video/x-raw,framerate={self.fps}/1 ! videorate ! "
            f"videoscale ! videoconvert ! "
            f"video/x-raw,width={width},height={height},format=I420 ! "
            f"x264enc speed-preset={self.preset} tune=zerolatency "
            f"bitrate={bitrate} key-int-max={self.fps} ! "
            f"video/x-h264,stream-format=avc,profile=baseline ! "
            f"mp4mux name=mux ! filesink location={self.out_path}"
        )

    def start(self):
        if self.pipeline is not None:
            raise RuntimeError("GstVideoCapturer already started")
        self.start_time = time.monotonic()
        # A GLib main loop (like gst-launch runs) is needed for state-change
        # completion and bus message delivery.
        if self._loop is None:
            self._loop = GLib.MainLoop()
            threading.Thread(target=self._loop.run, daemon=True).start()
        self.pipeline = Gst.parse_launch(self._pipeline_desc())
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer set_state(PLAYING) failed")
        # Block until the state change settles, then fail loudly on errors
        # (bad display, bad region, ...).
        res, state, _ts = self.pipeline.get_state(Gst.SECOND)
        msg = self.pipeline.get_bus().timed_pop_filtered(
            0, Gst.MessageType.ERROR)
        if msg is not None:
            err, _ = msg.parse_error()
            raise RuntimeError(f"GStreamer pipeline failed: {err}")
        if res == Gst.StateChangeReturn.FAILURE or \
                state not in (Gst.State.PLAYING, Gst.State.PAUSED):
            raise RuntimeError(
                f"GStreamer pipeline did not reach PLAYING "
                f"(res={res}, state={state})")
        self._running = True
        if self.duration > 0:
            threading.Thread(target=self._eos_at_duration, daemon=True).start()

    def _eos_at_duration(self):
        time.sleep(self.duration)
        if self.pipeline is not None:
            self.pipeline.send_event(Gst.Event.new_eos())

    def stop(self):
        """Stop the pipeline, finalize the mp4, and return the duration (s).

        mp4mux only writes the moov atom when it sees EOS, so send EOS and
        wait for the muxer to drain before going to NULL.
        """
        if self.pipeline is not None:
            self.pipeline.send_event(Gst.Event.new_eos())
            eos = self.pipeline.get_bus().timed_pop_filtered(
                5 * Gst.SECOND, Gst.MessageType.EOS)
            self.pipeline.set_state(Gst.State.NULL)
            if eos is None:
                print("warning: pipeline EOS timed out; mp4 may be truncated")
            self.pipeline = None
        self._running = False
        return time.monotonic() - self.start_time

    def running(self):
        return self._running
