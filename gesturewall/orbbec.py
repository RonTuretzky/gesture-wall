"""Orbbec Gemini 335 frame source via the pyorbbecsdk v2 Python bindings.

One Gemini 335 (serial ``CP0E8530002Y``) replaces the pair of Kinect v2s: it
watches BOTH walls A and B from the room's far corner, so the pipeline is back
to a single shared frame with joint single-camera autocal. This module speaks
the exact same source contract as :class:`gesturewall.kinect.KinectV2Source`:

  * light constructor — no SDK import, no device I/O;
  * ``start()`` — idempotent; opens the device and starts the pipeline;
  * ``read(timeout=None)`` — ``(color, depth_m, intr)`` or ``None`` on stall:
      - ``color``  : ``uint8`` (H, W, 3) **BGR** image,
      - ``depth_m``: ``float32`` (H, W) depth in **metres**, pixel-aligned to
        the color image via the SDK's :class:`AlignFilter`,
      - ``intr``   : the color camera's
        :class:`~gesturewall.geometry.CameraIntrinsics`;
  * ``intrinsics`` property — last-known intrinsics or ``None``;
  * ``close()`` — idempotent, safe pre-start; a later ``read()`` respawns.

Units: the SDK hands back ``uint16`` depth whose ``get_depth_scale()`` maps
RAW -> **millimetres**; we convert to metres (``raw * scale / 1000``) so
everything downstream keeps speaking metres, matching
:mod:`gesturewall.geometry` and the Kinect source.

Color order: the color stream is requested as ``OBFormat.RGB`` and flipped to
BGR with a numpy slice (no cv2 dependency), because the rest of the pipeline
(and OpenCV-based preview/autocal) expects BGR.

Laziness: ``pyorbbecsdk`` is only imported inside :meth:`OrbbecSource.start`.
Importing this module never needs the SDK or hardware — required because
un-sudo'd macOS processes currently fail ``uvc_open`` (-3), and tests stub the
SDK by planting a fake ``sys.modules["pyorbbecsdk"]`` before ``start()`` runs.

Probe CLI (verifies the hardware end-to-end)::

    sudo -E .venv/bin/python -m gesturewall.orbbec --serial CP0E8530002Y
"""

from __future__ import annotations

import sys

import numpy as np

from .geometry import CameraIntrinsics

# Defaults for the Gemini 335 color stream (depth uses the sensor default and
# is aligned to color, so it comes out at the color resolution).
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30

# With ``read(timeout=None)`` we still bound each native wait to this slice so
# a stalled pipeline can never hang the caller inside a single SDK call.
_WAIT_SLICE_MS = 1000

# With ``read(timeout=None)`` this many CONSECUTIVE empty wait slices count as
# end-of-stream and read() returns None — the Kinect source signals a dead
# bridge the same way, and the no-timeout callers (the live server, calibrate)
# rely on None to recover instead of blocking forever. A healthy pipeline
# delivers at 30 fps, so 20 s of nothing means the device is gone.
_MAX_STALL_SLICES = 20


class OrbbecSource:
    """Live Gemini 335 frames: aligned color+depth via pyorbbecsdk v2.

    ``device_index`` selects the camera: a ``str`` is matched against device
    serial numbers (e.g. ``"CP0E8530002Y"``), an ``int`` indexes the SDK's
    enumeration order. Construction has **no side effects**; the SDK is
    imported and the pipeline started lazily in :meth:`start` (called from the
    first :meth:`read`).
    """

    def __init__(self, device_index: int | str = 0,
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._ctx = None            # keep the SDK Context alive with the device
        self._device = None
        self._pipe = None
        self._align = None
        self._intrinsics: CameraIntrinsics | None = None
        self._warned_color_tuning = False

    # ----------------------------------------------------------------- #
    # lifecycle                                                          #
    # ----------------------------------------------------------------- #
    def start(self) -> None:
        """Open the device and start the aligned color+depth pipeline.

        Idempotent. Imports ``pyorbbecsdk`` lazily; ``self._pipe`` is only
        committed once the pipeline actually started, so a failed start can
        simply be retried.
        """
        if self._pipe is not None:
            return
        from pyorbbecsdk import (
            AlignFilter,
            Config,
            Context,
            OBError,
            OBFormat,
            OBFrameAggregateOutputMode,
            OBSensorType,
            OBStreamType,
            Pipeline,
        )

        device = self._open_device(Context, OBError)
        self._tune_color(device)

        pipe = Pipeline(device)
        cfg = Config()
        color_profile = pipe.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_video_stream_profile(
            self._width, self._height, OBFormat.RGB, self._fps)
        depth_profile = pipe.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()
        cfg.enable_stream(color_profile)
        cfg.enable_stream(depth_profile)
        # Only emit framesets that contain BOTH streams; read() still guards
        # against missing components defensively.
        cfg.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

        align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        pipe.start(cfg)

        self._device = device
        self._align = align
        self._pipe = pipe

    def _open_device(self, Context, OBError):
        """Enumerate devices and pick by serial (str) or index (int).

        The serial path reads serials from ENUMERATION data (no device open)
        and only opens the matching device — so a second Orbbec held by
        another process can never abort selecting this one.
        """
        try:
            self._ctx = Context()
            devices = self._ctx.query_devices()
            count = devices.get_count()
        except OBError as exc:
            raise RuntimeError(
                f"Orbbec device enumeration failed: {exc}. On macOS USB/UVC "
                "access needs elevated permissions - try re-running with "
                "'sudo -E'.") from exc
        if count == 0:
            raise RuntimeError(
                "no Orbbec devices found; is the Gemini 335 plugged in?")

        sel = self._device_index
        if isinstance(sel, str):
            serials = [devices.get_device_serial_number_by_index(i)
                       for i in range(count)]
            if sel not in serials:
                raise RuntimeError(
                    f"no Orbbec device with serial {sel!r}; "
                    f"found serial(s): {serials}")
            try:
                return devices.get_device_by_serial_number(sel)
            except OBError as exc:
                raise RuntimeError(
                    f"failed to open Orbbec device {sel!r}: {exc}. On macOS "
                    "USB/UVC access needs elevated permissions - try "
                    "re-running with 'sudo -E' (uvc_open error -3 is the "
                    "permission failure).") from exc

        idx = int(sel)
        if not 0 <= idx < count:
            raise RuntimeError(
                f"Orbbec device index {idx} out of range "
                f"({count} device(s) found)")
        return self._get_device(devices, idx, OBError)

    @staticmethod
    def _get_device(devices, index: int, OBError):
        """Open one enumerated device, translating SDK permission errors.

        On macOS an un-sudo'd process gets ``uvc_open`` error -3 right here
        (opening the device is what needs USB permission), so wrap it with an
        actionable message rather than a bare pybind exception.
        """
        try:
            return devices.get_device_by_index(index)
        except OBError as exc:
            raise RuntimeError(
                f"failed to open Orbbec device #{index}: {exc}. On macOS "
                "USB/UVC access needs elevated permissions - try re-running "
                "with 'sudo -E' (uvc_open error -3 is the permission "
                "failure).") from exc

    def _tune_color(self, device) -> None:
        """Best-effort: freeze auto white-balance and auto exposure.

        Autocal's OFF/ON magenta diffing compares color frames captured
        seconds apart, so a drifting auto-WB/auto-exposure pipeline shows up
        as fake "projector" deltas; a locked color pipeline keeps the diff
        clean. Firmware/SDK combos that reject these properties just get one
        warning and we carry on with auto everything.
        """
        try:
            from pyorbbecsdk import OBPropertyID
            device.set_bool_property(
                OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, False)
            device.set_bool_property(
                OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
        except Exception as exc:  # pragma: no cover - depends on firmware
            if not self._warned_color_tuning:
                self._warned_color_tuning = True
                print(f"orbbec: could not disable auto WB/exposure ({exc}); "
                      "autocal color diffs may be noisier", file=sys.stderr)

    def close(self) -> None:
        """Stop the pipeline and drop all SDK refs (idempotent, pre-start ok).

        After ``close()`` the source is back to its constructed state, so a
        later :meth:`read` transparently restarts the pipeline.
        """
        pipe = self._pipe
        self._pipe = None
        self._align = None
        self._device = None
        self._ctx = None
        self._intrinsics = None
        if pipe is None:
            return
        try:
            pipe.stop()
        except Exception:  # pragma: no cover - best effort
            pass

    # ----------------------------------------------------------------- #
    # frames                                                             #
    # ----------------------------------------------------------------- #
    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        """Color-camera intrinsics from the running pipeline, or ``None``."""
        return self._intrinsics

    def read(self, timeout: float | None = None):
        """Return the next aligned ``(color, depth_m, intr)``, or ``None``.

        Blocks until a frameset with both color and depth arrives. With
        ``timeout`` (seconds), returns ``None`` if none arrives in time —
        mirroring :meth:`KinectV2Source.read`'s None-on-stall semantics. With
        ``timeout=None`` it waits in bounded ``_WAIT_SLICE_MS`` native slices
        and returns ``None`` after ``_MAX_STALL_SLICES`` consecutive empty
        ones — the Kinect source's end-of-stream signal, which the no-timeout
        callers (server, calibrate) rely on to recover from a dead camera.
        """
        import time as _time

        if self._pipe is None:
            self.start()
        assert self._pipe is not None and self._align is not None

        deadline = None if timeout is None else _time.monotonic() + timeout
        stalls = 0
        while True:
            if deadline is None:
                wait_ms = _WAIT_SLICE_MS
            else:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    return None
                wait_ms = max(1, min(_WAIT_SLICE_MS, int(remaining * 1000)))

            frames = self._pipe.wait_for_frames(wait_ms)
            if frames is None:
                stalls += 1
                if deadline is None and stalls >= _MAX_STALL_SLICES:
                    return None  # ~20 s of silence: treat as end-of-stream
                continue  # bounded wait: deadline check at loop top decides
            stalls = 0
            aligned = self._align.process(frames)
            if aligned is None:
                continue
            frameset = aligned.as_frame_set()
            color = frameset.get_color_frame()
            depth = frameset.get_depth_frame()
            if color is None or depth is None:
                continue  # partial set despite FULL_FRAME_REQUIRE; keep going
            try:
                return self._decode(color, depth)
            except ValueError:
                # Transiently malformed frame (buffer size != WxH) — the
                # SDK's own aligned example skips these; so do we.
                continue

    def _decode(self, color, depth):
        """SDK frames -> (BGR uint8, metres float32, CameraIntrinsics).

        ``get_data()`` returns a numpy uint8 array in this wheel (raw bytes in
        tests/stubs); ``np.frombuffer`` accepts both via the buffer protocol
        with zero copies — the only copies are the deliberate BGR flip and the
        float32 conversion.
        """
        cw, ch = int(color.get_width()), int(color.get_height())
        rgb = np.frombuffer(
            color.get_data(), dtype=np.uint8).reshape(ch, cw, 3)
        bgr = rgb[:, :, ::-1].copy()  # RGB -> BGR without needing cv2

        dw, dh = int(depth.get_width()), int(depth.get_height())
        raw = np.frombuffer(
            depth.get_data(), dtype=np.uint16).reshape(dh, dw)
        # get_depth_scale() maps RAW -> MILLIMETRES; /1000 lands in metres.
        depth_m = raw.astype(np.float32) * (
            float(depth.get_depth_scale()) / 1000.0)

        if self._intrinsics is None:
            rgb_i = self._pipe.get_camera_param().rgb_intrinsic
            # Width/height are taken from the actual aligned frames (depth is
            # aligned to color, so both share the color geometry).
            self._intrinsics = CameraIntrinsics(
                fx=float(rgb_i.fx), fy=float(rgb_i.fy),
                cx=float(rgb_i.cx), cy=float(rgb_i.cy),
                width=cw, height=ch)
        return bgr, depth_m, self._intrinsics


# --------------------------------------------------------------------------- #
# probe CLI: sudo -E python -m gesturewall.orbbec --serial CP0E8530002Y        #
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    """Read a few frames off the real camera and print vital signs."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Probe an Orbbec Gemini 335: read frames, print "
                    "resolution, fps, center depth, and intrinsics.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--serial", help="match device by serial number")
    group.add_argument("--index", type=int, default=0,
                       help="device enumeration index (default 0)")
    parser.add_argument("--frames", type=int, default=30,
                        help="number of frames to read (default 30)")
    args = parser.parse_args(argv)

    device: int | str = args.serial if args.serial is not None else args.index
    src = OrbbecSource(device_index=device)
    try:
        got = 0
        t_first = t_last = None
        depth_m = intr = None
        for i in range(args.frames):
            item = src.read(timeout=5.0)
            if item is None:
                print(f"timed out waiting for frame {i}", file=sys.stderr)
                break
            color, depth_m, intr = item
            t_last = time.monotonic()
            if t_first is None:
                t_first = t_last
                print(f"color {color.shape[1]}x{color.shape[0]} BGR, "
                      f"depth {depth_m.shape[1]}x{depth_m.shape[0]} m")
            got += 1

        if got == 0:
            print("no frames received", file=sys.stderr)
            return 1
        if got > 1 and t_last > t_first:
            print(f"{got} frames, ~{(got - 1) / (t_last - t_first):.1f} fps")
        h, w = depth_m.shape
        patch = depth_m[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
        valid = patch[patch > 0]
        center = float(np.median(valid)) if valid.size else float("nan")
        print(f"center median depth: {center:.3f} m")
        print(f"intrinsics: fx={intr.fx:.2f} fy={intr.fy:.2f} "
              f"cx={intr.cx:.2f} cy={intr.cy:.2f} "
              f"{intr.width}x{intr.height}")
        return 0
    finally:
        src.close()


if __name__ == "__main__":  # pragma: no cover - hardware probe
    raise SystemExit(_main())
