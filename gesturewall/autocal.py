"""Automatic projector-based calibration for the depth gesture wall.

The idea: the projectors *are* a controllable light source on the exact
surfaces we need to calibrate. ``web/autocal.html`` (one window per wall)
polls this module's tiny HTTP server and displays a single bright magenta
disc at a known wall coordinate ``(u, v)``. For each marker we capture
frames from every Kinect with the marker OFF and then ON; the difference
image localizes the disc in each camera, the aligned depth gives its 3D
position, and:

  * per wall, the labeled ``(u, v, point3)`` samples fit the wall plane
    (:func:`gesturewall.geometry.fit_wall_plane`) — in the reference
    camera's frame, which *defines* the room frame;
  * markers seen by BOTH cameras give correspondences that solve the
    second camera's CAMERA->ROOM extrinsic
    (:func:`gesturewall.geometry.rigid_transform_from_points`, Kabsch);
  * the second camera's samples, transformed into the room frame, are
    pooled into the plane fits so obliquely-seen walls still calibrate.

No clicking, no pointing: the operator just puts the two autocal pages on
the right projectors and stays out of view for ~90 seconds.

Usage:
    .venv/bin/python -m gesturewall.autocal --config room.json [--port 8801]
    # open http://localhost:8801/autocal.html?wall=A  (projector on wall A)
    #      http://localhost:8801/autocal.html?wall=B  (projector on wall B)
    # then: curl -X POST http://localhost:8801/calib/start
    # progress: curl http://localhost:8801/calib/status
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .calibrate import (load_config_dict, merge_camera_pose, merge_wall_plane,
                        save_config_dict)
from .geometry import (CameraIntrinsics, Extrinsic, fit_wall_plane,
                       rigid_transform_from_points, sample_depth)
from .room import RoomConfig

# 3x3 grid of marker positions per wall, inset from the edges so the disc
# stays fully on screen (and away from projector edge blending).
MARKER_GRID = [(u, v) for v in (0.15, 0.5, 0.85) for u in (0.12, 0.5, 0.88)]

# Detection thresholds (tuned for a 512x424 registered color image).
MIN_PEAK = 18.0        # min magenta-score delta at the blob peak, else "not seen"
MIN_AREA_PX = 20       # min blob area in pixels
MAX_AREA_PX = 90000    # ~40% of frame: reject only a whole-scene flash, not a
                       # legitimately large disc seen close / very obliquely
# A projected MAGENTA disc raises RED and BLUE by roughly equal amounts; a
# person (skin/clothing) raises red (and green) but little blue. Requiring the
# blue rise to be a real fraction of the red rise is what rejects people —
# robustly, and without an area/shape gate that also kills big oblique discs.
MAGENTA_BLUE_RATIO = 0.5
DEPTH_WINDOW = 9       # sample_depth window around the blob centroid

# Sanity gates before we write anything into the config.
WIDTH_RANGE = (1.0, 4.5)     # metres
HEIGHT_RANGE = (0.7, 3.5)    # metres
ANGLE_RANGE = (55.0, 125.0)  # degrees between the two wall planes
MAX_RESIDUAL = 0.12          # metres, mean cam1 registration error


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable without hardware)                                #
# --------------------------------------------------------------------------- #
def detect_marker(off_bgr, on_bgr):
    """Find the projected magenta disc as the centroid of the OFF->ON change.

    Works on the *difference* so static scene content (windows, projected
    grids, furniture) cancels out. The score weights the magenta channels
    (R + B) and subtracts green, so broad-spectrum changes (a person moving)
    score much lower than the disc. Returns ``(px, py, peak)`` or ``None``.
    """
    import cv2
    import numpy as np

    off = off_bgr.astype(np.int16)
    on = on_bgr.astype(np.int16)
    d_b = np.clip(on[:, :, 0] - off[:, :, 0], 0, 255)
    d_g = np.clip(on[:, :, 1] - off[:, :, 1], 0, 255)
    d_r = np.clip(on[:, :, 2] - off[:, :, 2], 0, 255)
    score = np.clip(d_r.astype(np.float32) + d_b.astype(np.float32)
                    - d_g.astype(np.float32), 0, None)
    score = cv2.GaussianBlur(score, (9, 9), 0)

    peak = float(score.max())
    if peak < MIN_PEAK:
        return None
    py_peak, px_peak = np.unravel_index(int(score.argmax()), score.shape)
    mask = (score > 0.5 * peak).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # The disc is the blob CONTAINING the brightest point — NOT the largest by
    # area. A big oblique disc blooms and a dim reflection can out-area it, but
    # the score peak always sits on the real, directly-lit dot.
    peak_pt = (float(px_peak), float(py_peak))
    disc = max(contours, key=lambda c: cv2.pointPolygonTest(c, peak_pt, True))
    area = cv2.contourArea(disc)
    # Size gate: too small = noise/oblique sliver; too big = a whole-scene flash.
    if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
        return None
    # Ambiguity gate: a SECOND blob nearly as bright as the peak means two
    # markers/objects changed at once (spill, two dots) — refuse rather than
    # guess which is the real one.
    for c in contours:
        if c is disc or cv2.contourArea(c) < MIN_AREA_PX:
            continue
        m2 = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m2, [c], -1, 1, thickness=cv2.FILLED)
        if float(score[m2.astype(bool)].max()) > 0.8 * peak:
            return None
    # Magenta-balance gate (the real person-rejector): over the blob the BLUE
    # rise must be a real fraction of the RED rise. A projected magenta disc
    # lifts both; skin/clothing lifts red (and green) but little blue.
    blob = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(blob, [disc], -1, 1, thickness=cv2.FILLED)
    sel = blob.astype(bool)
    mean_r = float(d_r[sel].mean())
    mean_b = float(d_b[sel].mean())
    if mean_b < 8.0 or mean_b < MAGENTA_BLUE_RATIO * mean_r:
        return None
    m = cv2.moments(disc)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"], peak)


def median_frame(frames):
    """Per-pixel median of a list of uint8 images (robust to flicker)."""
    import numpy as np
    return np.median(np.stack(frames), axis=0).astype(frames[0].dtype)


def marker_point3(px, py, depth_frames, intr: CameraIntrinsics):
    """Median-of-frames 3D point (CAMERA frame) at a detected blob centroid."""
    depths = []
    for dm in depth_frames:
        d = sample_depth(dm, px, py, window=DEPTH_WINDOW)
        if d is not None and 0.4 < d < 8.0:
            depths.append(d)
    if not depths:
        return None
    depths.sort()
    d = depths[len(depths) // 2]
    return intr.deproject(px, py, d)


def plane_metrics(plane):
    """(width, height) of a fitted plane's u/v spans, in metres."""
    w = math.sqrt(sum(c * c for c in plane.u_vec))
    h = math.sqrt(sum(c * c for c in plane.v_vec))
    return w, h


def plane_angle_deg(pa, pb):
    """Angle between two wall planes' normals, folded into [0, 90]."""
    na, nb = pa.normal(), pb.normal()
    dot = abs(sum(a * b for a, b in zip(na, nb)))
    return math.degrees(math.acos(min(1.0, max(-1.0, dot))))


def plane_point(plane, u, v):
    return tuple(plane.origin[i] + u * plane.u_vec[i] + v * plane.v_vec[i]
                 for i in range(3))


def lateral_spread(points):
    """Second singular value of a point cloud ≈ its off-line spread (metres).

    ~0 for collinear points; a genuine 2D marker patch on a wall is > 0.3 m.
    Guards Kabsch against a degenerate (rotation-unconstrained) fit.
    """
    import numpy as np
    if len(points) < 3:
        return 0.0
    P = np.asarray(points, dtype=float)
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return float(s[1])


def constrained_corner_fit(anchor_plane, samples):
    """Fit a wall plane CONSTRAINED to form a true 90° corner with ``anchor_plane``.

    For a wall seen only obliquely by one camera, free least-squares tilts and
    stretches the plane (depth noise grows steeply with grazing angle). But
    physically we know more: the two walls meet at a right angle and both
    projected images hang level. So fix the orientation from the well-measured
    anchor wall — ``v̂`` parallel to the anchor's, ``û`` the anchor's û rotated
    90° about v̂ — and estimate only origin, width and height from ``samples``
    (linear least squares, 5 unknowns). Returns a :class:`WallPlane`.
    """
    import numpy as np

    from .geometry import WallPlane

    uA = np.asarray(anchor_plane.u_vec, dtype=float)
    vA = np.asarray(anchor_plane.v_vec, dtype=float)
    v_hat = vA / np.linalg.norm(vA)
    u_hat_A = uA / np.linalg.norm(uA)
    # Rotate û_A by ±90° about v̂ (Rodrigues); pick the sign that best matches
    # the data's own u-direction so we never flip the wall left-for-right.
    def rot(sign):
        return (np.cross(v_hat, u_hat_A) * sign
                + v_hat * np.dot(v_hat, u_hat_A))
    pts = np.array([p for (_u, _v, p) in samples], dtype=float)
    us = np.array([u for (u, _v, _p) in samples], dtype=float)
    if len(pts) >= 2 and (us.max() - us.min()) > 1e-6:
        # data's own u-direction: regress points against u
        du_dir = pts[us.argmax()] - pts[us.argmin()]
        u_hat_B = max((rot(+1), rot(-1)),
                      key=lambda c: float(np.dot(c, du_dir)))
    else:
        u_hat_B = rot(+1)
    u_hat_B = u_hat_B / np.linalg.norm(u_hat_B)

    # PIN the plane to the anchor's seam corner: the fitted plane must PASS
    # THROUGH the anchor wall's far-u top corner (the physical seam line), so
    # a coherent depth/extrinsic bias can never survive as a seam gap or a
    # normal offset. Only in-plane freedom remains: solve
    #   p_i = seam + (du + w*u_i) û_B + (dv + h*v_i) v̂
    # for (w, h, du, dv) — du/dv absorb where B's image starts vs the corner.
    o_anchor = np.asarray(anchor_plane.origin, dtype=float)
    centroid = pts.mean(axis=0)
    seam = min((o_anchor, o_anchor + uA),
               key=lambda e: float(np.linalg.norm(centroid - e)))
    vs = np.array([v for (_u, v, _p) in samples], dtype=float)
    r = (pts - seam)
    A = np.zeros((3 * len(pts), 4))
    b = r.reshape(-1)
    for i in range(len(pts)):
        A[3 * i:3 * i + 3, 0] = us[i] * u_hat_B   # w
        A[3 * i:3 * i + 3, 1] = vs[i] * v_hat     # h
        A[3 * i:3 * i + 3, 2] = u_hat_B           # du
        A[3 * i:3 * i + 3, 3] = v_hat             # dv
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    w, h, du, dv = (float(x) for x in sol)
    origin = seam + du * u_hat_B + dv * v_hat
    return WallPlane(origin=tuple(origin),
                     u_vec=tuple(u_hat_B * w),
                     v_vec=tuple(v_hat * h))


def out_of_plane_spread(points):
    """Third singular value of a point cloud ≈ its out-of-plane thickness (m).

    ~0 for coplanar points. A rigid transform anchored on coplanar
    correspondences is ill-conditioned about the in-plane axes, so a small value
    here means the registration would be a badly-tilted (unreliable) extrinsic.
    """
    import numpy as np
    if len(points) < 4:
        return 0.0
    P = np.asarray(points, dtype=float)
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return float(s[2])


def robust_register(src_pts, dst_pts, max_residual=None, min_spread=0.15,
                    inlier_tol=0.05, min_inliers=4, min_out_of_plane=0.10):
    """Kabsch (CAMERA->ROOM) via exhaustive-minimal-sample RANSAC.

    The reference camera can contribute SPURIOUS shared markers — e.g. it sees
    a *reflection* of a wall it can't view directly, a correspondence tens of cm
    wrong. With several such outliers a plain least-squares fit is so corrupted
    that iterative worst-dropping can discard the GOOD markers first, so instead
    we search: fit from every 3-correspondence minimal sample, keep the one with
    the most inliers (residual <= ``inlier_tol``), then refit on that consensus.
    Correspondence counts here are small (<= ~18), so all C(n,3) samples is cheap
    and fully deterministic. Returns ``(Extrinsic, kept, max_resid)`` or ``None``.
    """
    import itertools

    if max_residual is None:
        max_residual = MAX_RESIDUAL
    src, dst = list(src_pts), list(dst_pts)
    n = len(src)
    if n < 3 or lateral_spread(dst) < min_spread:
        return None

    best = None  # (inlier index list)
    for combo in itertools.islice(itertools.combinations(range(n), 3), 2000):
        d3 = [dst[i] for i in combo]
        if lateral_spread(d3) < min_spread * 0.5:  # collinear minimal sample
            continue
        try:
            ext = rigid_transform_from_points([src[i] for i in combo], d3)
        except Exception:  # noqa: BLE001 - degenerate sample
            continue
        inliers = [i for i in range(n)
                   if math.dist(ext.apply(src[i]), dst[i]) <= inlier_tol]
        if best is None or len(inliers) > len(best):
            best = inliers

    if best is None or len(best) < min_inliers:
        return None
    d_in = [dst[i] for i in best]
    if lateral_spread(d_in) < min_spread:
        return None
    # Refuse a registration anchored on (near-)coplanar correspondences: the
    # out-of-plane rotation is unconstrained, so the extrinsic would be tilted
    # and every live point systematically displaced (the merge-bias failure).
    # Better to leave the camera OUT of fusion than register it badly.
    if out_of_plane_spread(d_in) < min_out_of_plane:
        return None
    ext = rigid_transform_from_points([src[i] for i in best], d_in)
    max_r = max(math.dist(ext.apply(src[i]), dst[i]) for i in best)
    if max_r > max_residual:
        return None
    return ext, len(best), max_r


def reject_off_plane(samples, tol=0.05):
    """Drop (u,v,point3) samples lying > ``tol`` m off their best-fit plane.

    Catches reflections / wrong-wall / person-in-frame points before they skew
    the pooled fit. Removes the SINGLE worst point per iteration and refits, so
    one gross outlier can't tilt a plain least-squares fit into flagging the
    good points instead. Returns (kept, n_dropped); a no-op below 4 samples.
    """
    kept = list(samples)
    dropped = 0
    while len(kept) >= 4:
        plane = fit_wall_plane(kept)
        n, o = plane.normal(), plane.origin
        dists = [abs(sum(n[i] * (p[i] - o[i]) for i in range(3)))
                 for (_u, _v, p) in kept]
        worst = max(range(len(kept)), key=dists.__getitem__)
        if dists[worst] <= tol:
            break
        kept.pop(worst)
        dropped += 1
    return kept, dropped


# --------------------------------------------------------------------------- #
# capture orchestration                                                        #
# --------------------------------------------------------------------------- #
class AutoCalibrator:
    """Drives the marker sequence and computes planes + extrinsics."""

    def __init__(self, config_path: str, walls, cameras: dict,
                 cam_walls: dict | None = None):
        # cameras: cam_id -> KinectV2Source-like with .read() -> (color, depth, intr)
        # cam_walls: cam_id -> set of wall ids that camera can actually SEE. A
        #   camera mounted right beside its own wall sees the OTHER wall only as
        #   reflections; restricting it here keeps those phantoms out of the fit.
        self.config_path = config_path
        self.walls = list(walls)
        self.cameras = cameras
        self.cam_walls = ({c: set(walls) for c in cameras}
                          if cam_walls is None else
                          {c: set(cam_walls.get(c, walls)) for c in cameras})
        self.state = {"phase": "idle", "marker": None, "msg": "waiting"}
        self.status = {"progress": 0.0, "detections": {}, "report": []}
        self._lock = threading.Lock()

    # -- state served to the browser page -------------------------------- #
    def get_state(self):
        with self._lock:
            return dict(self.state)

    def get_status(self):
        with self._lock:
            return json.loads(json.dumps(self.status))

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def try_begin(self) -> bool:
        """Atomically claim the run (compare-and-set on phase).

        Prevents two POSTs racing past a check-then-spawn and running two
        marker sequences over the same (non-thread-safe) Kinect sources.
        """
        with self._lock:
            if self.state["phase"] == "running":
                return False
            self.state.update(phase="running", marker=None, msg="starting")
            return True

    def _log(self, msg):
        print(f"[autocal] {msg}", flush=True)
        with self._lock:
            self.status["report"].append(msg)

    # -- frame capture ----------------------------------------------------- #
    def _capture(self, cam_id, n_frames=4, deadline_s=4.0):
        """Grab ``n_frames`` fresh (color, depth) pairs; None on stall.

        Reads with a per-call timeout so a live-but-stalled bridge cannot
        block past the deadline (read() without one blocks indefinitely).
        """
        src = self.cameras[cam_id]
        colors, depths, intr = [], [], None
        t0 = time.monotonic()
        while len(colors) < n_frames:
            remaining = deadline_s - (time.monotonic() - t0)
            if remaining <= 0:
                break
            r = src.read(timeout=remaining)
            if r is None:
                continue  # timed out or no frame yet; deadline check exits
            color, depth, intr = r
            if color is not None and depth is not None:
                colors.append(color.copy())
                depths.append(depth.copy())
        if not colors:
            return None
        return colors, depths, intr

    def _drain(self, seconds):
        """Keep reading (and discarding) frames while the scene settles.

        The bridge streams continuously; if we simply sleep, we'd resume on
        stale buffered frames from *before* the marker changed. Bounded reads
        so one stalled camera can't starve draining the others.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            for src in self.cameras.values():
                src.read(timeout=0.1)

    # -- the sequence ------------------------------------------------------- #
    def run(self):
        try:
            self._run_inner()
        except Exception as e:  # noqa: BLE001 - report, don't die silently
            import traceback
            traceback.print_exc()
            self._set(phase="error", marker=None, msg=str(e))
            self._log(f"FAILED: {e}")

    def _run_inner(self):
        self._set(phase="running", msg="warming up cameras")
        cam_ids = list(self.cameras)
        # First read spawns each bridge; allow a generous first-frame window so
        # marker #1 doesn't read as a "stall" while the Kinects boot.
        for c in cam_ids:
            got = self._capture(c, n_frames=1, deadline_s=20.0)
            if got is None:
                raise RuntimeError(f"camera {c} produced no frames in 20 s — "
                                   f"is it plugged in and powered?")
            self._log(f"{c}: streaming")
        self._set(msg="capturing")
        # samples[wall][cam_id] = list of (u, v, point3-in-that-camera-frame)
        samples = {w: {c: [] for c in cam_ids} for w in self.walls}
        intrinsics: dict = {}
        total = len(self.walls) * len(MARKER_GRID)
        done = 0

        for wall in self.walls:
            for (u, v) in MARKER_GRID:
                # OFF baseline
                self._set(marker=None)
                self._drain(0.6)
                off = {c: self._capture(c, n_frames=3) for c in cam_ids}
                # ON
                self._set(marker={"wall": wall, "u": u, "v": v})
                self._drain(0.9)
                on = {c: self._capture(c, n_frames=4) for c in cam_ids}

                for c in cam_ids:
                    if off[c] is None or on[c] is None:
                        self._log(f"{wall}({u:.2f},{v:.2f}) {c}: camera stalled")
                        continue
                    off_med = median_frame(off[c][0])
                    on_med = median_frame(on[c][0])
                    intrinsics[c] = on[c][2]
                    # A camera that can't see this wall would only detect a
                    # REFLECTION of its dot — skip so phantoms never enter the fit.
                    if wall not in self.cam_walls[c]:
                        continue
                    det = detect_marker(off_med, on_med)
                    if det is None:
                        continue
                    px, py, peak = det
                    p3 = marker_point3(px, py, on[c][1], intrinsics[c])
                    if p3 is None:
                        self._log(f"{wall}({u:.2f},{v:.2f}) {c}: blob but no depth")
                        continue
                    samples[wall][c].append((u, v, p3))
                done += 1
                with self._lock:
                    self.status["progress"] = done / total
                    self.status["detections"] = {
                        w: {c: len(samples[w][c]) for c in cam_ids}
                        for w in self.walls}

        self._set(marker=None, msg="solving")
        counts = {w: {c: len(samples[w][c]) for c in cam_ids} for w in self.walls}
        self._log(f"detections: {counts}")

        # Reference camera = the one that sees the walls most SQUARELY, not the
        # one with the most dots. A camera viewing a wall edge-on still detects
        # its dots but collapses the plane to ~0 width; such a view must never
        # anchor the room frame. Score each camera by how many walls it fits to
        # a sane (non-degenerate) plane, tie-broken by total markers.
        def _wallish(cam):
            good = 0
            for w in self.walls:
                s = samples[w][cam]
                if len(s) >= 4:
                    pw, ph = plane_metrics(fit_wall_plane(s))
                    if WIDTH_RANGE[0] <= pw <= WIDTH_RANGE[1] \
                            and HEIGHT_RANGE[0] <= ph <= HEIGHT_RANGE[1]:
                        good += 1
            return good
        ref_cam = max(cam_ids,
                      key=lambda c: (_wallish(c),
                                     sum(len(samples[w][c]) for w in self.walls)))
        self._log(f"reference camera (room frame): {ref_cam} "
                  f"(sees {_wallish(ref_cam)} wall(s) squarely)")

        # --- per-camera per-wall outlier rejection ------------------------- #
        # tol=0.10: reflections/persons sit MUCH further off-plane than this,
        # while honest oblique-view depth noise at 4 m can reach ~8 cm — a
        # tighter tol throws away real markers a sole camera can't spare.
        for wall in self.walls:
            for c in cam_ids:
                kept, dropped = reject_off_plane(samples[wall][c], tol=0.10)
                if dropped:
                    self._log(f"wall {wall} {c}: dropped {dropped} off-plane "
                              f"(reflection/person/wrong-wall) sample(s)")
                samples[wall][c] = kept

        # --- second camera extrinsic via shared markers -------------------- #
        extrinsics = {ref_cam: Extrinsic.identity()}
        second = [c for c in cam_ids if c != ref_cam]
        for c in second:
            src_pts, dst_pts = [], []
            for wall in self.walls:
                ref_by_uv = {(u, v): p for (u, v, p) in samples[wall][ref_cam]}
                for (u, v, p_cam) in samples[wall][c]:
                    if (u, v) in ref_by_uv:
                        src_pts.append(p_cam)
                        dst_pts.append(ref_by_uv[(u, v)])
            reg = robust_register(src_pts, dst_pts)
            if reg is None:
                self._log(f"{c}: {len(src_pts)} shared markers — too few, "
                          f"(near-)collinear, or residual too high after "
                          f"dropping reflections; leaving it out of fusion")
                continue
            ext, kept, max_r = reg
            self._log(f"{c}: registered from {kept}/{len(src_pts)} markers "
                      f"(dropped {len(src_pts) - kept} outlier/reflection), "
                      f"max residual {max_r * 100:.1f} cm")
            extrinsics[c] = ext

        def pooled_for(wall):
            pool = list(samples[wall][ref_cam])
            for c in second:
                if c in extrinsics:
                    pool.extend((u, v, extrinsics[c].apply(p))
                                for (u, v, p) in samples[wall][c])
            return pool

        # --- save raw samples for offline debugging / refits ---------------- #
        try:
            dump = {w: {c: [[u, v, list(p)] for (u, v, p) in samples[w][c]]
                        for c in cam_ids} for w in self.walls}
            Path("autocal_samples.json").write_text(json.dumps(dump, indent=1))
            self._log("raw samples saved to autocal_samples.json")
        except Exception:  # noqa: BLE001 - debugging aid only
            pass

        # --- plane fits: anchor on the best-seen wall ---------------------- #
        # Free-fit every wall. A wall seen edge-on collapses to ~0 width, so we
        # classify fits as SANE or degenerate. The best-seen sane wall anchors
        # the corner; each other wall keeps its own free fit ONLY if it agrees
        # independently (sane, ~90° to the anchor, small seam gap), otherwise it
        # is re-fit CONSTRAINED to the anchor at exactly 90°. A good wall is
        # thus never wrecked by a degenerate neighbour.
        def _sane(plane):
            pw, ph = plane_metrics(plane)
            return (WIDTH_RANGE[0] <= pw <= WIDTH_RANGE[1]
                    and HEIGHT_RANGE[0] <= ph <= HEIGHT_RANGE[1])

        def _seam_gap(pa, pb):
            return min(math.dist(plane_point(pa, ea, 0.5),
                                 plane_point(pb, eb, 0.5))
                       for ea in (0.0, 1.0) for eb in (0.0, 1.0))

        pools = {w: reject_off_plane(pooled_for(w))[0] for w in self.walls}
        free = {w: fit_wall_plane(pools[w])
                for w in self.walls if len(pools[w]) >= 4}
        sane = {w: p for w, p in free.items() if _sane(p)}
        if not sane:
            raise RuntimeError(
                "no wall produced a sane plane — a camera is probably seeing "
                "its wall edge-on; nudge it to face the wall more squarely")
        anchor_wall = max(sane, key=lambda w: len(pools[w]))
        planes = {anchor_wall: sane[anchor_wall]}
        aw, ah = plane_metrics(planes[anchor_wall])
        self._log(f"anchor wall {anchor_wall}: {len(pools[anchor_wall])} "
                  f"markers -> width {aw:.2f} m, height {ah:.2f} m")

        for wall in self.walls:
            if wall == anchor_wall:
                continue
            pool, anchor = pools[wall], planes[anchor_wall]
            if (wall in sane
                    and 85.0 <= plane_angle_deg(sane[wall], anchor) <= 95.0
                    and _seam_gap(sane[wall], anchor) <= 0.15):
                planes[wall] = sane[wall]
                w, h = plane_metrics(planes[wall])
                self._log(f"wall {wall}: {len(pool)} markers -> width {w:.2f} m,"
                          f" height {h:.2f} m (free fit agrees with corner)")
                continue
            if len(pool) < 3:
                raise RuntimeError(
                    f"wall {wall}: only {len(pool)} usable markers — check the "
                    f"autocal page is fullscreen on that projector and faces a "
                    f"camera")
            plane0 = constrained_corner_fit(anchor, pool)
            n, o = plane0.normal(), plane0.origin
            kept = [(u, v, p) for (u, v, p) in pool
                    if abs(sum(n[i] * (p[i] - o[i]) for i in range(3))) <= 0.10]
            if 3 <= len(kept) < len(pool):
                self._log(f"wall {wall}: dropped {len(pool) - len(kept)} "
                          f"sample(s) off the constrained plane")
                plane0 = constrained_corner_fit(anchor, kept)
            planes[wall] = plane0
            w, h = plane_metrics(plane0)
            self._log(f"wall {wall}: corner-constrained to {anchor_wall} at 90° "
                      f"-> width {w:.2f} m, height {h:.2f} m")

        # --- sanity gates --------------------------------------------------- #
        problems = []
        for wall, plane in planes.items():
            w, h = plane_metrics(plane)
            if not (WIDTH_RANGE[0] <= w <= WIDTH_RANGE[1]):
                problems.append(f"wall {wall} width {w:.2f} m out of range")
            if not (HEIGHT_RANGE[0] <= h <= HEIGHT_RANGE[1]):
                problems.append(f"wall {wall} height {h:.2f} m out of range")
        if len(planes) == 2:
            a, b = (planes[w] for w in self.walls[:2])
            ang = plane_angle_deg(a, b)
            self._log(f"angle between walls: {ang:.1f} deg")
            if not (ANGLE_RANGE[0] <= ang <= ANGLE_RANGE[1]):
                problems.append(f"wall angle {ang:.1f} deg not corner-like")
        if problems:
            raise RuntimeError("; ".join(problems))

        # --- write the config ------------------------------------------------ #
        cfg = load_config_dict(self.config_path)
        for wall, plane in planes.items():
            cfg = merge_wall_plane(cfg, wall, plane)
        for c in cam_ids:
            if c in extrinsics and c in intrinsics:
                cfg = merge_camera_pose(cfg, c, intrinsics[c], extrinsics[c],
                                        kind="kinect_v2")
                # serve only the walls this camera actually PROVED it sees
                # (>= 3 detected markers) and that got calibrated — a camera
                # never drives a wall it sees only by reflection or edge-on.
                cfg["cameras"][c]["serves"] = [
                    w for w in self.walls
                    if w in self.cam_walls[c] and w in planes
                    and len(samples[w][c]) >= 3]
            elif c not in extrinsics:
                cfg["cameras"][c]["serves"] = []  # unregistered: keep out
        RoomConfig.from_dict(cfg)  # validate before persisting
        save_config_dict(self.config_path, cfg)
        self._log(f"wrote {self.config_path}")
        with self._lock:
            self.status["progress"] = 1.0
        self._set(phase="done", marker=None, msg="ok")


# --------------------------------------------------------------------------- #
# HTTP plumbing                                                                #
# --------------------------------------------------------------------------- #
def make_handler(web_dir: str, calib: AutoCalibrator):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=web_dir, **kw)

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/calib/state"):
                return self._json(calib.get_state())
            if self.path.startswith("/calib/status"):
                return self._json(calib.get_status())
            # --- debug: manually hold/clear a marker (diagnostics only) ------ #
            if self.path.startswith("/calib/hold"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                calib._set(phase="running", marker={
                    "wall": q["wall"][0],
                    "u": float(q["u"][0]), "v": float(q["v"][0])})
                return self._json({"ok": True})
            if self.path.startswith("/calib/clear"):
                calib._set(phase="running", marker=None)
                return self._json({"ok": True})
            if self.path.startswith("/calib/idle"):
                calib._set(phase="idle", marker=None)
                return self._json({"ok": True})
            return super().do_GET()

        def do_POST(self):  # noqa: N802
            if self.path.startswith("/calib/start"):
                if not calib.try_begin():  # atomic compare-and-set
                    return self._json({"ok": False, "msg": "already running"})
                threading.Thread(target=calib.run, daemon=True).start()
                return self._json({"ok": True})
            return self._json({"ok": False, "msg": "unknown endpoint"}, 404)

        def log_message(self, fmt, *args):  # quiet the per-poll request noise
            pass

    Handler.timeout = 5  # a dead client can't pin a handler thread forever
    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--web-dir", default=str(Path(__file__).parent.parent / "web"))
    args = ap.parse_args(argv)

    cfg = RoomConfig.from_dict(load_config_dict(args.config))
    from .kinect import KinectV2Source  # lazy: spawns bridges

    cameras = {cam_id: KinectV2Source(device_index=cam.device)
               for cam_id, cam in cfg.cameras.items()}
    walls = list(cfg.walls)
    # A camera's `serves` list doubles as "which walls it can SEE" — a camera
    # beside its own wall serves only that wall, so its view of the other wall
    # (reflections) is excluded from calibration. Empty serves = sees all.
    cam_walls = {cid: (set(cam.serves) if cam.serves else set(walls))
                 for cid, cam in cfg.cameras.items()}
    for cid, ws in cam_walls.items():
        print(f"[autocal] {cid} calibrates walls: {sorted(ws)}")
    calib = AutoCalibrator(args.config, walls, cameras, cam_walls=cam_walls)

    httpd = ThreadingHTTPServer(("", args.port),
                                make_handler(args.web_dir, calib))
    print(f"[autocal] serving on http://localhost:{args.port}")
    for w in walls:
        print(f"[autocal]   open http://localhost:{args.port}/autocal.html?wall={w}")
    print(f"[autocal] then: curl -X POST http://localhost:{args.port}/calib/start")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for src in cameras.values():
            try:
                src.close()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
