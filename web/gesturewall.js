// Gesture Wall — browser port of the Stack A pipeline.
//
// Ports the exact selection logic from the Python prototype so behaviour matches:
//   pose wrist -> mirror -> homography (calibration) -> 1-Euro smoothing
//   -> DwellSelector -> zone toggle, with raise-hand-to-engage gating.
//
// Pure-logic classes (OneEuroFilter, Zone, DwellSelector, Homography) are direct
// translations of gesturewall/{filters,zones,dwell,calibration}.py.

import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// BlazePose 33-landmark indices (same order as the Tasks API flat list).
const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
const LEFT_WRIST = 15, RIGHT_WRIST = 16;

const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/" +
  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// --------------------------------------------------------------------------- //
// 1-Euro filter  (port of filters.py)
// --------------------------------------------------------------------------- //
class LowPassFilter {
  constructor(alpha) { this._setAlpha(alpha); this._s = null; }
  _setAlpha(a) {
    if (!(a > 0 && a <= 1)) throw new Error(`alpha must be in (0,1], got ${a}`);
    this._alpha = a;
  }
  call(value, alpha) {
    if (alpha != null) this._setAlpha(alpha);
    const s = this._s == null ? value : this._alpha * value + (1 - this._alpha) * this._s;
    this._s = s;
    return s;
  }
  last() { return this._s; }
}

class OneEuroFilter {
  constructor(freq = 60, mincutoff = 1.0, beta = 0.0, dcutoff = 1.0) {
    this._freq = freq; this._mincutoff = mincutoff; this._beta = beta; this._dcutoff = dcutoff;
    this._x = new LowPassFilter(this._alpha(mincutoff));
    this._dx = new LowPassFilter(this._alpha(dcutoff));
    this._lasttime = null;
  }
  _alpha(cutoff) {
    const te = 1 / this._freq;
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / te);
  }
  call(x, timestamp) {
    if (this._lasttime != null && timestamp != null && timestamp > this._lasttime)
      this._freq = 1 / (timestamp - this._lasttime);
    this._lasttime = timestamp;
    const prev = this._x.last();
    const dx = prev == null ? 0 : (x - prev) * this._freq;
    const edx = this._dx.call(dx, this._alpha(this._dcutoff));
    const cutoff = this._mincutoff + this._beta * Math.abs(edx);
    return this._x.call(x, this._alpha(cutoff));
  }
}

class Point2DFilter {
  constructor(freq = 60, mincutoff = 1.0, beta = 0.007, dcutoff = 1.0) {
    this._fx = new OneEuroFilter(freq, mincutoff, beta, dcutoff);
    this._fy = new OneEuroFilter(freq, mincutoff, beta, dcutoff);
  }
  call(x, y, timestamp) {
    return [this._fx.call(x, timestamp), this._fy.call(y, timestamp)];
  }
}

// --------------------------------------------------------------------------- //
// Zones  (port of zones.py)
// --------------------------------------------------------------------------- //
class Zone {
  constructor(id, label, x, y, w, h) {
    this.id = id; this.label = label;
    this.x = x; this.y = y; this.w = w; this.h = h;
    this.selected = false;
  }
  contains(px, py, margin = 0) {
    const mx = margin * this.w, my = margin * this.h;
    return (this.x + mx <= px && px <= this.x + this.w - mx &&
            this.y + my <= py && py <= this.y + this.h - my);
  }
}

function buildGrid(rows, cols, padding = 0.06, labels = null) {
  if (rows < 1 || cols < 1) throw new Error("rows and cols must be >= 1");
  if (!(padding >= 0 && padding < 0.5)) throw new Error("padding must be in [0, 0.5)");
  const zones = [];
  const cellW = 1 / cols, cellH = 1 / rows;
  let idx = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cellW + padding * cellW;
      const y = r * cellH + padding * cellH;
      const w = cellW * (1 - 2 * padding);
      const h = cellH * (1 - 2 * padding);
      const label = labels && idx < labels.length ? labels[idx] : String(idx + 1);
      zones.push(new Zone(`r${r}c${c}`, label, x, y, w, h));
      idx++;
    }
  }
  return zones;
}

// --------------------------------------------------------------------------- //
// Dwell-to-select state machine  (port of dwell.py)
// --------------------------------------------------------------------------- //
class DwellSelector {
  constructor(dwellSeconds = 0.8, cooldownSeconds = 0.4, hysteresis = 0.15) {
    if (dwellSeconds <= 0) throw new Error("dwellSeconds must be > 0");
    if (cooldownSeconds < 0) throw new Error("cooldownSeconds must be >= 0");
    if (!(hysteresis >= 0 && hysteresis < 0.5)) throw new Error("hysteresis must be in [0, 0.5)");
    this.dwellSeconds = dwellSeconds;
    this.cooldownSeconds = cooldownSeconds;
    this.hysteresis = hysteresis;
    this.activeZone = null;
    this.progress = 0;
    this._enterTime = null;
    this._cooldownUntil = 0;
  }
  reset() { this.activeZone = null; this.progress = 0; this._enterTime = null; }
  _resolveTarget(zones, x, y) {
    if (this.activeZone && this.activeZone.contains(x, y, -this.hysteresis))
      return this.activeZone;
    const core = zones.find(z => z.contains(x, y, this.hysteresis));
    if (core) return core;
    return zones.find(z => z.contains(x, y)) || null;
  }
  update(zones, cursor, t, engaged = true) {
    if (!engaged || cursor == null) { this.reset(); return null; }
    if (t < this._cooldownUntil) {
      this.activeZone = null; this.progress = 0; this._enterTime = null; return null;
    }
    const target = this._resolveTarget(zones, cursor[0], cursor[1]);
    if (target == null) { this.reset(); return null; }
    if (target !== this.activeZone) {
      this.activeZone = target; this._enterTime = t; this.progress = 0; return null;
    }
    const elapsed = t - this._enterTime;
    this.progress = Math.max(0, Math.min(1, elapsed / this.dwellSeconds));
    if (elapsed >= this.dwellSeconds) {
      target.selected = !target.selected;
      const event = { zoneId: target.id, selected: target.selected };
      this._cooldownUntil = t + this.cooldownSeconds;
      this.reset();
      return event;
    }
    return null;
  }
}

// --------------------------------------------------------------------------- //
// Homography  (port of calibration.py, with an in-JS getPerspectiveTransform)
// --------------------------------------------------------------------------- //
const WALL_CORNERS = [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]];
const CORNER_NAMES = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"];

class Homography {
  constructor(matrix = null) {
    this.matrix = matrix || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  }
  static identity() { return new Homography(); }
  apply(x, y) {
    const m = this.matrix;
    const denom = m[2][0] * x + m[2][1] * y + m[2][2];
    if (Math.abs(denom) < 1e-12) return [x, y];
    return [(m[0][0] * x + m[0][1] * y + m[0][2]) / denom,
            (m[1][0] * x + m[1][1] * y + m[1][2]) / denom];
  }
  static fromCornerPoints(src, dst = WALL_CORNERS) {
    if (src.length !== 4) throw new Error("exactly 4 source points are required");
    let area = 0;
    for (let i = 0; i < 4; i++) {
      const [x1, y1] = src[i], [x2, y2] = src[(i + 1) % 4];
      area += x1 * y2 - x2 * y1;
    }
    if (Math.abs(area) / 2 < 1e-6)
      throw new Error("source points are degenerate (collinear/coincident)");
    return new Homography(getPerspectiveTransform(src, dst));
  }
}

// Solve the 8 homography params (h33 = 1) from 4 point correspondences.
function getPerspectiveTransform(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -x * v, -y * v]); b.push(v);
  }
  const h = solveLinear(A, b);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

// Gaussian elimination with partial pivoting for an n x n system.
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col];
    if (Math.abs(d) < 1e-12) throw new Error("singular system in homography solve");
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / d;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// --------------------------------------------------------------------------- //
// Pose source — webcam + MediaPipe Tasks PoseLandmarker  (port of PoseSource)
// --------------------------------------------------------------------------- //
class PoseSource {
  constructor(landmarker, video, { mirror = true } = {}) {
    this.landmarker = landmarker;
    this.video = video;
    this.mirror = mirror;
    this._lastTs = -1;
  }
  read() {
    const v = this.video;
    if (v.readyState < 2) return { pointer: null, engaged: false, status: "no_frame" };
    let ts = Math.round(performance.now());
    if (ts <= this._lastTs) ts = this._lastTs + 1;
    this._lastTs = ts;
    const result = this.landmarker.detectForVideo(v, ts);
    const lms = result.landmarks;
    if (!lms || lms.length === 0)
      return { pointer: null, engaged: false, status: "no_pose" };
    const lm = lms[0];
    const rw = lm[RIGHT_WRIST], lw = lm[LEFT_WRIST];
    // Pick the higher (more raised) wrist; image y grows downward.
    let wrist, shoulder;
    if (rw.y <= lw.y) { wrist = rw; shoulder = lm[RIGHT_SHOULDER]; }
    else { wrist = lw; shoulder = lm[LEFT_SHOULDER]; }
    const visible = (wrist.visibility ?? 1.0) >= 0.5;
    const engaged = visible && wrist.y < shoulder.y;
    // Mirror so moving right -> cursor right (Python flips the frame pre-detect).
    const px = this.mirror ? 1 - wrist.x : wrist.x;
    return { pointer: [px, wrist.y], engaged, status: "ok" };
  }
}

// Mouse source for camera-free testing of the full pipeline.
class MouseSource {
  constructor() { this.pointer = null; this.engaged = false; }
  setPointer(x, y) { this.pointer = [x, y]; this.engaged = true; }
  read() { return { pointer: this.pointer, engaged: this.engaged, status: "mouse" }; }
}

// --------------------------------------------------------------------------- //
// App
// --------------------------------------------------------------------------- //
const COLORS = {
  bg: "#18181c", zoneIdle: "#5a5a60", zoneSelected: "#46aa46",
  zoneActive: "#3cc8dc", text: "#ebebeb", cursor: "#3cc8dc",
  ringBg: "#46464c", ringFg: "#3cdcf0",
};

class App {
  constructor() {
    this.canvas = document.getElementById("wall");
    this.ctx = this.canvas.getContext("2d");
    this.video = document.getElementById("cam");
    this.preview = document.getElementById("preview");
    this.previewCtx = this.preview.getContext("2d");
    this.status = document.getElementById("status");

    this.rows = 2; this.cols = 3;
    this.padding = 0.06;
    this.dwell = 0.8; this.cooldown = 0.4; this.hysteresis = 0.15;
    this.minCutoff = 1.0; this.beta = 0.007; this.useFilter = true;
    this.mirror = true; this.showPreview = true;

    this.source = new MouseSource();   // start in mouse mode until camera starts
    this.mode = "MOUSE TEST";
    this.homography = Homography.identity();
    this._loadCalibration();

    this.landmarker = null;
    this.calibrating = false;
    this.calibCaptured = [];
    this.fps = 0; this._prev = performance.now() / 1000;

    this._rebuild();
    this._bindUI();
    this._resize();
    window.addEventListener("resize", () => this._resize());
    requestAnimationFrame(() => this._frame());
  }

  _rebuild() {
    this.zones = buildGrid(this.rows, this.cols, this.padding);
    this.selector = new DwellSelector(this.dwell, this.cooldown, this.hysteresis);
    this.pfilter = this.useFilter ? new Point2DFilter(60, this.minCutoff, this.beta) : null;
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(window.innerWidth * dpr);
    this.canvas.height = Math.round(window.innerHeight * dpr);
    this.canvas.style.width = window.innerWidth + "px";
    this.canvas.style.height = window.innerHeight + "px";
  }

  // --- input -------------------------------------------------------------
  async startCamera() {
    this.setStatus("starting camera…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      this.video.srcObject = stream;
      await this.video.play();
    } catch (e) {
      this.setStatus(`camera blocked: ${e.name}. Allow camera access and retry.`);
      return;
    }
    if (!this.landmarker) {
      this.setStatus("loading pose model…");
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
      this.landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }
    this.source = new PoseSource(this.landmarker, this.video, { mirror: this.mirror });
    this.mode = "POSE";
    this.setStatus("pose mode — raise a hand above your shoulder to engage");
  }

  useMouse() {
    this.source = new MouseSource();
    this.mode = "MOUSE TEST";
    this.setStatus("mouse test mode — move over a tile and hold still");
  }

  // --- calibration -------------------------------------------------------
  startCalibration() {
    this.calibrating = true;
    this.calibCaptured = [];
    this.setStatus(`calibration: point at the ${CORNER_NAMES[0]} corner, then press SPACE`);
  }

  _tickCalibration(read) {
    const clamp = v => Math.min(1, Math.max(0, v));
    const raw = read.pointer ? [clamp(read.pointer[0]), clamp(read.pointer[1])] : null;
    this._drawCalibration(raw);
  }

  _doCapture(raw) {
    this.calibCaptured.push([Math.min(1, Math.max(0, raw[0])), Math.min(1, Math.max(0, raw[1]))]);
    const n = this.calibCaptured.length;
    if (n === 4) {
      try {
        this.homography = Homography.fromCornerPoints(this.calibCaptured);
        this._saveCalibration();
        this.setStatus("calibration saved ✓");
      } catch (e) {
        this.setStatus(`calibration failed: ${e.message} — re-run and move clearly to each corner`);
      }
      this.calibrating = false;
    } else {
      this.setStatus(`captured ${n}/4 — now point at the ${CORNER_NAMES[n]} corner and hold`);
    }
  }

  _loadCalibration() {
    try {
      const raw = localStorage.getItem("gesturewall.calibration");
      if (raw) this.homography = new Homography(JSON.parse(raw).matrix);
    } catch { /* ignore */ }
  }
  _saveCalibration() {
    localStorage.setItem("gesturewall.calibration",
      JSON.stringify({ matrix: this.homography.matrix }));
  }
  resetCalibration() {
    this.homography = Homography.identity();
    localStorage.removeItem("gesturewall.calibration");
    this.setStatus("calibration reset to identity");
  }

  resetSelections() {
    for (const z of this.zones) z.selected = false;
    this.selector.reset();
  }

  // --- main loop ---------------------------------------------------------
  _frame() {
    const t = performance.now() / 1000;
    const read = this.source.read();
    this._lastRead = read;
    const { pointer, engaged } = read;

    let cursor = null;
    if (pointer != null) {
      let [wx, wy] = this.homography.apply(pointer[0], pointer[1]);
      if (this.pfilter) [wx, wy] = this.pfilter.call(wx, wy, t);
      cursor = [Math.min(1, Math.max(0, wx)), Math.min(1, Math.max(0, wy))];
    }

    if (this.calibrating) {
      this._tickCalibration(read);
    } else {
      const event = this.selector.update(this.zones, cursor, t, engaged);
      if (event) console.log(`[gesturewall] ${event.selected ? "SELECT" : "DESELECT"} ${event.zoneId}`);
      this._draw(cursor, engaged);
    }

    if (this.showPreview && this.mode === "POSE") this._drawPreview();
    else this.preview.style.display = "none";

    const dt = t - this._prev; this._prev = t;
    if (dt > 0) this.fps = 0.9 * this.fps + 0.1 * (1 / dt);
    requestAnimationFrame(() => this._frame());
  }

  // --- drawing -----------------------------------------------------------
  _draw(cursor, engaged) {
    const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, W, H);

    for (const z of this.zones) {
      const x1 = z.x * W, y1 = z.y * H, w = z.w * W, h = z.h * H;
      const isActive = this.selector.activeZone === z;
      if (z.selected) { ctx.fillStyle = COLORS.zoneSelected; ctx.fillRect(x1, y1, w, h); }
      ctx.lineWidth = isActive ? 6 : 3;
      ctx.strokeStyle = isActive ? COLORS.zoneActive : (z.selected ? COLORS.zoneSelected : COLORS.zoneIdle);
      ctx.strokeRect(x1, y1, w, h);
      ctx.fillStyle = COLORS.text;
      ctx.font = `${Math.round(H * 0.05)}px system-ui, sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(z.label, x1 + w / 2, y1 + h / 2);
    }

    if (engaged && cursor) this._drawCursor(cursor[0] * W, cursor[1] * H, this.selector.progress);

    const statusTxt = engaged ? "ENGAGED" : "idle (raise hand / move mouse in)";
    ctx.fillStyle = COLORS.text;
    ctx.font = `${Math.round(H * 0.025)}px system-ui, sans-serif`;
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(`${this.mode} | ${statusTxt} | ${this.fps.toFixed(1)} fps`, W * 0.012, H * 0.02);
  }

  _drawCursor(cx, cy, progress) {
    const ctx = this.ctx, r = Math.round(this.canvas.height * 0.03);
    ctx.lineWidth = 4; ctx.strokeStyle = COLORS.ringBg;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    if (progress > 0) {
      ctx.lineWidth = 7; ctx.strokeStyle = COLORS.ringFg;
      ctx.beginPath();
      ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress);
      ctx.stroke();
    }
    ctx.fillStyle = COLORS.cursor;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(5, r * 0.22), 0, Math.PI * 2); ctx.fill();
  }

  // Interactive corner calibration: a pulsing labeled target shows which corner
  // to point at, others are dim/numbered, captured ones get a green check, and a
  // guide line connects the live cursor to the active target. Press SPACE to capture.
  _drawCalibration(raw) {
    const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
    const pulse = 0.5 + 0.5 * Math.sin(performance.now() * 0.006);
    const idx = this.calibCaptured.length;
    const px = c => [c[0] * W, c[1] * H];
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, W, H);

    // Faint wall boundary through the four target corners.
    ctx.strokeStyle = "#3a3a44"; ctx.lineWidth = 2; ctx.setLineDash([10, 10]);
    ctx.beginPath();
    WALL_CORNERS.forEach((c, i) => { const [x, y] = px(c); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);

    // The "reach quad" forming from captured points (+ live cursor as next vertex).
    if (idx > 0) {
      const pts = this.calibCaptured.map(px);
      const poly = raw ? [...pts, px(raw)] : pts;
      ctx.beginPath();
      poly.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      if (idx >= 2) { ctx.closePath(); ctx.fillStyle = "rgba(60,200,220,0.08)"; ctx.fill(); }
      ctx.strokeStyle = "rgba(60,200,220,0.5)"; ctx.lineWidth = 2; ctx.stroke();
    }

    // Target corner markers.
    WALL_CORNERS.forEach((c, i) => {
      const [x, y] = px(c);
      const dir = [Math.sign(0.5 - c[0]) || 1, Math.sign(0.5 - c[1]) || 1];
      if (i < idx) {                                   // already captured -> green check
        ctx.fillStyle = COLORS.zoneSelected;
        ctx.beginPath(); ctx.arc(x, y, H * 0.018, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#e6ffe6"; ctx.lineWidth = 4; ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(x - H * 0.008, y); ctx.lineTo(x - H * 0.002, y + H * 0.007);
        ctx.lineTo(x + H * 0.009, y - H * 0.008); ctx.stroke();
      } else if (i === idx) {                          // active -> pulsing rings + label
        for (let k = 0; k < 3; k++) {
          const rr = H * 0.035 * (0.6 + 0.5 * k) + pulse * H * 0.012;
          ctx.strokeStyle = `rgba(60,220,240,${0.75 - 0.22 * k})`;
          ctx.lineWidth = 4; ctx.beginPath(); ctx.arc(x, y, rr, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.fillStyle = COLORS.ringFg;
        ctx.beginPath(); ctx.arc(x, y, H * 0.01, 0, Math.PI * 2); ctx.fill();
        ctx.font = `bold ${Math.round(H * 0.045)}px system-ui, sans-serif`;
        ctx.textAlign = dir[0] > 0 ? "left" : "right"; ctx.textBaseline = "middle";
        ctx.fillStyle = COLORS.ringFg;
        ctx.fillText(CORNER_NAMES[i], x + dir[0] * H * 0.06, y + dir[1] * H * 0.06);
      } else {                                         // pending -> dim numbered ring
        ctx.strokeStyle = "#55555f"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x, y, H * 0.02, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "#8a8a96"; ctx.font = `${Math.round(H * 0.025)}px system-ui, sans-serif`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), x, y);
      }
    });

    // Live cursor (the detected wrist) + a guide line to the active target.
    if (raw) {
      const [cx, cy] = px(raw), [tx, ty] = px(WALL_CORNERS[idx]);
      ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 2; ctx.setLineDash([6, 8]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
      this._drawCursor(cx, cy, 0);
    }

    // Header + sub-instruction + footer.
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.font = `bold ${Math.round(H * 0.034)}px system-ui, sans-serif`;
    ctx.fillStyle = COLORS.text;
    ctx.fillText(`Calibration — Corner ${idx + 1} of 4`, W / 2, H * 0.03);
    ctx.font = `${Math.round(H * 0.026)}px system-ui, sans-serif`;
    ctx.fillStyle = raw ? COLORS.ringFg : "#9a9ad0";
    ctx.fillText(
      raw ? `Point at the ${CORNER_NAMES[idx]} corner, then press SPACE`
          : (this.mode === "POSE" ? "Step into the camera view" : "Move the mouse to begin"),
      W / 2, H * 0.03 + H * 0.05);
    ctx.textBaseline = "bottom";
    ctx.font = `${Math.round(H * 0.022)}px system-ui, sans-serif`;
    ctx.fillStyle = "#bcbcc6";
    ctx.fillText("Point at the highlighted corner · press SPACE to capture · Esc to cancel", W / 2, H - H * 0.03);
  }

  _drawPreview() {
    const v = this.video;
    if (v.readyState < 2) { this.preview.style.display = "none"; return; }
    this.preview.style.display = "block";
    const pw = this.preview.width, ph = this.preview.height;
    this.previewCtx.save();
    if (this.mirror) { this.previewCtx.translate(pw, 0); this.previewCtx.scale(-1, 1); }
    this.previewCtx.drawImage(v, 0, 0, pw, ph);
    this.previewCtx.restore();
  }

  setStatus(msg) { this.status.textContent = msg; }

  // --- UI ----------------------------------------------------------------
  _bindUI() {
    const $ = id => document.getElementById(id);
    $("startCam").onclick = () => this.startCamera();
    $("useMouse").onclick = () => this.useMouse();
    $("calibrate").onclick = () => this.startCalibration();
    $("resetCalib").onclick = () => this.resetCalibration();
    $("reset").onclick = () => this.resetSelections();
    $("fullscreen").onclick = () => this._toggleFullscreen();

    const bind = (id, key, parse, rebuild = true) => {
      const el = $(id);
      el.oninput = () => {
        this[key] = parse(el.value);
        const out = $(id + "Val"); if (out) out.textContent = el.value;
        if (rebuild) this._rebuild();
      };
    };
    bind("rows", "rows", v => parseInt(v));
    bind("cols", "cols", v => parseInt(v));
    bind("dwell", "dwell", v => parseFloat(v));
    bind("minCutoff", "minCutoff", v => parseFloat(v));
    bind("beta", "beta", v => parseFloat(v));

    $("mirror").onchange = e => {
      this.mirror = e.target.checked;
      if (this.source instanceof PoseSource) this.source.mirror = this.mirror;
    };
    $("filter").onchange = e => { this.useFilter = e.target.checked; this._rebuild(); };
    $("previewToggle").onchange = e => { this.showPreview = e.target.checked; };

    // Mouse drives the cursor in mouse-test mode.
    this.canvas.addEventListener("mousemove", e => {
      if (this.source instanceof MouseSource)
        this.source.setPointer(e.clientX / window.innerWidth, e.clientY / window.innerHeight);
    });

    window.addEventListener("keydown", e => {
      if (e.key === "r") this.resetSelections();
      else if (e.key === "c") this.startCalibration();
      else if (e.key === "f") this._toggleFullscreen();
      else if (e.key === " " && this.calibrating) {
        e.preventDefault();
        // Capture the detected wrist regardless of the raise-hand engage gate —
        // the bottom corners need you to point low (hand below the shoulder).
        const r = this._lastRead?.pointer;
        if (r) this._doCapture(r);
        else this.setStatus(this.mode === "POSE" ? "step into the camera view to capture" : "move the mouse to capture");
      }
      else if (e.key === "Escape" && this.calibrating) { this.calibrating = false; this.setStatus("calibration cancelled"); }
    });
  }

  _toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      document.getElementById("panel").classList.remove("pinned");  // auto-hide while projecting
    } else {
      document.exitFullscreen();
    }
  }
}

window.addEventListener("DOMContentLoaded", () => new App());
