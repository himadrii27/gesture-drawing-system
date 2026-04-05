import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm";

// ── Constants ────────────────────────────────────────────────────────────────
const FRAME_W = 860;
const FRAME_H = 520;
const STROKE_WIDTH = 6;
const INACTIVITY_MS = 1500;

const PIXEL_SIZE = 20;
const BG_PALETTE = [
  "#0052ff", "#00c896", "#ffe135", "#ff6b9d",
  "#7fffdb", "#2d2d2d", "#008cff", "#b4ff64",
];

// Shape recognizer constants (mirrors Python)
const MIN_POINTS     = 30;
const CIRCLE_CV_THR  = 0.35;
const CLOSURE_RATIO  = 0.30;
const RECT_EPS_RATIO = 0.04;

// ── Pixel Art Background ─────────────────────────────────────────────────────
function drawPixelBg() {
  const canvas = document.getElementById("bg-canvas");
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx = canvas.getContext("2d");
  // Seeded pseudo-random (same seed = same pattern every load)
  let seed = 42;
  function rand() {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    return (seed >>> 0) / 0xffffffff;
  }
  for (let y = 0; y < canvas.height; y += PIXEL_SIZE) {
    for (let x = 0; x < canvas.width; x += PIXEL_SIZE) {
      const color = BG_PALETTE[Math.floor(rand() * BG_PALETTE.length)];
      ctx.fillStyle = color;
      ctx.fillRect(x, y, PIXEL_SIZE, PIXEL_SIZE);
    }
  }
}

// ── Canvas Manager (two-layer) ───────────────────────────────────────────────
class CanvasManager {
  constructor(displayCanvas) {
    this.displayCanvas = displayCanvas;
    this.dCtx = displayCanvas.getContext("2d");

    // Off-screen layers
    this.baseCanvas = new OffscreenCanvas(FRAME_W, FRAME_H);
    this.currentCanvas = new OffscreenCanvas(FRAME_W, FRAME_H);
    this.bCtx = this.baseCanvas.getContext("2d");
    this.cCtx = this.currentCanvas.getContext("2d");

    this._prev = null;
    this._strokePoints = [];
    this._strokeColor = [255, 255, 255];
  }

  drawPoint(x, y, color) {
    this._strokeColor = color;
    const [r, g, b] = color;
    const ctx = this.cCtx;
    const w = STROKE_WIDTH;

    if (this._prev) {
      const [px, py] = this._prev;
      const dx = x - px, dy = y - py;
      if (Math.hypot(dx, dy) > 80) { this._prev = [x, y]; return; }

      // Glow layers: wide faint → narrow solid
      _strokeLine(ctx, r, g, b, 0.14, px, py, x, y, w * 4);
      _strokeLine(ctx, r, g, b, 0.31, px, py, x, y, w * 2);
      _strokeLine(ctx, r, g, b, 1.00, px, py, x, y, w);
    } else {
      const ctx2 = this.cCtx;
      _dot(ctx2, r, g, b, 0.14, x, y, w * 2);
      _dot(ctx2, r, g, b, 0.31, x, y, w);
      _dot(ctx2, r, g, b, 1.00, x, y, w / 2);
    }

    this._strokePoints.push([x, y]);
    this._prev = [x, y];
  }

  liftPen() {
    this._prev = null;
  }

  finalizeStroke() {
    if (this._strokePoints.length > 0) {
      const shape = recognize(this._strokePoints);
      if (shape) {
        // Replace rough stroke with perfect shape
        this.cCtx.clearRect(0, 0, FRAME_W, FRAME_H);
        this._drawShape(shape, this._strokeColor);
      } else {
        // Commit freehand stroke to base as-is
        this.bCtx.drawImage(this.currentCanvas, 0, 0);
        this.cCtx.clearRect(0, 0, FRAME_W, FRAME_H);
      }
    } else {
      this.cCtx.clearRect(0, 0, FRAME_W, FRAME_H);
    }
    this._strokePoints = [];
    this._prev = null;
  }

  _drawShape(shape, color) {
    const [r, g, b] = color;
    const ctx = this.bCtx;
    const w = STROKE_WIDTH;

    if (shape.type === "circle") {
      const { cx, cy, radius } = shape;
      for (const [alpha, lw] of [[0.14, w*4],[0.31, w*2],[1.0, w]]) {
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth = lw;
        ctx.stroke();
      }
    } else if (shape.type === "rectangle") {
      const { x, y, rw, rh } = shape;
      for (const [alpha, lw] of [[0.14, w*4],[0.31, w*2],[1.0, w]]) {
        ctx.beginPath();
        ctx.rect(x, y, rw, rh);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth = lw;
        ctx.stroke();
      }
    }
  }

  clear() {
    this.bCtx.clearRect(0, 0, FRAME_W, FRAME_H);
    this.cCtx.clearRect(0, 0, FRAME_W, FRAME_H);
    this._strokePoints = [];
    this._prev = null;
  }

  // Composite both layers onto display canvas
  // Call once per frame AFTER skeleton/particles are drawn
  blit() {
    const dCtx = this.dCtx;
    dCtx.clearRect(0, 0, FRAME_W, FRAME_H);
    dCtx.drawImage(this.baseCanvas, 0, 0);
    dCtx.drawImage(this.currentCanvas, 0, 0);
  }
}

// Canvas drawing helpers
function _strokeLine(ctx, r, g, b, alpha, x1, y1, x2, y2, lw) {
  ctx.save();
  ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
  ctx.lineWidth = lw;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function _dot(ctx, r, g, b, alpha, x, y, radius) {
  ctx.save();
  ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
  ctx.beginPath();
  ctx.arc(x, y, Math.max(1, radius), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ── Particle System ───────────────────────────────────────────────────────────
const GRAVITY     = 120;   // px/s²
const SPAWN_COUNT = 3;

class Particle {
  constructor(x, y, color) {
    this.x = x; this.y = y;
    this.color = color;
    const angle = Math.random() * Math.PI * 2;
    const speed = 60 + Math.random() * 80;
    this.vx = Math.cos(angle) * speed;
    this.vy = Math.sin(angle) * speed - 40;
    this.life = 0.4 + Math.random() * 0.5;
    this.maxLife = this.life;
    this.radius = 2 + Math.random() * 3;
  }

  update(dt) {
    this.vy += GRAVITY * dt;
    this.vx *= (1 - 3 * dt);
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.life -= dt;
  }

  get alive() { return this.life > 0; }

  draw(ctx) {
    const alpha = Math.max(0, this.life / this.maxLife);
    const [r, g, b] = this.color;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

class ParticleSystem {
  constructor() { this.particles = []; }

  spawn(x, y, color) {
    for (let i = 0; i < SPAWN_COUNT; i++) {
      this.particles.push(new Particle(x, y, color));
    }
  }

  update(dt) {
    this.particles = this.particles.filter(p => { p.update(dt); return p.alive; });
  }

  draw(ctx) {
    for (const p of this.particles) p.draw(ctx);
  }
}

// ── Shape Recognizer (ported from Python) ────────────────────────────────────

// Andrew's monotone chain convex hull — returns indices of hull points
function convexHull(points) {
  const n = points.length;
  if (n < 3) return points;

  const sorted = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  function cross(O, A, B) {
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
  }

  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

// Ramer–Douglas–Peucker simplification
function rdp(points, epsilon) {
  if (points.length < 3) return points;
  let maxDist = 0, maxIdx = 0;
  const first = points[0], last = points[points.length - 1];
  const dx = last[0] - first[0], dy = last[1] - first[1];
  const len = Math.hypot(dx, dy);

  for (let i = 1; i < points.length - 1; i++) {
    const d = len === 0
      ? Math.hypot(points[i][0] - first[0], points[i][1] - first[1])
      : Math.abs(dy * points[i][0] - dx * points[i][1] + last[0]*first[1] - last[1]*first[0]) / len;
    if (d > maxDist) { maxDist = d; maxIdx = i; }
  }

  if (maxDist > epsilon) {
    const left  = rdp(points.slice(0, maxIdx + 1), epsilon);
    const right = rdp(points.slice(maxIdx), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [first, last];
}

function arcLength(pts) {
  let len = 0;
  for (let i = 1; i < pts.length; i++) {
    len += Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
  }
  return len;
}

function isClosed(pts) {
  const first = pts[0], last = pts[pts.length-1];
  const closureDist = Math.hypot(last[0]-first[0], last[1]-first[1]);
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const bboxDiag = Math.hypot(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
  return bboxDiag > 0 && (closureDist / bboxDiag) < CLOSURE_RATIO;
}

function rectAnglesOk(corners) {
  const n = corners.length;
  for (let i = 0; i < n; i++) {
    const prev   = corners[(i-1+n)%n];
    const vertex = corners[i];
    const next   = corners[(i+1)%n];
    const v1 = [prev[0]-vertex[0], prev[1]-vertex[1]];
    const v2 = [next[0]-vertex[0], next[1]-vertex[1]];
    const denom = Math.hypot(...v1) * Math.hypot(...v2);
    if (denom < 1e-6) return false;
    const cosA = Math.max(-1, Math.min(1, (v1[0]*v2[0]+v1[1]*v2[1]) / denom));
    const angle = Math.acos(cosA) * 180 / Math.PI;
    if (angle < 60 || angle > 120) return false;
  }
  return true;
}

function recognize(points) {
  if (points.length < MIN_POINTS) return null;
  if (!isClosed(points)) return null;

  const hull = convexHull(points);
  const perim = arcLength(hull.concat([hull[0]]));
  const epsilon = RECT_EPS_RATIO * perim;
  const approx = rdp(hull, epsilon);

  // Rectangle check first (4 vertices, roughly right angles)
  if (approx.length === 4 && rectAnglesOk(approx)) {
    const xs = approx.map(p=>p[0]), ys = approx.map(p=>p[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    const rw = Math.max(...xs) - x, rh = Math.max(...ys) - y;
    const aspect = rh > 0 ? rw/rh : 0;
    if (rw >= 30 && rh >= 30 && aspect >= 0.2 && aspect <= 5.0) {
      return { type: "rectangle", x, y, rw, rh };
    }
  }

  // Circle check
  const cx = points.reduce((s,p)=>s+p[0],0) / points.length;
  const cy = points.reduce((s,p)=>s+p[1],0) / points.length;
  const dists = points.map(p=>Math.hypot(p[0]-cx, p[1]-cy));
  const meanR = dists.reduce((s,d)=>s+d,0) / dists.length;
  if (meanR < 20) return null;
  const variance = dists.reduce((s,d)=>s+(d-meanR)**2,0) / dists.length;
  const cv = Math.sqrt(variance) / meanR;
  if (cv < CIRCLE_CV_THR) {
    return { type: "circle", cx: Math.round(cx), cy: Math.round(cy), radius: Math.round(meanR) };
  }

  return null;
}

// ── Rainbow color ─────────────────────────────────────────────────────────────
function getRainbowColor() {
  const hue = ((Date.now() * 0.00015) % 1) * 360;
  // HSL to RGB
  const s = 1, l = 0.5;
  const c = (1 - Math.abs(2*l - 1)) * s;
  const x = c * (1 - Math.abs((hue/60)%2 - 1));
  const m = l - c/2;
  let r,g,b;
  if (hue < 60)       [r,g,b]=[c,x,0];
  else if (hue < 120) [r,g,b]=[x,c,0];
  else if (hue < 180) [r,g,b]=[0,c,x];
  else if (hue < 240) [r,g,b]=[0,x,c];
  else if (hue < 300) [r,g,b]=[x,0,c];
  else                [r,g,b]=[c,0,x];
  return [Math.round((r+m)*255), Math.round((g+m)*255), Math.round((b+m)*255)];
}

// ── Hand skeleton drawing ─────────────────────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

function drawSkeleton(ctx, landmarks, drawing) {
  const lineColor = drawing ? "rgba(255,160,40,0.9)" : "rgba(255,255,255,0.9)";
  ctx.save();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a][0], landmarks[a][1]);
    ctx.lineTo(landmarks[b][0], landmarks[b][1]);
    ctx.stroke();
  }
  for (let i = 0; i < landmarks.length; i++) {
    const [x, y] = landmarks[i];
    const isIndexTip = i === 8;
    let color, radius;
    if (isIndexTip && drawing) {
      color = "rgb(255,80,0)"; radius = 9;
    } else if (isIndexTip) {
      color = "rgb(0,229,255)"; radius = 7;
    } else {
      color = "rgb(255,255,255)"; radius = 5;
    }
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI*2);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.restore();
}

// ── Button icon drawing ───────────────────────────────────────────────────────
function updateButtonIcon(btn, active) {
  btn.innerHTML = "";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", "20"); svg.setAttribute("height", "20");
  svg.setAttribute("viewBox", "0 0 20 20");
  if (active) {
    // Stop square
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x","5"); rect.setAttribute("y","5");
    rect.setAttribute("width","10"); rect.setAttribute("height","10");
    rect.setAttribute("rx","2"); rect.setAttribute("fill","white");
    svg.appendChild(rect);
  } else {
    // Play circle (record dot)
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx","10"); circle.setAttribute("cy","10");
    circle.setAttribute("r","6"); circle.setAttribute("fill","#ff3b3b");
    svg.appendChild(circle);
  }
  btn.appendChild(svg);
}

// ── Main App ──────────────────────────────────────────────────────────────────
class App {
  constructor() {
    this.video      = document.getElementById("video");
    this.dispCanvas = document.getElementById("display-canvas");
    this.toggleBtn  = document.getElementById("toggle-btn");
    this.pill       = document.getElementById("pill");

    this.dispCanvas.width  = FRAME_W;
    this.dispCanvas.height = FRAME_H;
    this.ctx = this.dispCanvas.getContext("2d");

    this.canvasMgr  = new CanvasManager(this.dispCanvas);
    this.particles  = new ParticleSystem();

    this.tracking    = false;
    this.kHeld       = false;
    this.landmarks   = null;
    this.fingerPos   = null;
    this.lastDrawT   = 0;
    this.handLandmarker = null;

    this._prevTime  = null;
    this._animId    = null;

    this._setupUI();
    this._setupKeys();
    this._initMediaPipe().then(() => {
      this._startCamera();
      this._loop(performance.now());
    });
  }

  _setupUI() {
    updateButtonIcon(this.toggleBtn, false);
    this._updatePill();

    this.toggleBtn.addEventListener("click", () => {
      this.tracking = !this.tracking;
      this.toggleBtn.classList.toggle("active", this.tracking);
      updateButtonIcon(this.toggleBtn, this.tracking);
      if (!this.tracking) {
        this.canvasMgr.finalizeStroke();
        this.landmarks  = null;
        this.fingerPos  = null;
        this.kHeld      = false;
        this.lastDrawT  = 0;
      }
      this._updatePill();
    });
  }

  _setupKeys() {
    window.addEventListener("keydown", e => {
      if (e.key === "k" || e.key === "K") {
        if (!this.kHeld) {
          this.kHeld = true;
          this._updatePill();
        }
      } else if (e.key === "Delete" || e.key === "Backspace") {
        this.canvasMgr.clear();
        this.lastDrawT = 0;
      }
    });
    window.addEventListener("keyup", e => {
      if (e.key === "k" || e.key === "K") {
        this.kHeld = false;
        this.canvasMgr.liftPen();
        this._updatePill();
      }
    });
  }

  _updatePill() {
    if (!this.tracking) {
      this.pill.textContent = "Press ● to start tracking";
    } else if (this.kHeld) {
      this.pill.textContent = "✏  Drawing...   •   DELETE to clear";
    } else {
      this.pill.textContent = "Hold K to draw   •   DELETE to clear";
    }
  }

  async _initMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
    );
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });
  }

  _startCamera() {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then(stream => { this.video.srcObject = stream; })
      .catch(err => console.error("Camera error:", err));
  }

  _processHand() {
    if (!this.handLandmarker || this.video.readyState < 2) return;
    const now = performance.now();
    const result = this.handLandmarker.detectForVideo(this.video, now);
    if (!result.landmarks || result.landmarks.length === 0) {
      this.landmarks  = null;
      this.fingerPos  = null;
      return;
    }
    // Mirror: video is CSS-mirrored (scaleX(-1)), landmarks come from original
    // We mirror them here so they match the visual
    this.landmarks = result.landmarks[0].map(lm => [
      FRAME_W - lm.x * FRAME_W,
      lm.y * FRAME_H,
    ]);
    this.fingerPos = this.landmarks[8]; // INDEX_TIP
  }

  _loop(now) {
    this._animId = requestAnimationFrame(t => this._loop(t));

    const dt = this._prevTime ? Math.min((now - this._prevTime) / 1000, 0.1) : 0.016;
    this._prevTime = now;

    // ── Update ───────────────────────────────────────────────────────────────
    this.particles.update(dt);

    if (this.tracking) {
      this._processHand();

      if (this.fingerPos && this.kHeld) {
        const color = getRainbowColor();
        this.canvasMgr.drawPoint(...this.fingerPos, color);
        this.particles.spawn(...this.fingerPos, color);
        this.lastDrawT = Date.now();
      }
    } else {
      this.landmarks = null;
      this.fingerPos = null;
    }

    // Inactivity timer — finalize stroke after 1.5s
    if (this.lastDrawT && (Date.now() - this.lastDrawT) > INACTIVITY_MS) {
      this.canvasMgr.finalizeStroke();
      this.lastDrawT = 0;
    }

    // ── Render ───────────────────────────────────────────────────────────────
    const ctx = this.ctx;
    ctx.clearRect(0, 0, FRAME_W, FRAME_H);

    // Draw strokes (base + current)
    ctx.drawImage(this.canvasMgr.baseCanvas, 0, 0);
    ctx.drawImage(this.canvasMgr.currentCanvas, 0, 0);

    // Hand skeleton
    if (this.tracking && this.landmarks) {
      drawSkeleton(ctx, this.landmarks, this.kHeld);
    }

    // Particles (on top of skeleton)
    this.particles.draw(ctx);
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
drawPixelBg();
window.addEventListener("resize", drawPixelBg);
new App();
