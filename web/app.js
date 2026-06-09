import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm";

// ── Constants ────────────────────────────────────────────────────────────────
const FRAME_W = 860;
const FRAME_H = 520;
const GAME_GRAVITY = 600; // px/s² for fruit physics
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

  getStrokeBBox() {
    if (!this._strokePoints.length) return null;
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const [x, y] of this._strokePoints) {
      if (x < x1) x1 = x; if (x > x2) x2 = x;
      if (y < y1) y1 = y; if (y > y2) y2 = y;
    }
    return {
      x1: Math.max(0, x1 - 24),
      y1: Math.max(0, y1 - 24),
      x2: Math.min(FRAME_W, x2 + 24),
      y2: Math.min(FRAME_H, y2 + 24),
    };
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

// ── Math Recognition ──────────────────────────────────────────────────────────

function mergeBBox(a, b) {
  return {
    x1: Math.min(a.x1, b.x1),
    y1: Math.min(a.y1, b.y1),
    x2: Math.max(a.x2, b.x2),
    y2: Math.max(a.y2, b.y2),
  };
}

function _roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

async function extractMathImage(baseCanvas, bbox) {
  const scale = 3;
  const w = Math.max(1, Math.round((bbox.x2 - bbox.x1) * scale));
  const h = Math.max(1, Math.round((bbox.y2 - bbox.y1) * scale));
  const tmp = new OffscreenCanvas(w, h);
  const ctx = tmp.getContext('2d');

  // White background
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  // Draw strokes onto white background
  ctx.save();
  ctx.scale(scale, scale);
  ctx.translate(-bbox.x1, -bbox.y1);
  ctx.drawImage(baseCanvas, 0, 0);
  ctx.restore();

  // Threshold: transparent / near-white pixels → white, anything else → black
  const imgData = ctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const bright = d[i] > 200 && d[i+1] > 200 && d[i+2] > 200;
    const transparent = d[i+3] < 30;
    if (bright || transparent) {
      d[i]=255; d[i+1]=255; d[i+2]=255; d[i+3]=255;
    } else {
      d[i]=0; d[i+1]=0; d[i+2]=0; d[i+3]=255;
    }
  }
  ctx.putImageData(imgData, 0, 0);

  return tmp.convertToBlob({ type: 'image/png' });
}

function safeMathEval(expr) {
  const clean = expr.replace(/[=\s]+$/, '').trim();
  if (!/^[\d\s\+\-\*\/\.\(\)]+$/.test(clean)) return null;
  try {
    // eslint-disable-next-line no-new-func
    const result = Function(`"use strict"; return (${clean})`)();
    if (typeof result !== 'number' || !isFinite(result)) return null;
    return Number(result.toPrecision(10)).toString();
  } catch { return null; }
}

// ── Fruit Ninja ───────────────────────────────────────────────────────────────

const FRUIT_TYPES = ['watermelon', 'orange', 'apple', 'lemon'];
const FRUIT_COLORS = {
  watermelon: { body: '#3cb34a', inner: '#e8203c', juice: [220, 40, 40] },
  orange:     { body: '#ff8c00', inner: '#ffb84d', juice: [255, 160, 0] },
  apple:      { body: '#e8003c', inner: '#ff6680', juice: [255, 80, 80] },
  lemon:      { body: '#ffe135', inner: '#fff176', juice: [220, 200, 0] },
  bomb:       { body: '#1a1a1a', inner: '#333',    juice: [80,  80,  80] },
};

function _drawHeart(ctx, x, y, size, filled) {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  ctx.moveTo(0, size * 0.3);
  ctx.bezierCurveTo(-size, -size * 0.3, -size * 1.2, size * 0.8, 0, size * 1.2);
  ctx.bezierCurveTo(size * 1.2, size * 0.8, size, -size * 0.3, 0, size * 0.3);
  ctx.closePath();
  if (filled) {
    ctx.fillStyle = '#ff3b3b';
    ctx.fill();
  } else {
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

class JuiceParticle {
  constructor(x, y, color) {
    this.x = x; this.y = y;
    this.color = color;
    const angle = Math.random() * Math.PI * 2;
    const speed = 120 + Math.random() * 180;
    this.vx = Math.cos(angle) * speed;
    this.vy = Math.sin(angle) * speed - 60;
    this.life = 0.5 + Math.random() * 0.4;
    this.maxLife = this.life;
    this.radius = 4 + Math.random() * 5;
  }
  update(dt) {
    this.vy += GAME_GRAVITY * 0.4 * dt;
    this.vx *= (1 - 2 * dt);
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

class Fruit {
  constructor(x, y, vx, vy, type) {
    this.x = x; this.y = y;
    this.vx = vx; this.vy = vy;
    this.type = type;
    this.radius = type === 'bomb' ? 38 : 40 + Math.random() * 12;
    this.rotation = Math.random() * Math.PI * 2;
    this.rotationSpeed = (Math.random() - 0.5) * 5;
    this.sliced = false;
  }
  update(dt) {
    this.vy += GAME_GRAVITY * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.rotation += this.rotationSpeed * dt;
  }
  get offScreen() {
    return this.y > FRAME_H + this.radius + 10;
  }
  draw(ctx) {
    if (this.sliced) return;
    const { body, inner } = FRUIT_COLORS[this.type];
    const r = this.radius;
    ctx.save();
    ctx.translate(this.x, this.y);
    ctx.rotate(this.rotation);

    if (this.type === 'watermelon') {
      // Green outer
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      // Red inner
      ctx.beginPath(); ctx.arc(0, 0, r * 0.78, 0, Math.PI * 2);
      ctx.fillStyle = inner; ctx.fill();
      // Seeds
      ctx.fillStyle = '#1a1a1a';
      for (let i = 0; i < 5; i++) {
        const a = (i / 5) * Math.PI * 2;
        ctx.save();
        ctx.translate(Math.cos(a) * r * 0.38, Math.sin(a) * r * 0.38);
        ctx.rotate(a);
        ctx.beginPath();
        ctx.ellipse(0, 0, r * 0.07, r * 0.13, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
      // Stripe lines on green
      ctx.strokeStyle = 'rgba(0,100,0,0.5)';
      ctx.lineWidth = 2;
      for (let i = 0; i < 4; i++) {
        const a = (i / 4) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(Math.cos(a) * r * 0.78, Math.sin(a) * r * 0.78);
        ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
        ctx.stroke();
      }

    } else if (this.type === 'orange') {
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      ctx.beginPath(); ctx.arc(0, 0, r * 0.72, 0, Math.PI * 2);
      ctx.fillStyle = inner; ctx.fill();
      // Segments
      ctx.strokeStyle = 'rgba(200,100,0,0.35)';
      ctx.lineWidth = 1.5;
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(a) * r * 0.72, Math.sin(a) * r * 0.72);
        ctx.stroke();
      }
      // Highlight
      ctx.beginPath(); ctx.arc(-r * 0.25, -r * 0.25, r * 0.18, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.25)'; ctx.fill();

    } else if (this.type === 'apple') {
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      // Indent top
      ctx.beginPath(); ctx.arc(0, -r * 0.85, r * 0.18, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fill();
      // Stem
      ctx.strokeStyle = '#5c3d1e'; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(0, -r * 0.9);
      ctx.quadraticCurveTo(r * 0.25, -r * 1.25, r * 0.1, -r * 1.4);
      ctx.stroke();
      // Highlight
      ctx.beginPath(); ctx.arc(-r * 0.3, -r * 0.3, r * 0.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.28)'; ctx.fill();

    } else if (this.type === 'lemon') {
      // Lemon: slightly oval
      ctx.save();
      ctx.scale(1.2, 0.85);
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      ctx.restore();
      // Tip bumps
      ctx.beginPath(); ctx.arc(r * 1.05, 0, r * 0.2, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      ctx.beginPath(); ctx.arc(-r * 1.05, 0, r * 0.2, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      // Highlight
      ctx.beginPath(); ctx.arc(-r * 0.2, -r * 0.3, r * 0.22, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.fill();

    } else { // bomb
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fillStyle = body; ctx.fill();
      // Shine
      ctx.beginPath(); ctx.arc(-r * 0.3, -r * 0.3, r * 0.18, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.2)'; ctx.fill();
      // Skull
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.beginPath(); ctx.arc(0, -r * 0.08, r * 0.35, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = body;
      // Eye sockets
      ctx.beginPath(); ctx.arc(-r * 0.13, -r * 0.13, r * 0.1, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(r * 0.13, -r * 0.13, r * 0.1, 0, Math.PI * 2); ctx.fill();
      // Teeth
      ctx.fillStyle = body;
      ctx.fillRect(-r * 0.22, r * 0.06, r * 0.14, r * 0.14);
      ctx.fillRect(-r * 0.04, r * 0.06, r * 0.14, r * 0.14);
      ctx.fillRect(r * 0.12, r * 0.06, r * 0.12, r * 0.14);
      // Fuse
      ctx.strokeStyle = '#8B4513'; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(0, -r);
      ctx.quadraticCurveTo(r * 0.4, -r * 1.4, r * 0.2, -r * 1.7);
      ctx.stroke();
      // Fuse spark
      ctx.fillStyle = '#FFA500';
      ctx.beginPath(); ctx.arc(r * 0.2, -r * 1.7, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#FFD700';
      ctx.beginPath(); ctx.arc(r * 0.2, -r * 1.7, 2.5, 0, Math.PI * 2); ctx.fill();
    }

    ctx.restore();
  }
}

class FruitHalf {
  constructor(fruit, arcStart, vx, vy) {
    this.x = fruit.x; this.y = fruit.y;
    this.vx = vx; this.vy = vy;
    this.type = fruit.type;
    this.radius = fruit.radius;
    this.arcStart = arcStart;
    this.rotation = fruit.rotation;
    this.rotationSpeed = (Math.random() - 0.5) * 6;
    this.life = 0.8 + Math.random() * 0.4;
    this.maxLife = this.life;
  }
  update(dt) {
    this.vy += GAME_GRAVITY * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.rotation += this.rotationSpeed * dt;
    this.life -= dt;
  }
  get alive() { return this.life > 0 && this.y < FRAME_H + this.radius + 20; }
  draw(ctx) {
    const alpha = Math.max(0, this.life / this.maxLife);
    const { body, inner } = FRUIT_COLORS[this.type];
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(this.x, this.y);
    ctx.rotate(this.rotation);
    // Clip to semicircle
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, this.radius, this.arcStart, this.arcStart + Math.PI);
    ctx.closePath();
    ctx.clip();
    // Outer color
    ctx.fillStyle = body;
    ctx.beginPath(); ctx.arc(0, 0, this.radius, 0, Math.PI * 2); ctx.fill();
    // Inner flesh
    ctx.fillStyle = inner;
    ctx.beginPath(); ctx.arc(0, 0, this.radius * 0.72, 0, Math.PI * 2); ctx.fill();
    // Cut face white sheen
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.fillRect(-this.radius, -2, this.radius * 2, 4);
    ctx.restore();
  }
}

class FruitNinjaGame {
  constructor() {
    this.reset();
  }

  reset() {
    this.score       = 0;
    this.lives       = 3;
    this.combo       = 0;
    this.comboTimer  = 0;
    this.fruits      = [];
    this.halves      = [];
    this.juiceParticles = [];
    this.fingerTrail = [];
    this.gameState   = 'playing';
    this.spawnTimer  = 1.2;
    this.sliceCount  = 0;
    this.flashTimer  = 0; // red flash on bomb
  }

  update(dt, fingerPos) {
    if (this.gameState === 'gameover') return;

    // Combo timer
    if (this.comboTimer > 0) {
      this.comboTimer -= dt;
      if (this.comboTimer <= 0) this.combo = 0;
    }

    // Flash timer
    if (this.flashTimer > 0) this.flashTimer -= dt;

    // Finger trail (keep last 120ms)
    if (fingerPos) {
      const now = performance.now() / 1000;
      this.fingerTrail.push({ x: fingerPos[0], y: fingerPos[1], t: now });
      this.fingerTrail = this.fingerTrail.filter(s => now - s.t < 0.20);
    } else {
      this.fingerTrail = [];
    }

    // Spawn
    this.spawnTimer -= dt;
    if (this.spawnTimer <= 0) {
      this._spawnFruit();
      const baseInterval = Math.max(0.6, 1.8 - Math.floor(this.sliceCount / 5) * 0.05);
      this.spawnTimer = baseInterval * (0.8 + Math.random() * 0.4);
      // Occasionally spawn a second fruit shortly after
      if (Math.random() < 0.22) this.spawnTimer = Math.min(this.spawnTimer, 0.18);
    }

    // Update fruits
    for (const f of this.fruits) f.update(dt);

    // Slice detection
    this._checkSlices();

    // Remove sliced + off-screen fruits, penalize misses
    this.fruits = this.fruits.filter(f => {
      if (f.sliced) return false;
      if (f.offScreen) {
        if (f.type !== 'bomb') {
          this.lives--;
          this.combo = 0;
          this.comboTimer = 0;
          if (this.lives <= 0) this.gameState = 'gameover';
        }
        return false;
      }
      return true;
    });

    // Update halves + juice
    for (const h of this.halves)   h.update(dt);
    for (const p of this.juiceParticles) p.update(dt);
    this.halves          = this.halves.filter(h => h.alive);
    this.juiceParticles  = this.juiceParticles.filter(p => p.alive);
  }

  _spawnFruit() {
    const isBomb = Math.random() < (this.score > 20 ? 0.22 : 0.15);
    const type = isBomb ? 'bomb' : FRUIT_TYPES[Math.floor(Math.random() * FRUIT_TYPES.length)];
    const radius = type === 'bomb' ? 28 : 28 + Math.random() * 10;
    const x = FRAME_W * (0.15 + Math.random() * 0.70);
    const y = FRAME_H + radius;
    const minRise = FRAME_H * (0.55 + Math.random() * 0.30);
    const vy = -Math.sqrt(2 * GAME_GRAVITY * minRise);
    const totalFlight = 2 * Math.abs(vy) / GAME_GRAVITY;
    const maxVx = (FRAME_W * 0.4) / totalFlight;
    const vx = (Math.random() - 0.5) * 2 * maxVx;
    this.fruits.push(new Fruit(x, y, vx, vy, type));
  }

  _checkSlices() {
    if (this.fingerTrail.length < 2) return;

    // Test every consecutive segment in the trail — catches fast swipes
    // that skip over a fruit between frames
    for (let i = 1; i < this.fingerTrail.length; i++) {
      const prev = this.fingerTrail[i - 1];
      const curr = this.fingerTrail[i];
      const tDiff = curr.t - prev.t;
      if (tDiff < 0.001) continue;

      const sdx = curr.x - prev.x;
      const sdy = curr.y - prev.y;
      const segSpeed = Math.hypot(sdx, sdy) / tDiff;
      if (segSpeed < 250) continue; // per-segment velocity gate

      const abLen2 = sdx * sdx + sdy * sdy;
      if (abLen2 < 1) continue;

      for (const fruit of this.fruits) {
        if (fruit.sliced) continue;
        const acx = fruit.x - prev.x;
        const acy = fruit.y - prev.y;
        const t = Math.max(0, Math.min(1, (acx * sdx + acy * sdy) / abLen2));
        const closestX = prev.x + t * sdx;
        const closestY = prev.y + t * sdy;
        const dist = Math.hypot(fruit.x - closestX, fruit.y - closestY);

        if (dist <= fruit.radius + 15) { // +15px forgiveness buffer
          this._onSlice(fruit, Math.atan2(sdy, sdx));
        }
      }
    }
  }

  _onSlice(fruit, cutAngle) {
    fruit.sliced = true;

    if (fruit.type === 'bomb') {
      this.lives--;
      this.combo = 0;
      this.comboTimer = 0;
      this.flashTimer = 0.35;
      if (this.lives <= 0) this.gameState = 'gameover';
    } else {
      this.combo++;
      this.comboTimer = 2.0;
      const points = this.combo;
      this.score += points;
      this.sliceCount++;
    }

    // Two halves
    const perpAngle = cutAngle - Math.PI / 2;
    const halfSpeed = 80 + Math.random() * 60;
    const h1 = new FruitHalf(fruit, cutAngle, fruit.vx + Math.cos(perpAngle) * halfSpeed, fruit.vy + Math.sin(perpAngle) * halfSpeed);
    const h2 = new FruitHalf(fruit, cutAngle + Math.PI, fruit.vx - Math.cos(perpAngle) * halfSpeed, fruit.vy - Math.sin(perpAngle) * halfSpeed);
    this.halves.push(h1, h2);

    // Juice splash
    const count = 14 + Math.floor(Math.random() * 7);
    const juiceColor = FRUIT_COLORS[fruit.type].juice;
    for (let i = 0; i < count; i++) {
      this.juiceParticles.push(new JuiceParticle(fruit.x, fruit.y, juiceColor));
    }
  }

  draw(ctx) {
    // Background: semi-dark overlay so fruits are readable over webcam feed
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.fillRect(0, 0, FRAME_W, FRAME_H);

    // Red flash on bomb
    if (this.flashTimer > 0) {
      ctx.fillStyle = `rgba(255,0,0,${Math.min(0.45, this.flashTimer * 1.5)})`;
      ctx.fillRect(0, 0, FRAME_W, FRAME_H);
    }

    // Juice particles
    for (const p of this.juiceParticles) p.draw(ctx);

    // Fruit halves
    for (const h of this.halves) h.draw(ctx);

    // Active fruits
    for (const f of this.fruits) f.draw(ctx);

    // Swipe trail
    if (this.fingerTrail.length >= 2) {
      ctx.save();
      for (let i = 1; i < this.fingerTrail.length; i++) {
        const prev = this.fingerTrail[i - 1];
        const curr = this.fingerTrail[i];
        const alpha = (i / this.fingerTrail.length) * 0.7;
        ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
        ctx.lineWidth = 3 * (i / this.fingerTrail.length);
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // HUD
    this._drawHUD(ctx);

    // Game over overlay
    if (this.gameState === 'gameover') {
      ctx.fillStyle = 'rgba(0,0,0,0.72)';
      ctx.fillRect(0, 0, FRAME_W, FRAME_H);

      ctx.save();
      ctx.textAlign = 'center';
      ctx.shadowColor = 'rgba(0,0,0,0.9)';
      ctx.shadowBlur = 12;

      ctx.font = 'bold 72px "SF Pro Display", Arial';
      ctx.fillStyle = '#ff3b3b';
      ctx.fillText('GAME OVER', FRAME_W / 2, FRAME_H / 2 - 50);

      ctx.font = 'bold 36px "SF Pro Display", Arial';
      ctx.fillStyle = '#fff';
      ctx.fillText(`Score: ${this.score}`, FRAME_W / 2, FRAME_H / 2 + 10);

      ctx.font = '22px "SF Pro Display", Arial';
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.fillText('Press SPACE or click to play again', FRAME_W / 2, FRAME_H / 2 + 60);
      ctx.restore();
    }
  }

  _drawHUD(ctx) {
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 8;

    // Score
    ctx.font = 'bold 36px "SF Pro Display", Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText(`${this.score}`, 20, 52);

    // Combo
    if (this.combo >= 2 && this.comboTimer > 0) {
      ctx.font = 'bold 20px "SF Pro Display", Arial';
      ctx.fillStyle = '#FFD700';
      ctx.fillText(`×${this.combo} COMBO!`, 20, 78);
    }

    // Hearts (top-right)
    for (let i = 0; i < 3; i++) {
      const hx = FRAME_W - 28 - i * 38;
      const hy = 26;
      _drawHeart(ctx, hx, hy, 13, i < this.lives);
    }

    ctx.restore();
  }
}

// ── Light Up Mode ────────────────────────────────────────────────────────────

const LIGHTUP_BRUSH_RADIUS = 80;
const LIGHTUP_FADE_SPEED   = 0.15; // alpha per second — ~6-7s full fade

class LightUpMode {
  constructor() {
    this.image = null;
    this.maskCanvas = new OffscreenCanvas(FRAME_W, FRAME_H);
    this.maskCtx = this.maskCanvas.getContext('2d');
    // Start fully dark
    this.maskCtx.fillStyle = '#000';
    this.maskCtx.fillRect(0, 0, FRAME_W, FRAME_H);
    this._imageDrawParams = null; // {x, y, w, h} for contain-fit
  }

  setImage(img) {
    this.image = img;
    // Compute contain-fit dimensions
    const scale = Math.min(FRAME_W / img.width, FRAME_H / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    const x = (FRAME_W - w) / 2;
    const y = (FRAME_H - h) / 2;
    this._imageDrawParams = { x, y, w, h };
    // Reset mask to fully dark
    this.maskCtx.globalCompositeOperation = 'source-over';
    this.maskCtx.fillStyle = '#000';
    this.maskCtx.fillRect(0, 0, FRAME_W, FRAME_H);
  }

  update(dt, fingerPos) {
    const ctx = this.maskCtx;

    // Fade back: draw semi-transparent black over the mask to restore darkness
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = `rgba(0,0,0,${Math.min(1, LIGHTUP_FADE_SPEED * dt)})`;
    ctx.fillRect(0, 0, FRAME_W, FRAME_H);

    // Punch a soft hole at the fingertip
    if (fingerPos) {
      const [fx, fy] = fingerPos;
      ctx.globalCompositeOperation = 'destination-out';
      const grad = ctx.createRadialGradient(fx, fy, 0, fx, fy, LIGHTUP_BRUSH_RADIUS);
      grad.addColorStop(0, 'rgba(0,0,0,1)');
      grad.addColorStop(0.6, 'rgba(0,0,0,0.7)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(fx, fy, LIGHTUP_BRUSH_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }

    // Reset composite mode
    ctx.globalCompositeOperation = 'source-over';
  }

  draw(ctx) {
    if (!this.image || !this._imageDrawParams) return;
    const { x, y, w, h } = this._imageDrawParams;

    // Draw the image
    ctx.drawImage(this.image, x, y, w, h);

    // Draw the darkness mask on top
    ctx.drawImage(this.maskCanvas, 0, 0);
  }
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
    this.drawBtn    = document.getElementById("draw-btn");
    this.pill       = document.getElementById("pill");
    this.isTouch    = false;

    this.dispCanvas.width  = FRAME_W;
    this.dispCanvas.height = FRAME_H;
    this.ctx = this.dispCanvas.getContext("2d");

    this.canvasMgr  = new CanvasManager(this.dispCanvas);
    this.particles  = new ParticleSystem();
    this.gameMode   = 'drawing'; // 'drawing' | 'fruity' | 'lightup'
    this.fruitGame  = new FruitNinjaGame();
    this.lightUp    = new LightUpMode();
    this.imageUpload = document.getElementById('image-upload');
    this.modeBtn    = document.getElementById('mode-btn');

    this.tracking    = false;
    this.kHeld       = false;
    this.landmarks   = null;
    this.fingerPos   = null;
    this.lastDrawT   = 0;
    this.handLandmarker = null;

    this.mathBBox    = null;   // accumulated bbox of strokes for current expression
    this.mathResult  = null;   // {text, x, y, timer, maxTimer} for result bubble
    this.tesseract   = null;   // lazy-loaded Tesseract worker

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

    // Detect touch device — show draw button, update pill text
    const markTouch = () => {
      if (this.isTouch) return;
      this.isTouch = true;
      document.body.classList.add('touch');
      this._updatePill();
    };
    window.addEventListener('touchstart', markTouch, { once: true, passive: true });
    // Also detect on first pointer of touch type
    window.addEventListener('pointerdown', e => {
      if (e.pointerType === 'touch') markTouch();
    }, { once: true });

    // Draw button — hold to draw (touch alternative to K key)
    this.drawBtn.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (!this.kHeld) { this.kHeld = true; this.drawBtn.classList.add('pressed'); this._updatePill(); }
    });
    const stopDraw = () => {
      if (this.kHeld) {
        this.kHeld = false;
        this.drawBtn.classList.remove('pressed');
        this.canvasMgr.liftPen();
        this._updatePill();
      }
    };
    this.drawBtn.addEventListener('pointerup',     stopDraw);
    this.drawBtn.addEventListener('pointercancel', stopDraw);
    this.drawBtn.addEventListener('pointerleave',  stopDraw);

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

    this.modeBtn.addEventListener("click", () => {
      if (this.gameMode === 'drawing') {
        this.gameMode = 'fruity';
        this.fruitGame.reset();
        this.canvasMgr.liftPen();
        this.modeBtn.textContent = '🔦 Light Up';
        this.modeBtn.classList.remove('lightup-active');
        this.modeBtn.classList.add('fruity-active');
        this.pill.textContent = 'Slice the fruits!  •  Press ●';
      } else if (this.gameMode === 'fruity') {
        // Trigger file upload — only switch mode once an image is loaded
        this.imageUpload.click();
      } else {
        this.gameMode = 'drawing';
        this.modeBtn.textContent = '🍉 Fruit Ninja';
        this.modeBtn.classList.remove('fruity-active', 'lightup-active');
        this._updatePill();
      }
    });

    // Handle image upload for Light Up mode
    this.imageUpload.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = new Image();
      img.onload = () => {
        this.lightUp.setImage(img);
        this.gameMode = 'lightup';
        this.canvasMgr.liftPen();
        this.modeBtn.textContent = '✏ Drawing';
        this.modeBtn.classList.remove('fruity-active');
        this.modeBtn.classList.add('lightup-active');
        this.pill.textContent = 'Move your hand to reveal the image';
        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(file);
      // Reset input so the same file can be re-selected
      this.imageUpload.value = '';
    });

    this.dispCanvas.addEventListener("pointerdown", () => {
      if (this.gameMode === 'fruity' && this.fruitGame.gameState === 'gameover') {
        this.fruitGame.reset();
      }
    });
  }

  _setupKeys() {
    window.addEventListener("keydown", e => {
      if (e.key === " " && this.gameMode === 'fruity' && this.fruitGame.gameState === 'gameover') {
        this.fruitGame.reset();
        return;
      }
      if (e.key === "k" || e.key === "K") {
        if (!this.kHeld) {
          this.kHeld = true;
          this._updatePill();
        }
      } else if (e.key === "Delete" || e.key === "Backspace") {
        this.canvasMgr.clear();
        this.lastDrawT  = 0;
        this.mathBBox   = null;
        this.mathResult = null;
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
      this.pill.textContent = "✏  Drawing...";
    } else if (this.isTouch) {
      this.pill.textContent = "Hold ✏ to draw";
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
    if (this.tracking) {
      this._processHand();
    } else {
      this.landmarks = null;
      this.fingerPos = null;
    }

    if (this.gameMode === 'fruity') {
      this.fruitGame.update(dt, this.fingerPos);
    } else if (this.gameMode === 'lightup') {
      this.lightUp.update(dt, this.fingerPos);
    } else {
      this.particles.update(dt);

      if (this.tracking && this.fingerPos && this.kHeld) {
        const color = getRainbowColor();
        this.canvasMgr.drawPoint(...this.fingerPos, color);
        this.particles.spawn(...this.fingerPos, color);
        this.lastDrawT = Date.now();
      }

      // Inactivity timer — finalize stroke after 1.5s
      if (this.lastDrawT && (Date.now() - this.lastDrawT) > INACTIVITY_MS) {
        const strokeBBox = this.canvasMgr.getStrokeBBox(); // capture before clear
        this.canvasMgr.finalizeStroke();
        this.lastDrawT = 0;
        if (strokeBBox) {
          this.mathBBox = this.mathBBox ? mergeBBox(this.mathBBox, strokeBBox) : strokeBBox;
          this._attemptMathOCR();
        }
      }

      // Decay math result bubble
      if (this.mathResult) {
        this.mathResult.timer -= dt;
        if (this.mathResult.timer <= 0) this.mathResult = null;
      }
    }

    // ── Render ───────────────────────────────────────────────────────────────
    const ctx = this.ctx;
    ctx.clearRect(0, 0, FRAME_W, FRAME_H);

    if (this.gameMode === 'fruity') {
      this.fruitGame.draw(ctx);
      if (this.tracking && this.landmarks) {
        drawSkeleton(ctx, this.landmarks, false);
      }
    } else if (this.gameMode === 'lightup') {
      this.lightUp.draw(ctx);
      if (this.tracking && this.landmarks) {
        drawSkeleton(ctx, this.landmarks, false);
      }
    } else {
      // Draw strokes (base + current)
      ctx.drawImage(this.canvasMgr.baseCanvas, 0, 0);
      ctx.drawImage(this.canvasMgr.currentCanvas, 0, 0);

      // Hand skeleton
      if (this.tracking && this.landmarks) {
        drawSkeleton(ctx, this.landmarks, this.kHeld);
      }

      // Particles (on top of skeleton)
      this.particles.draw(ctx);

      // Math result bubble
      this._drawMathResult(ctx);
    }
  }

  _drawMathResult(ctx) {
    if (!this.mathResult) return;
    const { text, x, y, timer, maxTimer } = this.mathResult;
    // Fade in during first 0.3s, fade out during last 25%
    const alpha = Math.min(1, timer / 0.3) * Math.min(1, (timer / maxTimer) * 4);
    if (alpha <= 0) return;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = 'bold 36px "SF Pro Display", Arial';
    ctx.textBaseline = 'middle';

    const tw  = ctx.measureText(text).width;
    const pad = 16;
    const rw  = tw + pad * 2;
    const rh  = 54;
    // Clamp so bubble stays inside canvas
    const bx  = Math.min(x, FRAME_W - rw - 8);
    const by  = Math.max(rh / 2 + 8, Math.min(y - rh / 2, FRAME_H - rh - 8));

    // Shadow glow
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur  = 12;

    // Gold pill
    ctx.fillStyle = '#FFD700';
    _roundRect(ctx, bx, by, rw, rh, 14);
    ctx.fill();

    // Dark text
    ctx.shadowBlur  = 0;
    ctx.fillStyle   = '#1a1a1a';
    ctx.fillText(text, bx + pad, by + rh / 2);
    ctx.restore();
  }

  async _ensureTesseract() {
    if (this.tesseract) return;
    const { createWorker } = await import(
      'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.esm.min.js'
    );
    this.tesseract = await createWorker('eng', 1, {
      tessedit_char_whitelist: '0123456789+-*/=().',
      tessedit_pageseg_mode:   '7',
    });
  }

  async _attemptMathOCR() {
    if (!this.mathBBox || this.gameMode !== 'drawing') return;
    const bbox = this.mathBBox; // snapshot — may be updated by next stroke
    try {
      await this._ensureTesseract();
      const blob = await extractMathImage(this.canvasMgr.baseCanvas, bbox);
      const url  = URL.createObjectURL(blob);
      const { data: { text } } = await this.tesseract.recognize(url);
      URL.revokeObjectURL(url);
      const cleaned = text.trim().replace(/\s+/g, '');
      // Match longest expression ending with =
      const match = cleaned.match(/[\d\+\-\*\/\.\(\)]+=/);
      if (!match) return;
      const result = safeMathEval(match[0]);
      if (result === null) return;
      this.mathResult = {
        text:     `= ${result}`,
        x:        Math.min(bbox.x2 + 14, FRAME_W - 80),
        y:        (bbox.y1 + bbox.y2) / 2,
        timer:    3.5,
        maxTimer: 3.5,
      };
      this.mathBBox = null; // reset for next expression
    } catch (err) {
      console.warn('Math OCR failed:', err);
    }
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
drawPixelBg();
window.addEventListener("resize", drawPixelBg);
new App();
