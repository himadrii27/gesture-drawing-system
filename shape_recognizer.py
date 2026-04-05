import cv2
import math
import numpy as np
from typing import Optional

MIN_POINTS = 30                # enough points to cover most of a shape
CIRCLE_CV_THRESHOLD = 0.35    # coefficient of variation — low = consistent radius
RECT_EPSILON_RATIO  = 0.04    # approxPolyDP tolerance

# Closure: start and end must be within this fraction of the bounding-box diagonal
CLOSURE_RATIO = 0.30


def _is_closed(pts: np.ndarray) -> bool:
    """Return True if the stroke loops back close to where it started."""
    first = pts[0]
    last  = pts[-1]
    closure_dist = float(np.linalg.norm(last - first))
    bbox_diag = math.hypot(
        float(pts[:, 0].max() - pts[:, 0].min()),
        float(pts[:, 1].max() - pts[:, 1].min()),
    )
    return bbox_diag > 0 and (closure_dist / bbox_diag) < CLOSURE_RATIO


def _rect_angles_ok(approx: np.ndarray) -> bool:
    """All 4 corners of a candidate rectangle must be within 30° of 90°."""
    corners = approx.reshape(-1, 2).astype(np.float64)
    n = len(corners)
    for i in range(n):
        p_prev   = corners[(i - 1) % n]
        p_vertex = corners[i]
        p_next   = corners[(i + 1) % n]
        v1 = p_prev   - p_vertex
        v2 = p_next   - p_vertex
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return False
        cos_a = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_a))
        if not (60 <= angle <= 120):   # must be roughly 90°
            return False
    return True


def recognize(points: list) -> Optional[dict]:
    """
    Returns {'type': 'circle', ...} or {'type': 'rectangle', ...} or None.
    Only fires when the stroke is closed (loops back to start).
    """
    if len(points) < MIN_POINTS:
        return None

    pts     = np.array(points, dtype=np.float32)
    pts_int = pts.astype(np.int32)

    # Gate: shape must be a closed stroke
    if not _is_closed(pts):
        return None

    # ── Rectangle first (most specific) ─────────────────────────────────────────
    hull    = cv2.convexHull(pts_int)
    epsilon = RECT_EPSILON_RATIO * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 4 and _rect_angles_ok(approx):
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / h if h > 0 else 0
        if w >= 30 and h >= 30 and 0.2 <= aspect <= 5.0:
            return {'type': 'rectangle', 'x': x, 'y': y, 'w': w, 'h': h}

    # ── Circle ───────────────────────────────────────────────────────────────────
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    mean_r   = dists.mean()

    if mean_r < 20:
        return None

    cv = dists.std() / mean_r
    if cv < CIRCLE_CV_THRESHOLD:
        return {
            'type': 'circle',
            'cx': int(centroid[0]),
            'cy': int(centroid[1]),
            'r':  int(mean_r),
        }

    return None
