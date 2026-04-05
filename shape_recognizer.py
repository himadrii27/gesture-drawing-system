import cv2
import numpy as np
from typing import Optional

# Minimum stroke points before attempting recognition
MIN_POINTS = 20

# Circle: coefficient of variation of distances from centroid (lower = more circular)
CIRCLE_CV_THRESHOLD = 0.15

# Rectangle: approxPolyDP epsilon as fraction of arc length
RECT_EPSILON_RATIO = 0.04


def recognize(points: list) -> Optional[dict]:
    """
    Analyze a list of (x, y) canvas-space points and return a recognized shape or None.

    Returns one of:
      {'type': 'circle', 'cx': int, 'cy': int, 'r': int}
      {'type': 'rectangle', 'x': int, 'y': int, 'w': int, 'h': int}
      None  — unrecognized, keep as freehand
    """
    if len(points) < MIN_POINTS:
        return None

    pts = np.array(points, dtype=np.float32)
    pts_int = pts.astype(np.int32)

    # ── Rectangle check first (more specific) ───────────────────────────────────
    hull    = cv2.convexHull(pts_int)
    epsilon = RECT_EPSILON_RATIO * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / h if h > 0 else 0
        if w >= 30 and h >= 30 and 0.2 <= aspect <= 5.0:
            return {'type': 'rectangle', 'x': x, 'y': y, 'w': w, 'h': h}

    # ── Circle check ────────────────────────────────────────────────────────────
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    mean_r   = dists.mean()

    if mean_r < 20:
        return None

    cv = dists.std() / mean_r  # low = consistent radius = circle
    if cv < CIRCLE_CV_THRESHOLD:
        return {
            'type': 'circle',
            'cx': int(centroid[0]),
            'cy': int(centroid[1]),
            'r':  int(mean_r),
        }

    return None
