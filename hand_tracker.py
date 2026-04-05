import os
import time
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional

INDEX_TIP  = 8
INDEX_PIP  = 6
INDEX_MCP  = 5
THUMB_TIP  = 4

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Hysteresis thresholds — prevents flickering between draw/no-draw states.
# Once pinching starts (distance < ON), keep drawing until distance > OFF.
PINCH_ON_THRESHOLD  = 0.06   # enter pinch state — fingers must be clearly touching
PINCH_OFF_THRESHOLD = 0.12   # exit pinch state — fingers must be clearly open

# Standard MediaPipe hand skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                   # wrist to pinky base
]


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.5):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_ms  = int(time.time() * 1000)
        self._pinching  = False   # tracks current state for hysteresis

    def process(self, frame: np.ndarray, frame_w: int, frame_h: int):
        """
        Returns (index_tip_pos, landmarks, pinching) where:
          - index_tip_pos: (x, y) canvas-space coords of index tip (always when hand visible)
          - landmarks:     list of 21 (x, y) canvas-space coords, or None if no hand
          - pinching:      True when thumb and index tips are close together (pen-down gesture)
        """
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._pinching = False   # reset so re-entering frame requires fresh pinch
            return None, None, False

        lm = result.hand_landmarks[0]

        landmarks = [
            (int(p.x * frame_w), int(p.y * frame_h))
            for p in lm
        ]

        # Pinch detection with hysteresis — avoids rapid flicker
        dx = lm[THUMB_TIP].x - lm[INDEX_TIP].x
        dy = lm[THUMB_TIP].y - lm[INDEX_TIP].y
        dist = math.hypot(dx, dy)

        if self._pinching:
            # Already drawing — only stop when fingers clearly open
            if dist > PINCH_OFF_THRESHOLD:
                self._pinching = False
        else:
            # Not drawing — only start when fingers clearly close
            if dist < PINCH_ON_THRESHOLD:
                self._pinching = True

        return landmarks[INDEX_TIP], landmarks, self._pinching

    def close(self):
        self._landmarker.close()
