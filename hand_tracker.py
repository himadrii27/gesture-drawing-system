import os
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional

INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

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
        self._start_ms = int(time.time() * 1000)

    def process(self, frame: np.ndarray, frame_w: int, frame_h: int):
        """
        Returns (index_tip_pos, landmarks) where:
          - index_tip_pos: (x, y) in canvas space if index finger is extended, else None
          - landmarks: list of 21 (x, y) pixel coords in canvas space, or None if no hand
        """
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, None

        lm = result.hand_landmarks[0]

        # All 21 landmarks in canvas-space pixel coords
        landmarks = [
            (int(p.x * frame_w), int(p.y * frame_h))
            for p in lm
        ]

        # Only return drawing position when index finger is clearly extended
        index_extended = lm[INDEX_TIP].y < lm[INDEX_PIP].y
        tip_above_mcp  = lm[INDEX_TIP].y < lm[INDEX_MCP].y
        index_tip_pos  = landmarks[INDEX_TIP] if (index_extended and tip_above_mcp) else None

        return index_tip_pos, landmarks

    def close(self):
        self._landmarker.close()
