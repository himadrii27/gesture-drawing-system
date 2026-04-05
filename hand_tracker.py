import os
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

INDEX_TIP = 8

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
        self._start_ms   = int(time.time() * 1000)

    def process(self, frame: np.ndarray, frame_w: int, frame_h: int):
        """
        Returns (index_tip_pos, landmarks) where:
          - index_tip_pos: (x, y) canvas-space coords of index tip, or None
          - landmarks:     list of 21 (x, y) canvas-space coords, or None
        Drawing state is controlled by the K key in app.py, not here.
        """
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result   = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None, None

        lm = result.hand_landmarks[0]
        landmarks = [(int(p.x * frame_w), int(p.y * frame_h)) for p in lm]
        return landmarks[INDEX_TIP], landmarks

    def close(self):
        self._landmarker.close()
