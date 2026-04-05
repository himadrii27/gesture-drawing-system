import os
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional

# MediaPipe Hand landmark indices
INDEX_TIP = 8    # index fingertip
INDEX_PIP = 6    # index finger middle knuckle (proximal interphalangeal)
INDEX_MCP = 5    # index finger base knuckle (metacarpophalangeal)
MIDDLE_TIP = 12  # middle fingertip (used to distinguish point vs fist)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


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

    def process(self, frame: np.ndarray, frame_w: int, frame_h: int) -> Optional[tuple]:
        """
        Process an RGB contiguous uint8 frame.
        Returns (x, y) canvas-space coords of the index fingertip ONLY when
        the index finger is clearly extended (pointing gesture).
        Returns None if no hand found or finger is curled/fist.
        """
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]

        # Index finger is extended when its tip is above (lower y) its PIP knuckle
        # AND the middle finger tip is below the index tip (pointing gesture, not peace sign mess)
        index_extended = lm[INDEX_TIP].y < lm[INDEX_PIP].y
        # Extra check: tip must be clearly above the base knuckle
        tip_above_mcp = lm[INDEX_TIP].y < lm[INDEX_MCP].y

        if not (index_extended and tip_above_mcp):
            return None

        x = int(lm[INDEX_TIP].x * frame_w)
        y = int(lm[INDEX_TIP].y * frame_h)
        return (x, y)

    def close(self):
        self._landmarker.close()
