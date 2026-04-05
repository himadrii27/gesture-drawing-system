import os
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional

INDEX_TIP = 8
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.5):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            # VIDEO mode uses temporal smoothing across frames — far more stable
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
        Process an RGB numpy frame (must be contiguous uint8).
        Returns (x, y) canvas-space pixel coords of the index finger tip,
        or None if no hand detected.
        frame_w / frame_h define the output coordinate space (the canvas size).
        """
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        tip = result.hand_landmarks[0][INDEX_TIP]
        x = int(tip.x * frame_w)
        y = int(tip.y * frame_h)
        return (x, y)

    def close(self):
        self._landmarker.close()
