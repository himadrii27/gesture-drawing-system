import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from typing import Optional

INDEX_TIP = 8

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class HandTracker:
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        BaseOptions = mp_python.BaseOptions
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, frame_w: int, frame_h: int) -> Optional[tuple]:
        """
        Process an RGB numpy frame and return (x, y) pixel coords of the index
        finger tip relative to frame_w/frame_h, or None if no hand detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        tip = result.hand_landmarks[0][INDEX_TIP]
        x = int(tip.x * frame_w)
        y = int(tip.y * frame_h)
        return (x, y)

    def close(self):
        self._landmarker.close()
