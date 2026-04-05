import cv2
import numpy as np
from typing import Optional


class Camera:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 860)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    @property
    def is_open(self) -> bool:
        return self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns an RGB numpy array (H, W, 3), horizontally flipped, or None on failure."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()
