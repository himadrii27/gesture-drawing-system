import cv2
import numpy as np
from typing import Optional

# Process at 640x480 — Mac ignores cap.set() resolution requests for most cameras
PROCESS_W, PROCESS_H = 640, 480


class Camera:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)

    @property
    def is_open(self) -> bool:
        return self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Returns a (480, 640, 3) RGB uint8 numpy array, horizontally flipped.
        Resizes from native camera resolution (e.g. 1920x1080) down to
        PROCESS_W x PROCESS_H so MediaPipe runs efficiently.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame)

    def release(self):
        self.cap.release()
