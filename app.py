import pygame
import sys

from camera import Camera
from hand_tracker import HandTracker
from canvas import Canvas
from ui import UI, WIN_W, WIN_H, FRAME_W, FRAME_H

FPS = 30


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Gesture Drawing")

        self.clock   = pygame.time.Clock()
        self.camera  = Camera()
        self.tracker = HandTracker()
        self.canvas  = Canvas(FRAME_W, FRAME_H)
        self.ui      = UI(self.screen)

        self.tracking   = False
        self.finger_pos = None
        self.landmarks  = None

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update()
            self._render(dt)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                    self.canvas.clear()
                elif event.key == pygame.K_ESCAPE:
                    self._quit()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.ui.button_rect.collidepoint(event.pos):
                    self.tracking = not self.tracking
                    if not self.tracking:
                        self.canvas.lift_pen()
                        self.finger_pos = None
                        self.landmarks  = None

    def _update(self):
        frame = self.camera.get_frame()
        self._last_frame = frame

        if self.tracking and frame is not None:
            tip_pos, landmarks = self.tracker.process(frame, FRAME_W, FRAME_H)
            self.landmarks = landmarks

            if tip_pos is not None:
                self.canvas.draw_point(*tip_pos)
                self.finger_pos = tip_pos
            else:
                self.canvas.lift_pen()
                self.finger_pos = None
        else:
            self.finger_pos = None
            self.landmarks  = None

    def _render(self, dt: float):
        self.ui.render(
            camera_frame=getattr(self, "_last_frame", None),
            canvas_surface=self.canvas.surface,
            tracking=self.tracking,
            finger_pos=self.finger_pos,
            landmarks=self.landmarks,
            dt=dt,
        )
        pygame.display.flip()

    def _quit(self):
        self.camera.release()
        self.tracker.close()
        pygame.quit()
        sys.exit()
