import pygame
import sys
import colorsys
import time

from camera import Camera
from hand_tracker import HandTracker
from canvas import Canvas
from ui import UI, WIN_W, WIN_H, FRAME_W, FRAME_H
from particles import ParticleSystem

FPS = 30


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Gesture Drawing")

        self.clock     = pygame.time.Clock()
        self.camera    = Camera()
        self.tracker   = HandTracker()
        self.canvas    = Canvas(FRAME_W, FRAME_H)
        self.ui        = UI(self.screen)
        self.particles = ParticleSystem()

        self.tracking      = False
        self.finger_pos    = None
        self.landmarks     = None
        self.pinching      = False
        self._was_pinching = False
        self._last_draw_t  = 0.0    # timestamp of last draw_point call

    def _get_rainbow_color(self) -> tuple:
        hue = (time.time() * 0.15) % 1.0   # full rainbow cycle every ~6.7s
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return (int(r * 255), int(g * 255), int(b * 255))

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._update(dt)
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
                        self.canvas.finalize_stroke()
                        self.finger_pos    = None
                        self.landmarks     = None
                        self.pinching      = False
                        self._was_pinching = False
                        self._last_draw_t  = 0.0

    def _update(self, dt: float):
        frame = self.camera.get_frame()
        self._last_frame = frame

        # Always advance particles so they fade out naturally after tracking stops
        self.particles.update(dt)

        if self.tracking and frame is not None:
            tip_pos, landmarks, pinching = self.tracker.process(frame, FRAME_W, FRAME_H)
            self.landmarks  = landmarks
            self.finger_pos = tip_pos
            self.pinching   = pinching

            if tip_pos is not None and pinching:
                color = self._get_rainbow_color()
                self.canvas.draw_point(*tip_pos, color=color)
                self.particles.spawn(*tip_pos, color)
                self._last_draw_t = time.time()
            elif self._was_pinching:
                # Pen just lifted — break stroke continuity but keep points
                self.canvas.lift_pen()

            self._was_pinching = pinching
        else:
            if self._was_pinching:
                self.canvas.lift_pen()
            self.finger_pos    = None
            self.landmarks     = None
            self.pinching      = False
            self._was_pinching = False

        # After 0.8s of no drawing, finalize the stroke (runs shape recognition)
        if self._last_draw_t and (time.time() - self._last_draw_t) > 0.8:
            self.canvas.finalize_stroke()
            self._last_draw_t = 0.0

    def _render(self, dt: float):
        self.ui.render(
            camera_frame=getattr(self, "_last_frame", None),
            canvas_surface=self.canvas.surface,
            tracking=self.tracking,
            finger_pos=self.finger_pos,
            landmarks=self.landmarks,
            pinching=self.pinching,
            particles=self.particles,
            dt=dt,
        )
        pygame.display.flip()

    def _quit(self):
        self.camera.release()
        self.tracker.close()
        pygame.quit()
        sys.exit()
