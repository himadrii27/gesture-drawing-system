import pygame
from typing import Optional
from shape_recognizer import recognize


class Canvas:
    STROKE_WIDTH = 6
    STROKE_COLOR = (255, 255, 255)

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self.base_surface    = pygame.Surface((width, height), pygame.SRCALPHA)
        self.current_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.base_surface.fill((0, 0, 0, 0))
        self.current_surface.fill((0, 0, 0, 0))

        self._prev: Optional[tuple] = None
        # Accumulated across the whole shape — NOT cleared on pen lift,
        # only cleared in finalize_stroke() after inactivity timeout.
        self._stroke_points: list = []
        self._stroke_color: tuple = self.STROKE_COLOR

    @property
    def surface(self) -> pygame.Surface:
        composite = self.base_surface.copy()
        composite.blit(self.current_surface, (0, 0))
        return composite

    # ── Drawing ─────────────────────────────────────────────────────────────────

    def draw_point(self, x: int, y: int, color: tuple = None):
        r, g, b = color if color is not None else self.STROKE_COLOR
        self._stroke_color = (r, g, b)
        w = self.STROKE_WIDTH

        if self._prev is not None:
            dx = x - self._prev[0]
            dy = y - self._prev[1]
            if (dx * dx + dy * dy) ** 0.5 > 80:
                self._prev = (x, y)
                return

            prev = self._prev
            curr = (x, y)
            pygame.draw.line(self.current_surface,   (r, g, b, 35),  prev, curr, w * 4)
            pygame.draw.circle(self.current_surface, (r, g, b, 35),  curr, (w * 4) // 2)
            pygame.draw.circle(self.current_surface, (r, g, b, 35),  prev, (w * 4) // 2)
            pygame.draw.line(self.current_surface,   (r, g, b, 80),  prev, curr, w * 2)
            pygame.draw.circle(self.current_surface, (r, g, b, 80),  curr, (w * 2) // 2)
            pygame.draw.circle(self.current_surface, (r, g, b, 80),  prev, (w * 2) // 2)
            pygame.draw.line(self.current_surface,   (r, g, b, 255), prev, curr, w)
            pygame.draw.circle(self.current_surface, (r, g, b, 255), curr, w // 2)
            pygame.draw.circle(self.current_surface, (r, g, b, 255), prev, w // 2)
        else:
            pygame.draw.circle(self.current_surface, (r, g, b, 35),  (x, y), (w * 4) // 2)
            pygame.draw.circle(self.current_surface, (r, g, b, 80),  (x, y), (w * 2) // 2)
            pygame.draw.circle(self.current_surface, (r, g, b, 255), (x, y), w // 2)

        self._stroke_points.append((x, y))
        self._prev = (x, y)

    def lift_pen(self):
        """Only lifts the pen (breaks line continuity). Does NOT clear points.
        Points accumulate across gaps until finalize_stroke() is called."""
        self._prev = None

    def finalize_stroke(self):
        """Run shape recognition on all accumulated points, then reset.
        Called after inactivity timeout — sees the complete shape."""
        if self._stroke_points:
            shape = recognize(self._stroke_points)
            if shape:
                self.current_surface.fill((0, 0, 0, 0))
                self._draw_shape(shape, self._stroke_color)
            else:
                self.base_surface.blit(self.current_surface, (0, 0))
                self.current_surface.fill((0, 0, 0, 0))
        else:
            self.current_surface.fill((0, 0, 0, 0))

        self._stroke_points = []
        self._prev = None

    # ── Perfect shape renderer ───────────────────────────────────────────────────

    def _draw_shape(self, shape: dict, color: tuple):
        r, g, b = color
        w = self.STROKE_WIDTH

        if shape['type'] == 'circle':
            cx, cy, radius = shape['cx'], shape['cy'], shape['r']
            for alpha, width in ((35, w * 4), (80, w * 2), (255, w)):
                pygame.draw.circle(self.base_surface, (r, g, b, alpha),
                                   (cx, cy), radius, width)

        elif shape['type'] == 'rectangle':
            rect = pygame.Rect(shape['x'], shape['y'], shape['w'], shape['h'])
            for alpha, width in ((35, w * 4), (80, w * 2), (255, w)):
                pygame.draw.rect(self.base_surface, (r, g, b, alpha),
                                 rect, width)

    # ── Clear ────────────────────────────────────────────────────────────────────

    def clear(self):
        self.base_surface.fill((0, 0, 0, 0))
        self.current_surface.fill((0, 0, 0, 0))
        self._stroke_points = []
        self._prev = None
