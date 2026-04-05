import pygame
from typing import Optional


class Canvas:
    STROKE_WIDTH = 6
    STROKE_COLOR = (255, 255, 255)

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))
        self._prev: Optional[tuple] = None

    def draw_point(self, x: int, y: int, color: tuple = None):
        """
        Draw a glow stroke segment from the previous point to (x, y).
        color: (r, g, b) RGB tuple. Defaults to white if None.
        Renders 3 alpha layers: outer glow → inner glow → core stroke.
        """
        r, g, b = color if color is not None else self.STROKE_COLOR
        w = self.STROKE_WIDTH

        if self._prev is not None:
            dx = x - self._prev[0]
            dy = y - self._prev[1]
            if (dx * dx + dy * dy) ** 0.5 > 80:
                self._prev = (x, y)
                return

            prev = self._prev
            curr = (x, y)

            # Outer glow
            pygame.draw.line(self.surface,   (r, g, b, 35),  prev, curr, w * 4)
            pygame.draw.circle(self.surface, (r, g, b, 35),  curr, (w * 4) // 2)
            pygame.draw.circle(self.surface, (r, g, b, 35),  prev, (w * 4) // 2)
            # Inner glow
            pygame.draw.line(self.surface,   (r, g, b, 80),  prev, curr, w * 2)
            pygame.draw.circle(self.surface, (r, g, b, 80),  curr, (w * 2) // 2)
            pygame.draw.circle(self.surface, (r, g, b, 80),  prev, (w * 2) // 2)
            # Core stroke
            pygame.draw.line(self.surface,   (r, g, b, 255), prev, curr, w)
            pygame.draw.circle(self.surface, (r, g, b, 255), curr, w // 2)
            pygame.draw.circle(self.surface, (r, g, b, 255), prev, w // 2)
        else:
            # First point of a new stroke — just dot with glow
            pygame.draw.circle(self.surface, (r, g, b, 35),  (x, y), (w * 4) // 2)
            pygame.draw.circle(self.surface, (r, g, b, 80),  (x, y), (w * 2) // 2)
            pygame.draw.circle(self.surface, (r, g, b, 255), (x, y), w // 2)

        self._prev = (x, y)

    def lift_pen(self):
        self._prev = None

    def clear(self):
        self.surface.fill((0, 0, 0, 0))
        self._prev = None
