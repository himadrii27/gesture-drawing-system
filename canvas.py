import pygame
from typing import Optional


class Canvas:
    STROKE_WIDTH = 6
    STROKE_COLOR = (255, 255, 255)  # white

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))
        self._prev: Optional[tuple[int, int]] = None

    # If finger jumps more than this many pixels in one frame, lift the pen
    # instead of drawing a wild line across the canvas
    JUMP_THRESHOLD = 80

    def draw_point(self, x: int, y: int):
        """Draw a smooth line segment from the previous point to (x, y)."""
        if self._prev is not None:
            dx = x - self._prev[0]
            dy = y - self._prev[1]
            if (dx * dx + dy * dy) ** 0.5 > self.JUMP_THRESHOLD:
                # Position jumped — start a fresh stroke from here next frame
                self._prev = (x, y)
                return
            pygame.draw.line(
                self.surface,
                self.STROKE_COLOR,
                self._prev,
                (x, y),
                self.STROKE_WIDTH,
            )
            # Round caps at each point to avoid jagged joins
            pygame.draw.circle(self.surface, self.STROKE_COLOR, (x, y), self.STROKE_WIDTH // 2)
            pygame.draw.circle(self.surface, self.STROKE_COLOR, self._prev, self.STROKE_WIDTH // 2)
        else:
            pygame.draw.circle(self.surface, self.STROKE_COLOR, (x, y), self.STROKE_WIDTH // 2)
        self._prev = (x, y)

    def lift_pen(self):
        """Call when tracking pauses so next draw_point starts a fresh stroke."""
        self._prev = None

    def clear(self):
        self.surface.fill((0, 0, 0, 0))
        self._prev = None
