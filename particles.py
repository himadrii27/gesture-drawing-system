import pygame
import random
import math
from typing import List


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'size', 'lifetime', 'age', 'color')

    def __init__(self, x: float, y: float, color: tuple):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(30, 90)
        upward_bias = random.uniform(20, 60)

        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - upward_bias
        self.size = random.uniform(2, 5)
        self.lifetime = random.uniform(0.4, 0.9)
        self.age = 0.0
        self.color = color

    @property
    def alive(self) -> bool:
        return self.age < self.lifetime


class ParticleSystem:
    GRAVITY = 120.0
    SPAWN_COUNT = 3

    def __init__(self):
        self._particles: List[Particle] = []

    def spawn(self, x: float, y: float, color: tuple):
        for _ in range(self.SPAWN_COUNT):
            self._particles.append(Particle(x, y, color))

    def update(self, dt: float):
        for p in self._particles:
            p.age += dt
            p.vx *= max(0.0, 1 - 2.0 * dt)
            p.vy += self.GRAVITY * dt
            p.x  += p.vx * dt
            p.y  += p.vy * dt
        self._particles = [p for p in self._particles if p.alive]

    def draw(self, screen: pygame.Surface, cam_to_screen_fn):
        for p in self._particles:
            alpha = max(0, min(255, int(255 * (1.0 - p.age / p.lifetime))))
            sx, sy = cam_to_screen_fn(p.x, p.y)
            r = max(1, int(p.size))
            surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p.color, alpha), (r + 1, r + 1), r)
            screen.blit(surf, (int(sx) - r - 1, int(sy) - r - 1))
