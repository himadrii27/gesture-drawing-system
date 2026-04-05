import pygame
import random
import math
import numpy as np
from typing import Optional, List
from hand_tracker import HAND_CONNECTIONS
from particles import ParticleSystem

# ── Palette ────────────────────────────────────────────────────────────────────
BG_PIXEL_PALETTE = [
    (0, 82, 255),    # electric blue
    (0, 200, 150),   # teal green
    (255, 225, 53),  # yellow
    (255, 107, 157), # pink
    (127, 255, 219), # mint
    (45, 45, 45),    # dark tile
    (0, 140, 255),   # sky blue
    (180, 255, 100), # lime
]
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
DARK_TEXT   = (26, 26, 26)
CYAN_DOT    = (0, 229, 255)
BTN_IDLE    = (255, 255, 255)
BTN_ACTIVE  = (255, 59, 59)
PILL_BG     = (255, 255, 255)
FRAME_STROKE = (255, 255, 255)
OVERLAY_BG  = (0, 0, 0, 120)   # semi-transparent overlay strip

PIXEL_SIZE  = 20
WIN_W, WIN_H = 1280, 720
FRAME_W, FRAME_H = 860, 520
FRAME_X = (WIN_W - FRAME_W) // 2
FRAME_Y = (WIN_H - FRAME_H) // 2 - 10
FRAME_RADIUS = 40

BTN_RADIUS  = 28
BTN_X = FRAME_X + FRAME_W - BTN_RADIUS - 16
BTN_Y = FRAME_Y + FRAME_H - BTN_RADIUS - 16


class UI:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_pill  = pygame.font.SysFont("SF Pro Display, Helvetica Neue, Arial", 18, bold=False)
        self.font_label = pygame.font.SysFont("SF Pro Display, Helvetica Neue, Arial", 13)
        self._bg = self._make_pixel_bg()
        self._pulse_t = 0.0  # for button pulse animation

    # ── Background ─────────────────────────────────────────────────────────────
    def _make_pixel_bg(self) -> pygame.Surface:
        surf = pygame.Surface((WIN_W, WIN_H))
        rng = random.Random(42)
        for y in range(0, WIN_H, PIXEL_SIZE):
            for x in range(0, WIN_W, PIXEL_SIZE):
                color = rng.choice(BG_PIXEL_PALETTE)
                pygame.draw.rect(surf, color, (x, y, PIXEL_SIZE, PIXEL_SIZE))
        return surf

    # ── Camera frame clip mask ──────────────────────────────────────────────────
    def _make_frame_mask(self) -> pygame.Surface:
        mask = pygame.Surface((FRAME_W, FRAME_H), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 0))
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, FRAME_W, FRAME_H), border_radius=FRAME_RADIUS)
        return mask

    # ── Main render ─────────────────────────────────────────────────────────────
    def render(
        self,
        camera_frame: Optional[np.ndarray],
        canvas_surface: pygame.Surface,
        tracking: bool,
        finger_pos: Optional[tuple],
        landmarks: Optional[List[tuple]],
        pinching: bool,
        particles: ParticleSystem,
        dt: float,
    ):
        self._pulse_t += dt

        # 1. Pixel art background
        self.screen.blit(self._bg, (0, 0))

        # 2. Camera feed inside rounded rect
        if camera_frame is not None:
            self._draw_camera(camera_frame)

        # 3. Drawing canvas overlay
        self._draw_canvas_overlay(canvas_surface)

        # 4. Frame border stroke
        pygame.draw.rect(
            self.screen, FRAME_STROKE,
            (FRAME_X - 2, FRAME_Y - 2, FRAME_W + 4, FRAME_H + 4),
            width=3, border_radius=FRAME_RADIUS + 2,
        )

        # 5. Hand skeleton overlay
        if tracking and landmarks is not None:
            self._draw_skeleton(landmarks, finger_pos, pinching)

        # 5b. Particle trail
        particles.draw(self.screen, self._cam_to_screen)

        # 6. Circle toggle button
        self._draw_button(tracking)

        # 7. Instruction pill
        self._draw_pill(tracking, pinching)

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _draw_camera(self, frame: np.ndarray):
        """Blit camera frame clipped to rounded rectangle."""
        h, w = frame.shape[:2]
        cam_surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        cam_surf = pygame.transform.smoothscale(cam_surf, (FRAME_W, FRAME_H))

        # Clip with mask
        mask = self._make_frame_mask()
        cam_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Convert to per-pixel alpha so transparent corners are respected
        clipped = pygame.Surface((FRAME_W, FRAME_H), pygame.SRCALPHA)
        clipped.blit(cam_surf, (0, 0))
        self.screen.blit(clipped, (FRAME_X, FRAME_Y))

    def _draw_canvas_overlay(self, canvas_surface: pygame.Surface):
        """Scale canvas surface to frame size and blit it."""
        scaled = pygame.transform.smoothscale(canvas_surface, (FRAME_W, FRAME_H))
        self.screen.blit(scaled, (FRAME_X, FRAME_Y))

    def _draw_skeleton(self, landmarks: List[tuple], finger_pos: Optional[tuple], pinching: bool):
        """Draw all 21 hand landmarks + connecting bones on screen."""
        screen_pts = [self._cam_to_screen(x, y) for x, y in landmarks]

        # Bone lines — orange when pinching (drawing), white otherwise
        line_color = (255, 160, 40) if pinching else WHITE
        for a, b in HAND_CONNECTIONS:
            pygame.draw.line(self.screen, line_color, screen_pts[a], screen_pts[b], 2)

        # Landmark dots
        for i, pt in enumerate(screen_pts):
            is_index_tip = (i == 8)
            if is_index_tip and pinching:
                color = (255, 80, 0)   # orange = pen down
                radius = 9
            elif is_index_tip:
                color = CYAN_DOT       # cyan = pen up / hovering
                radius = 7
            else:
                color = WHITE
                radius = 5
            pygame.draw.circle(self.screen, color, pt, radius)
            pygame.draw.circle(self.screen, WHITE, pt, radius, 1)

    def _draw_button(self, active: bool):
        pulse = 0.0
        if active:
            pulse = (math.sin(self._pulse_t * 4) + 1) / 2 * 6  # 0–6 extra radius
        r = BTN_RADIUS + int(pulse)

        if active:
            # Glow halo
            halo = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            pygame.draw.circle(halo, (*BTN_ACTIVE, 60), (r * 2, r * 2), r + 10)
            self.screen.blit(halo, (BTN_X - r * 2 + BTN_RADIUS, BTN_Y - r * 2 + BTN_RADIUS))

        color = BTN_ACTIVE if active else BTN_IDLE
        pygame.draw.circle(self.screen, color, (BTN_X + BTN_RADIUS, BTN_Y + BTN_RADIUS), r)
        pygame.draw.circle(self.screen, WHITE, (BTN_X + BTN_RADIUS, BTN_Y + BTN_RADIUS), r, 2)

        # Inner icon: ● when idle, ■ when active
        cx, cy = BTN_X + BTN_RADIUS, BTN_Y + BTN_RADIUS
        if active:
            s = 10
            pygame.draw.rect(self.screen, WHITE, (cx - s // 2, cy - s // 2, s, s), border_radius=3)
        else:
            pygame.draw.circle(self.screen, BTN_ACTIVE, (cx, cy), 10)

    def _draw_pill(self, tracking: bool, pinching: bool = False):
        if not tracking:
            text = "Press ● to start drawing"
        elif pinching:
            text = "✏  Drawing...   •   DELETE to clear"
        else:
            text = "Pinch to draw   •   Open hand to pause   •   DELETE to clear"

        text_surf = self.font_pill.render(text, True, DARK_TEXT)
        pad_x, pad_y = 24, 12
        pill_w = text_surf.get_width() + pad_x * 2
        pill_h = text_surf.get_height() + pad_y * 2
        pill_x = WIN_W // 2 - pill_w // 2
        pill_y = FRAME_Y + FRAME_H + 16

        # Shadow
        shadow = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 40), (2, 4, pill_w, pill_h), border_radius=pill_h // 2)
        self.screen.blit(shadow, (pill_x, pill_y))

        # Pill background
        pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
        pygame.draw.rect(pill_surf, (*PILL_BG, 240), (0, 0, pill_w, pill_h), border_radius=pill_h // 2)
        self.screen.blit(pill_surf, (pill_x, pill_y))

        self.screen.blit(text_surf, (pill_x + pad_x, pill_y + pad_y))

    def _cam_to_screen(self, cx: int, cy: int) -> tuple[int, int]:
        """Map camera frame pixel coords to screen coords."""
        sx = FRAME_X + int(cx * FRAME_W / 860)
        sy = FRAME_Y + int(cy * FRAME_H / 520)
        return sx, sy

    @property
    def button_rect(self) -> pygame.Rect:
        return pygame.Rect(BTN_X, BTN_Y, BTN_RADIUS * 2, BTN_RADIUS * 2)

    @property
    def frame_rect(self) -> pygame.Rect:
        return pygame.Rect(FRAME_X, FRAME_Y, FRAME_W, FRAME_H)
