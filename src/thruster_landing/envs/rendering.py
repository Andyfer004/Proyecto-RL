"""Pygame opcional: nunca se importa durante una simulación sin gráficos."""
import numpy as np
from .physics import ACTIONS, ACTION_NAMES, geometry, rotation, thrusters


class Renderer:
    def __init__(self, mode, config):
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("Instala los gráficos con: pip install -e '.[render]'") from exc
        self.pg, self.mode, self.config = pygame, mode, config
        pygame.display.init()
        pygame.font.init()
        self.size = (900, 800)
        self.surface = pygame.Surface(self.size)
        self.window = pygame.display.set_mode(self.size) if mode == "human" else None
        if self.window:
            pygame.display.set_caption("Thruster Landing — simulador académico")
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.close_requested = False

    def point(self, point):
        c = self.config
        scale = min(840 / (2 * c.x_limit), 660 / c.ceiling)
        return (int(450 + point[0] * scale), int(740 - point[1] * scale))

    def manual_action(self):
        keys = self.pg.key.get_pressed()
        main = keys[self.pg.K_SPACE]
        left, right = keys[self.pg.K_RIGHT], keys[self.pg.K_LEFT]
        if keys[self.pg.K_UP]:
            return 7
        for action, mask in enumerate(ACTIONS):
            if tuple(mask) == (bool(main), bool(left), bool(right)):
                return action
        return 0

    def draw(self, state, action, time, wind):
        pg, c = self.pg, self.config
        if self.window:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    self.close_requested = True
        self.surface.fill((18, 24, 40))
        pg.draw.line(self.surface, (130, 140, 160), self.point((-c.x_limit, 0)), self.point((c.x_limit, 0)), 2)
        pg.draw.line(self.surface, (70, 220, 150), self.point((-c.platform_half_width, 0)),
                     self.point((c.platform_half_width, 0)), 7)
        points = geometry(state, c)
        pg.draw.polygon(self.surface, (110, 180, 245), [self.point(p) for p in points[:4]])
        for body, foot in ((0, 4), (1, 5)):
            pg.draw.line(self.surface, (240, 240, 250), self.point(points[body]), self.point(points[foot]), 3)
            pg.draw.circle(self.surface, (240, 240, 250), self.point(points[foot]), 4)
        if state.gas > 0:
            locations, _ = thrusters(c)
            for active, local in zip(ACTIONS[action], locations):
                if active:
                    origin = rotation(state.theta) @ local + [state.x, state.y]
                    end = origin + rotation(state.theta) @ [0.0, -0.6]
                    pg.draw.line(self.surface, (255, 170, 65), self.point(origin), self.point(end), 6)
        rows = [f"t={time:.2f} s | gas={state.gas:.0f}/{c.gas_capacity:.0f} N s | viento={wind:.1f} N",
                f"vx={state.vx:.2f} vy={state.vy:.2f} m/s | angulo={state.theta:.3f} rad | {ACTION_NAMES[action]}",
                "Espacio: central | izquierda/derecha: inclinar | arriba: todos | Esc: salir"]
        for index, row in enumerate(rows):
            self.surface.blit(self.font.render(row, True, (230, 235, 245)), (20, 15 + index * 26))
        if self.window:
            self.window.blit(self.surface, (0, 0))
            pg.display.flip()
            self.clock.tick(max(1, round(1 / c.decision_dt)))
            return None
        return np.transpose(pg.surfarray.array3d(self.surface), (1, 0, 2)).copy()

    def close(self):
        if self.window:
            self.pg.display.quit()
        self.pg.font.quit()
