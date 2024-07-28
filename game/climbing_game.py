import pygame
import numpy as np

class ClimbingGame:


    def __init__(self, climb_zone_width: float, climb_zone_height: float, scale: float = 1) -> None:
        self._game_actions = {}

        indent = 30
        self._scale = scale
        self._window_width = climb_zone_width * self._scale + indent * 2
        self._window_height = climb_zone_height * self._scale + indent * 2
        self._climb_zone = pygame.Rect(indent, indent, climb_zone_width * self._scale, climb_zone_height * self._scale)
        self._route = []
        


    def run(self) -> None:
        run = True
        main_surface = pygame.display.set_mode((self._window_width, self._window_height))
        while run:
            for event in pygame.event.get():
                if event.type == pygame.constants.QUIT:
                    run = False
                    pygame.quit()
                
                if event.type == pygame.constants.MOUSEBUTTONDOWN:
                    if self._climb_zone.collidepoint(event.pos):
                        self._route.append(event.pos)