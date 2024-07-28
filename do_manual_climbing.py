import pygame
import pygame_render
import climber_model
import routes
import sys
import numpy as np


def generate_simple_route(zone_width, zone_heigth, count):
    prev = (110, 30)
    route = []
    for i in range(count):
        for p in range(4):
            point = None
            if p == 1 or p == 2:
                point = (prev[0] + 60, prev[1] + 30)
            elif p == 0 or p == 3 or p == 4:
                point = (prev[0] - 60, prev[1] + 30)
            route.append(point)
            prev = point
    return route


def create_route_button_on_click(button):
    button.color = pygame_render.LIGHT_BLUE


WIDTH = 500
HEIGHT = 500
# ROUTE = generate_simple_route(120, 120, 3)
ROUTE = routes.generate_random_route(WIDTH - 150, HEIGHT, 40)

FPS = 30

if __name__ == '__main__':
    pygame.init()

    main_surface = pygame.display.set_mode((WIDTH, HEIGHT))

    create_route_button = pygame_render.Button(pos=(25, 25), width=100, height=40, text="Create route", color=pygame_render.GREEN, surface=main_surface, font=pygame.font.SysFont('arial', 24))
    create_route_button.on_click = create_route_button_on_click

    climber = climber_model.Climber()
    climber.set_start_pos(
        ROUTE[4], #left hand
        ROUTE[3], #right hand
        ROUTE[0], #left leg
        None, #right leg
        climber_model.RIGHT_HAND_LEFT_LEG)
    climber.adjust_body()
    
    clock = pygame.time.Clock()
    curr_limb = None
    while True:
        for i in pygame.event.get():
            if i.type == pygame.constants.QUIT:
                sys.exit()
            if i.type == pygame.constants.MOUSEBUTTONDOWN:
                create_route_button.click(i)
                hole = None
                for h in ROUTE:
                    pos = pygame_render.to_pygame(np.array(i.pos), HEIGHT)
                    if np.linalg.norm(np.array(pos) - np.array(h)) <= 3:
                        hole = np.array(h)
                        break
                if curr_limb is not None:
                    is_posible = climber.is_transition_possible(curr_limb, hole)
                    if is_posible:
                        climber.do_transition(curr_limb, hole)
                        climber.adjust_body()
            if i.type == pygame.constants.KEYDOWN:
                # 1 (49)- правая рука
                # 2 (50)- левая рука
                # 3 (51)- правая нога
                # 4 (52)- левая нога
                # 5 (53)- поменять опорные руку и ногу
                if i.key == 49:
                    curr_limb = climber_model.RIGHT_HAND
                elif i.key == 50:
                    curr_limb = climber_model.LEFT_HAND
                elif i.key == 51:
                    curr_limb = climber_model.RIGHT_LEG
                elif i.key == 52:
                    curr_limb = climber_model.LEFT_LEG
                elif i.key == 53:
                    climber.change_support()
                    climber.adjust_body()

        climbing_surface.fill(pygame_render.BLACK)
        pygame_render.render_route(ROUTE, main_surface)
        pygame_render.render_climber(climber, main_surface)
        create_route_button.render()
        
        pygame.display.update()
        clock.tick(FPS)