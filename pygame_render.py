import pygame
import numpy as np

import climber_model

SCALE = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (125, 125, 125)
LIGHT_BLUE = (64, 128, 255)
DARK_BLUE = (0, 13, 138)
GREEN = (0, 200, 64)
YELLOW = (225, 225, 0)
PINK = (230, 50, 230)
ORANGE = (255, 128, 0)
SKIN = (255, 210, 135)


def render_climber(climber, surface, thickness=7):
    climber.adjust_body()
    left_leg_color = BLACK
    right_leg_color = BLACK
    left_hand_color = BLACK
    right_hand_color = BLACK
    body_color = BLACK
    joint_color = GRAY
    foots_color = DARK_BLUE
    hands_color = SKIN

    joint_thickness = thickness - 1 * SCALE
    #if climber.support == climber_model.RIGHT_HAND_LEFT_LEG:
    #    left_leg_color = LIGHT_BLUE
    #    right_hand_color = YELLOW
    #else:
    #    right_leg_color = LIGHT_BLUE
    #    left_hand_color = YELLOW

    # тело
    pygame.draw.line(surface, body_color,
        to_pygame(climber.shoulders_pos, surface.get_height()),
        to_pygame(climber.pelvis_pos, surface.get_height()),
        thickness)

    # левая рука
    pygame.draw.line(surface, left_hand_color, 
        to_pygame(climber.left_hand_pos, surface.get_height()),
        to_pygame(climber.left_elbow_pos, surface.get_height()),
        thickness)

    pygame.draw.line(surface, left_hand_color, 
        to_pygame(climber.shoulders_pos, surface.get_height()),
        to_pygame(climber.left_elbow_pos, surface.get_height()),
        thickness)
    
    pygame.draw.circle(surface, joint_color, to_pygame(climber.left_elbow_pos, surface.get_height()), joint_thickness)
    pygame.draw.circle(surface, hands_color, to_pygame(climber.left_hand_pos, surface.get_height()), joint_thickness)


    # правая рука
    pygame.draw.line(surface, right_hand_color, 
        to_pygame(climber.right_hand_pos, surface.get_height()),
        to_pygame(climber.right_elbow_pos, surface.get_height()),
        thickness)

    pygame.draw.line(surface, right_hand_color, 
        to_pygame(climber.shoulders_pos, surface.get_height()),
        to_pygame(climber.right_elbow_pos, surface.get_height()),
        thickness)

    pygame.draw.circle(surface, joint_color, to_pygame(climber.right_elbow_pos, surface.get_height()), joint_thickness)
    pygame.draw.circle(surface, hands_color, to_pygame(climber.right_hand_pos, surface.get_height()), joint_thickness)

        
    # левая нога
    if climber.left_leg_pos is None:
        pygame.draw.line(surface, left_leg_color,
            to_pygame(climber.pelvis_pos, surface.get_height()),
            to_pygame(climber.pelvis_pos + np.array([climber.legs_len, 0]), surface.get_height()),
            thickness)
    else:
        pygame.draw.line(surface, left_leg_color,
            to_pygame(climber.pelvis_pos, surface.get_height()),
            to_pygame(climber.left_knee_pos, surface.get_height()),
            thickness)

        pygame.draw.line(surface, left_leg_color,
            to_pygame(climber.left_knee_pos, surface.get_height()),
            to_pygame(climber.left_leg_pos, surface.get_height()),
            thickness)

        pygame.draw.circle(surface, joint_color, to_pygame(climber.left_knee_pos, surface.get_height()), joint_thickness)
        pygame.draw.circle(surface, foots_color, to_pygame(climber.left_leg_pos, surface.get_height()), joint_thickness)

    # правая нога
    if climber.right_leg_pos is None:
        pygame.draw.line(surface, right_leg_color,
            to_pygame(climber.pelvis_pos, surface.get_height()),
            to_pygame(climber.pelvis_pos + np.array([climber.legs_len, 0]), surface.get_height()),
            thickness)
    else:
        pygame.draw.line(surface, right_leg_color,
            to_pygame(climber.pelvis_pos, surface.get_height()),
            to_pygame(climber.right_knee_pos, surface.get_height()),
            thickness)

        pygame.draw.line(surface, right_leg_color, 
            to_pygame(climber.right_knee_pos, surface.get_height()),
            to_pygame(climber.right_leg_pos, surface.get_height()),
            thickness)
        
        pygame.draw.circle(surface, joint_color, to_pygame(climber.right_knee_pos, surface.get_height()), joint_thickness)
        pygame.draw.circle(surface, foots_color, to_pygame(climber.right_leg_pos, surface.get_height()), joint_thickness)
    

    pygame.draw.circle(surface, joint_color, to_pygame(climber.pelvis_pos, surface.get_height()), joint_thickness)

    # голова 
    pygame.draw.circle(surface, body_color, to_pygame(climber.head_pos, surface.get_height()), 10 * SCALE)

    # шея
    pygame.draw.line(surface, body_color, 
        to_pygame(climber.shoulders_pos, surface.get_height()),
        to_pygame(climber.head_pos, surface.get_height()),
        thickness)

    pygame.draw.circle(surface, joint_color, to_pygame(climber.shoulders_pos, surface.get_height()), joint_thickness)



def render_route(route, surface):
    for hole in route:
        render_hole(hole, surface)


def render_hole(hole, surface, color=PINK):
    pygame_hole = to_pygame(hole, surface.get_height())
    p1 = pygame_hole + np.array([-5 * SCALE, 0])
    p2 = pygame_hole + np.array([5 * SCALE, 0])
    p3 = pygame_hole + np.array([0, 5 * SCALE])
    pygame.draw.polygon(surface, color, [p1, p2, p3])
   # pygame.draw.circle(surface, color, , radius)


def render_quickdraw(quickdraw, surface, rope_thickness = 3, carbine_radius = 5, carbine_thickness = 2):
    pygame.draw.circle(surface, ORANGE, to_pygame(quickdraw, surface.get_height()), carbine_radius * SCALE, carbine_thickness * SCALE)
    p1 = np.copy(quickdraw)
    p1[1] += 15
    p2 = np.copy(quickdraw)
    p2[1] += 5
    pygame.draw.line(surface, LIGHT_BLUE,
        to_pygame(p1, surface.get_height()),
        to_pygame(p2, surface.get_height()),
        rope_thickness * SCALE
    )


def render_quickdraws(quickdraws, clipped, climber, surface, rope_thickness = 2):
    render_quickdraw(quickdraws[0], surface)
    for i in range(1, len(quickdraws)):
        curr = quickdraws[i]
        render_quickdraw(curr, surface)
        if clipped[i - 1] and clipped[i]:
            prev = quickdraws[i - 1]
            pygame.draw.line(surface, GRAY,
                to_pygame(prev, surface.get_height()),
                to_pygame(curr, surface.get_height()),
                rope_thickness * SCALE
            )
    clipped_quickdraws = quickdraws[clipped]
    if len(clipped_quickdraws) != 0:
        pygame.draw.line(surface, GRAY,
            to_pygame(climber.pelvis_pos, surface.get_height()),
            to_pygame(clipped_quickdraws[-1], surface.get_height()),
            rope_thickness * SCALE
        )
        pygame.draw.line(surface, GRAY,
            to_pygame(clipped_quickdraws[0], surface.get_height()),
            to_pygame([surface.get_width() // 2, 0], surface.get_height()),
            rope_thickness * SCALE
        )
    else:
        pygame.draw.line(surface, GRAY,
            to_pygame(climber.pelvis_pos, surface.get_height()),
            to_pygame([surface.get_width() // 2, 0], surface.get_height()),
            rope_thickness * SCALE
        )
    


def to_pygame(p, height):
    return np.array([p[0] * SCALE, height - p[1] * SCALE])


class Button:

    def __init__(self, pos, width, height, surface, font, text, color, on_click=None, text_color=WHITE):
        self._pos = pos
        self._width = width
        self._height = height
        self._text = text
        self._color = color
        self._surface = surface
        self._text_color = text_color
        self._font = font
        self._on_click = on_click


    def render(self):
        pygame.draw.rect(self._surface, self._color, (self._pos[0], self._pos[1], self._width, self._height))
        text = self._font.render(self._text, False, self._text_color)
        place = text.get_rect(center=(self._width // 2, self._height // 2))
        self._surface.blit(text, place)

    def click(self, event):
        print(str(self._surface.get_rect().collidepoint(event.pos)))
        if event.type == pygame.constants.MOUSEBUTTONDOWN \
            and self._surface.get_rect().collidepoint(event.pos) \
            and event.pos[0] >= self._pos[0] and event.pos[0] <= self._pos[0] + self._width \
            and event.pos[1] >= self._pos[1] and event.pos[1] <= self._pos[1] + self._height:

            if (self._on_click is not None):
                self._on_click(self)

    @property
    def pos(self):
        return self._pos

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def text(self):
        return self._text

    @property
    def color(self):
        return self._color

    @property
    def surface(self):
        return self._surface

    @property
    def text_color(self):
        return self._text_color
    
    @property
    def font(self):
        return self._font

    @property   
    def on_click(self):
        return self._on_click

    @pos.setter
    def pos(self, value):
        self._pos = value 
        self.render()

    @width.setter
    def width(self, value):
        self._width = value 
        self.render()

    @height.setter
    def height(self, value):
        self._height = value 
        self.render()

    @text.setter
    def text(self, value):
        self._text = value 
        self.render()

    @color.setter
    def color(self, value):
        self._color = value 
        self.render()

    @surface.setter
    def surface(self, value):
        self._surface = value
        self.render()

    @text_color.setter
    def text_color(self, value):
        self._text_color = value 
        self.render()

    @font.setter
    def font(self, value):
        self._font = value 
        self.render()

    @on_click.setter
    def on_click(self, value):
        self._on_click = value