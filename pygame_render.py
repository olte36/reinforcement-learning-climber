import pygame
import climber_model


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (125, 125, 125)
LIGHT_BLUE = (64, 128, 255)
GREEN = (0, 200, 64)
YELLOW = (225, 225, 0)
PINK = (230, 50, 230)


def render_climber(climber, surface):
    climber.adjust_body()
    left_leg_color = WHITE
    right_leg_color = WHITE
    left_hand_color = WHITE
    right_hand_color = WHITE
    if climber.support == climber_model.RIGHT_HAND_LEFT_LEG:
        left_leg_color = LIGHT_BLUE
        right_hand_color = YELLOW
    else:
        right_leg_color = LIGHT_BLUE
        left_hand_color = YELLOW

    # голова 
    pygame.draw.circle(surface, WHITE, to_pygame(climber.head_pos, surface.get_height()), 10)

    # шея
    pygame.draw.line(surface, WHITE, 
        to_pygame(climber.shoulders_pos, surface.get_height()), 
        to_pygame(climber.head_pos, surface.get_height()))

    # левая рука
    pygame.draw.line(surface, left_hand_color, 
        to_pygame(climber.left_hand_pos, surface.get_height()), 
        to_pygame(climber.left_elbow_pos, surface.get_height()))

    pygame.draw.line(surface, left_hand_color, 
        to_pygame(climber.shoulders_pos, surface.get_height()), 
        to_pygame(climber.left_elbow_pos, surface.get_height()))

    # правая рука
    pygame.draw.line(surface, right_hand_color, 
        to_pygame(climber.right_hand_pos, surface.get_height()), 
        to_pygame(climber.right_elbow_pos, surface.get_height()))

    pygame.draw.line(surface, right_hand_color, 
        to_pygame(climber.shoulders_pos, surface.get_height()), 
        to_pygame(climber.right_elbow_pos, surface.get_height()))

    # тело
    pygame.draw.line(surface, WHITE, 
        to_pygame(climber.shoulders_pos, surface.get_height()), 
        to_pygame(climber.pelvis_pos, surface.get_height()))
        
    # левая нога
    pygame.draw.line(surface, left_leg_color, 
        to_pygame(climber.pelvis_pos, surface.get_height()), 
        to_pygame(climber.left_knee_pos, surface.get_height()))

    pygame.draw.line(surface, left_leg_color, 
        to_pygame(climber.left_knee_pos, surface.get_height()), 
        to_pygame(climber.left_leg_pos, surface.get_height()))

    # правая нога
    pygame.draw.line(surface, right_leg_color, 
        to_pygame(climber.pelvis_pos, surface.get_height()), 
        to_pygame(climber.right_knee_pos, surface.get_height()))

    pygame.draw.line(surface, right_leg_color, 
        to_pygame(climber.right_knee_pos, surface.get_height()), 
        to_pygame(climber.right_leg_pos, surface.get_height()))


def render_route(route, surface):
    for hole in route:
        render_hole(hole, surface)


def render_hole(hole, surface, color=PINK, radius=3):
    pygame.draw.circle(surface, color, to_pygame(hole, surface.get_height()), radius)


def to_pygame(p, height):
    return (p[0], height - p[1])


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