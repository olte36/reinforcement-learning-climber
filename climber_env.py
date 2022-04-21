import gym
import climber_model
import routes
import pygame
import pygame_render
import numpy as np
import utils
import time
import random
from dataclasses import dataclass

CLIMB_DIRECTION_RANDOM = ["random", "r"]
CLIMB_DIRECTION_FROM_BOTTOM_TO_TOP = ["from_bottom_to_top", "bt"]
CLIMB_DIRECTION_FROM_TOP_TO_BOTTOM = ["from_top_to_bottom", "tb"]
CLIMB_DIRECTION_FROM_LEFT_TO_RIGHT = ["from_left_to_right", "lr"]
CLIMB_DIRECTION_FROM_RIGHT_TO_LEFT = ["from_right_to_left", "rl"]


class ClimberEnv(gym.Env):

    def __init__(self, route: np.ndarray, climber: climber_model.Climber = None, climb_direction: str = CLIMB_DIRECTION_RANDOM[0]):
        self._climber = climber_model.Climber() if climber is None else climber
        self._route = route
        #self._route = self._route[self._route[:, 1].argsort()]
        
        start = time.perf_counter()
        self._init_action_space()
        print("Action space is initialized in " + str(time.perf_counter() - start) + " sec")
        print("Action space size: " + str(self.action_space.n))

        start = time.perf_counter()
        self._init_observation_space()
        print("Observation space is initialized in " + str(time.perf_counter() - start) + " sec")
        print("Observation space size: " + str(self.observation_space.n))

        self._random_direction = climb_direction in CLIMB_DIRECTION_RANDOM
        if not self._random_direction:
            self._init_start_and_finish(climb_direction)

        self._eposodes_count = 0
        self._eposodes_max = 50
        self._render_surface = None


    def step(self, action):
        self._eposodes_count += 1
        if (self._eposodes_count == self._eposodes_max - 1):
            return (self._get_state_ind(), -120, True, "")

        decoded_action = self._actions[action]
        if decoded_action == CHANGE_SUPPORT_ACTION:
            self._climber.change_support()
            return (self._get_state_ind(), -1, False, "")

        if self._climber.is_transition_possible(limb=decoded_action.limb, point=decoded_action.hole):
            self._climber.do_transition(limb=decoded_action.limb, point=decoded_action.hole)
            done = np.array_equal(self._climber.left_hand_pos, self._finish_hole) \
                or np.array_equal(self._climber.right_hand_pos, self._finish_hole) \
                or np.array_equal(self._climber.left_leg_pos, self._finish_hole) \
                or np.array_equal(self._climber.right_leg_pos, self._finish_hole)
            if done:
                reward = 200
            else:
                reward = -1
                #reward = 500 - min(np.linalg.norm(self._finish_hole - self._climber.right_hand_pos), np.linalg.norm(self._finish_hole - self._climber.left_hand_pos))

            return (self._get_state_ind(), reward, done, "")

        return (self._get_state_ind(), -5, False, "")
             


    def reset(self):
        self._eposodes_count = 0
        if self._random_direction:
            self._start_state = random.choice(list(self._states.keys()))
            self._finish_hole = random.choice(self._route)

        self._climber.set_start_pos(
            left_hand=self._start_state.left_hand,
            right_hand=self._start_state.right_hand,
            left_leg=self._start_state.left_leg,
            right_leg=self._start_state.right_leg,
            support=self._start_state.support
        )
        return self._get_state_ind()


    def render(self, mode='human'):
        if self._render_surface is None:
            width = self._route[self._route[:, 0].argsort()[-1]][0] + 30
            height = self._route[self._route[:, 1].argsort()[-1]][1] + 30
            self._render_surface = pygame.display.set_mode((width, height))

        self._render_surface.fill(pygame_render.BLACK)
        pygame_render.render_route(self._route, self._render_surface)
        pygame_render.render_hole(self._finish_hole, self._render_surface, color=pygame_render.GREEN)
        pygame_render.render_climber(self._climber, self._render_surface)
        pygame.display.update()



    def close(self):
        pygame.quit()
        

    def _init_observation_space(self):
        self._states = {}

        max_distance = self._climber.hands_len + self._climber.torso_len + self._climber.legs_len
        max_distance += max_distance // 2
        quads = utils.quad_points_dist_np(self._route, max_distance)
        counter = 0
        for quad in quads:
            for left_hand in quad:
                for right_hand in quad:
                    for left_leg in quad:
                        for right_leg in quad:
                            if self._climber.can_start(left_hand=left_hand, right_hand=right_hand, left_leg=left_leg, right_leg=right_leg):
                                state1 = ClimberState(
                                    left_hand=left_hand, 
                                    right_hand=right_hand, 
                                    left_leg=left_leg, 
                                    right_leg=right_leg, 
                                    support=climber_model.RIGHT_HAND_LEFT_LEG
                                )
                                self._states[state1] = counter
                                counter += 1
                                state2 = state1.with_support(climber_model.LEFT_HAND_RIGH_LEG)
                                self._states[state2] = counter
                                counter += 1  

        self.observation_space = gym.spaces.Discrete(len(self._states))


    def _init_action_space(self):
        self._actions = []
        self._actions.append(CHANGE_SUPPORT_ACTION)
        for hole in self._route:
            for limb in (climber_model.LEFT_HAND, climber_model.RIGHT_HAND, climber_model.LEFT_LEG, climber_model.RIGHT_LEG):
                self._actions.append(ClimberAction(limb, hole))

        self.action_space = gym.spaces.Discrete(len(self._actions))


    def _init_start_and_finish(self, climb_direction: str):
        states_list = list(self._states.keys())
        start_state_ind = None
        finish_hole_ind = None
        if climb_direction in CLIMB_DIRECTION_FROM_BOTTOM_TO_TOP:
            start_state_ind = np.array([min(st.left_leg[1], st.right_leg[1]) for st in states_list]).argmin()
            finish_hole_ind = self._route[:, 1].argmax()
        
        elif climb_direction in CLIMB_DIRECTION_FROM_TOP_TO_BOTTOM:
            start_state_ind = np.array([max(st.left_hand[1], st.right_hand[1]) for st in states_list]).argmax()
            finish_hole_ind = self._route[:, 1].argmin()
        
        elif climb_direction in CLIMB_DIRECTION_FROM_LEFT_TO_RIGHT:
            start_state_ind = np.array([min(st.left_hand[0], st.left_leg[0]) for st in states_list]).argmin()
            finish_hole_ind = self._route[:, 0].argmax()

        elif climb_direction in CLIMB_DIRECTION_FROM_RIGHT_TO_LEFT:
            start_state_ind = np.array([max(st.right_hand[0], st.right_leg[0]) for st in states_list]).argmax()
            finish_hole_ind = self._route[:, 0].argmin()

        else:
            raise Exception("Unexpected climbing direction: " + str(climb_direction))

        self._start_state = states_list[start_state_ind]
        self._finish_hole = self._route[finish_hole_ind]


    def _get_state_ind(self):
        return self._states[ClimberState(
            left_hand=self._climber.left_hand_pos,
            right_hand=self._climber.right_hand_pos,
            left_leg=self._climber.left_leg_pos,
            right_leg=self._climber.right_leg_pos,
            support=self._climber.support
        )]

    # def _encode_obs(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, support: int):
    #     return (tuple(left_hand), tuple(right_hand), tuple(left_leg), tuple(right_leg), support)


@dataclass(frozen=True, init=False)
class ClimberState:

    _left_hand: tuple
    _right_hand: tuple
    _left_leg: tuple
    _right_leg: tuple
    _support: int


    def __init__(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, support: int):
        object.__setattr__(self, "_left_hand", tuple(left_hand))
        object.__setattr__(self, "_right_hand", tuple(right_hand))
        object.__setattr__(self, "_left_leg", tuple(left_leg))
        object.__setattr__(self, "_right_leg", tuple(right_leg))
        object.__setattr__(self, "_support", support)


    @property
    def left_hand(self) -> np.ndarray:
        return np.array(self._left_hand)

    @property
    def right_hand(self) -> np.ndarray:
        return np.array(self._right_hand)

    @property
    def left_leg(self) -> np.ndarray:
        return np.array(self._left_leg)

    @property
    def right_leg(self) -> np.ndarray:
        return np.array(self._right_leg)

    @property
    def support(self) -> np.ndarray:
        return self._support


    def with_support(self, support: int):
        return ClimberState(left_hand=self.left_hand, right_hand=self.right_hand, left_leg=self.left_leg, right_leg=self.right_leg, support=support)


@dataclass
class ClimberAction:

    limb: int
    hole: np.ndarray


CHANGE_SUPPORT_ACTION = ClimberAction(-1, np.empty(1))