import time
import random
from dataclasses import dataclass

import gym
import numpy as np
import pygame

import climber_model
import pygame_render
import utils


CLIMB_DIRECTION_RANDOM = ["random", "r"]
CLIMB_DIRECTION_FROM_BOTTOM_TO_TOP = ["from_bottom_to_top", "bt"]
CLIMB_DIRECTION_FROM_TOP_TO_BOTTOM = ["from_top_to_bottom", "tb"]
CLIMB_DIRECTION_FROM_LEFT_TO_RIGHT = ["from_left_to_right", "lr"]
CLIMB_DIRECTION_FROM_RIGHT_TO_LEFT = ["from_right_to_left", "rl"]


class ClimberEnv(gym.Env):

    def __init__(self, route: np.ndarray, climber: climber_model.Climber = None, climb_direction: str = CLIMB_DIRECTION_RANDOM[0]):
        self._climber = climber_model.Climber() if climber is None else climber
        self._route = route
        self._route = self._route[self._route[:, 1].argsort()]
        
        start = time.perf_counter()
        self._actions = ClimberEnv._generate_climber_actions(self._route)
        print("Climber actions has been initialized in " + str(time.perf_counter() - start) + " sec")
        self.action_space = gym.spaces.Discrete(len(self._actions))
        print("Action space size: " + str(self.action_space.n))

        start = time.perf_counter()
        self._states = ClimberEnv._generate_climber_states(self._route, self._climber)
        print("Climber states has been initialized in " + str(time.perf_counter() - start) + " sec")
        self.observation_space = gym.spaces.Discrete(len(self._states))
        print("Observation space size: " + str(self.observation_space.n))

        self._start_state = None
        self._finish_hole = None
        self._route_dist = None
        self._random_direction = climb_direction in CLIMB_DIRECTION_RANDOM
        if not self._random_direction:
            self._start_state, self._finish_hole = ClimberEnv._init_start_and_finish(self._states, self._route, climb_direction)
            self._route_dist = self._dist_to_finish()

        self._eposodes_count = 0
        self._eposodes_max = 100
        self._render_surface = None


    def step(self, action):
        self._eposodes_count += 1
        if (self._eposodes_count == self._eposodes_max - 1):
            return (self._get_state_ind(), -120, True, "")

        decoded_action = self._actions[action]
        if decoded_action == CHANGE_SUPPORT_ACTION:
            self._climber.change_support()
            return (self._get_state_ind(), 1, False, "")

        if self._climber.is_transition_possible(limb=decoded_action.limb, point=decoded_action.hole):
            self._climber.do_transition(limb=decoded_action.limb, point=decoded_action.hole)
            done = np.array_equal(self._climber.left_hand_pos, self._finish_hole) \
                or np.array_equal(self._climber.right_hand_pos, self._finish_hole) \
                or np.array_equal(self._climber.left_leg_pos, self._finish_hole) \
                or np.array_equal(self._climber.right_leg_pos, self._finish_hole)
            if done:
                reward = 200
            else:
                reward = self._route_dist - self._dist_to_finish()

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
        

    @staticmethod
    def _generate_climber_states(route: np.ndarray, climber: climber_model.Climber) -> dict:
        states = {}
        max_distance = climber.hands_len + climber.torso_len + climber.legs_len
        max_distance += max_distance // 2
        quads = utils.quad_points_dist_set2(route, max_distance, axis=1)
        counter = 0
        for quad in quads:
            limb_options = utils.sequence_without_repetition_4(quad.points)
            for option in limb_options:
                if climber.can_start(left_hand=option[0], right_hand=option[1], left_leg=option[2], right_leg=option[3]):
                    state1 = ClimberState(
                        left_hand=option[0],
                        right_hand=option[1],
                        left_leg=option[2],
                        right_leg=option[3],
                        support=climber_model.RIGHT_HAND_LEFT_LEG
                    )
                    states[state1] = counter
                    counter += 1
                    state2 = state1.with_support(climber_model.LEFT_HAND_RIGH_LEG)
                    states[state2] = counter
                    counter += 1
        return states


    @staticmethod
    def _generate_climber_actions(route: np.ndarray) -> list:
        actions = []
        actions.append(CHANGE_SUPPORT_ACTION)
        for hole in route:
            for limb in (climber_model.LEFT_HAND, climber_model.RIGHT_HAND, climber_model.LEFT_LEG, climber_model.RIGHT_LEG):
                actions.append(ClimberAction(limb, hole))
        return actions


    @staticmethod
    def _init_start_and_finish(states: dict, route: np.ndarray, climb_direction: str):
        states_list = list(states.keys())
        start_state_ind = None
        finish_hole_ind = None
        if climb_direction in CLIMB_DIRECTION_FROM_BOTTOM_TO_TOP:
            start_state_ind = np.array([min(st.left_leg[1], st.right_leg[1]) for st in states_list]).argmin()
            finish_hole_ind = route[:, 1].argmax()
        
        elif climb_direction in CLIMB_DIRECTION_FROM_TOP_TO_BOTTOM:
            start_state_ind = np.array([max(st.left_hand[1], st.right_hand[1]) for st in states_list]).argmax()
            finish_hole_ind = route[:, 1].argmin()

        elif climb_direction in CLIMB_DIRECTION_FROM_LEFT_TO_RIGHT:
            start_state_ind = np.array([min(st.left_hand[0], st.left_leg[0]) for st in states_list]).argmin()
            finish_hole_ind = route[:, 0].argmax()

        elif climb_direction in CLIMB_DIRECTION_FROM_RIGHT_TO_LEFT:
            start_state_ind = np.array([max(st.right_hand[0], st.right_leg[0]) for st in states_list]).argmax()
            finish_hole_ind = route[:, 0].argmin()

        else:
            raise Exception("Unexpected climbing direction: " + str(climb_direction))

        return (states_list[start_state_ind], route[finish_hole_ind])


    def _get_state_ind(self):
        return self._states[ClimberState(
            left_hand=self._climber.left_hand_pos,
            right_hand=self._climber.right_hand_pos,
            left_leg=self._climber.left_leg_pos,
            right_leg=self._climber.right_leg_pos,
            support=self._climber.support
        )]
    

    def _dist_to_finish(self):
        avg_pos = (self._climber.left_hand_pos + self._climber.right_hand_pos + self._climber.left_leg_pos + self._climber.right_leg_pos) / 4
        return np.linalg.norm(self._finish_hole - avg_pos)



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