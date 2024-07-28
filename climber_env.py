import time
import random
from dataclasses import dataclass
from typing import Optional
from enum import Enum, unique

import gym
import numpy as np
import pygame

import climber_model
import pygame_render
import utils


#CLIMB_DIRECTION_RANDOM = ["random", "r"]
CLIMB_DIRECTION_FROM_BOTTOM_TO_TOP = ["from_bottom_to_top", "bt"]
CLIMB_DIRECTION_FROM_TOP_TO_BOTTOM = ["from_top_to_bottom", "tb"]
CLIMB_DIRECTION_FROM_LEFT_TO_RIGHT = ["from_left_to_right", "lr"]
CLIMB_DIRECTION_FROM_RIGHT_TO_LEFT = ["from_right_to_left", "rl"]


class ClimberEnv(gym.Env):

    def __init__(
        self, 
        route: np.ndarray,
        belay_points: np.ndarray,
        max_transitions: int,
        climber: climber_model.Climber = None,
        climb_direction: str = CLIMB_DIRECTION_FROM_BOTTOM_TO_TOP[0]
    ):
        self._climber = climber_model.Climber() if climber is None else climber
        self._route = route
        self._route = self._route[self._route[:, 1].argsort()]
        
        self._quickdraws = belay_points
        self._quickdraws = self._quickdraws[self._quickdraws[:, 1].argsort()]
        self._clipped = np.full(self._quickdraws.shape[0], False)

        start = time.perf_counter()
        self._actions = ClimberEnv._generate_climber_actions(self._route, self._quickdraws)
        print("Climber actions has been initialized in " + str(time.perf_counter() - start) + " sec")
        self.action_space = gym.spaces.Discrete(len(self._actions))
        print("Action space size: " + str(self.action_space.n))

        start = time.perf_counter()
        self._states = ClimberEnv._generate_climber_states(self._route, self._quickdraws, self._climber)
        print("Climber states has been initialized in " + str(time.perf_counter() - start) + " sec")
        self.observation_space = gym.spaces.Discrete(len(self._states))
        print("Observation space size: " + str(self.observation_space.n))

        self._start_state = None
        self._finish_hole = None
        self._route_dist = None
        #self._random_direction = climb_direction in CLIMB_DIRECTION_RANDOM
        self._random_direction = False
        if not self._random_direction:
            self._start_state, self._finish_hole = ClimberEnv._init_start_and_finish(self._states, self._route, climb_direction)
            self._route_dist = self._dist_to_finish(self._start_state)

        self._max_transitions = max_transitions
        self._count_transitions_done = 0
        self._render_surface = None


    def step(self, action, v=False):
        climber_action = self._actions[action]
        if (self._count_transitions_done == self._max_transitions - 1):
            reward = -10
            done = True
        else:
            #self._count_transitions_done += 1
            #if self.np_random.uniform() <= (self._count_transitions_done / self._max_transitions) * 0.01:
            #    reward = -7
            #    done = True
            if climber_action == CHANGE_SUPPORT_ACTION:
                self._climber.change_support()
                self._count_transitions_done += 1
                reward = 0
                done = False
            elif climber_action.type == PointType.HOLE:
                if self._climber.is_transition_possible(limb=climber_action.limb, point=climber_action.point):
                    self._climber.do_transition(limb=climber_action.limb, point=climber_action.point)
                    self._count_transitions_done += 1
                    done = np.array_equal(self._climber.left_hand_pos, self._finish_hole) \
                        or np.array_equal(self._climber.right_hand_pos, self._finish_hole) \
                        or np.array_equal(self._climber.left_leg_pos, self._finish_hole) \
                        or np.array_equal(self._climber.right_leg_pos, self._finish_hole)

                    if done:
                        reward = 10 if np.all(self._clipped) else -10
                    else:
                        climber_state = ClimberState.make(climber=self._climber, clipped_quickdraws=self._quickdraws[self._clipped])
                        reward = (self._route_dist - self._dist_to_finish(climber_state)) / self._route_dist
                        #reward -= self._count_transitions_done / 100
                        a = self._count_transitions_done / self._max_transitions / 10
                        if v:
                            print("Transition penalty:", a)
                        if climber_state.support == climber_model.LEFT_HAND_RIGH_LEG:
                            support_leg = climber_state.right_leg
                            support_hand = climber_state.left_hand
                        else:
                            support_leg = climber_state.left_leg
                            support_hand = climber_state.right_hand
                        b = (support_hand[2] - support_leg[2]) / (self._climber.hands_len + self._climber.torso_len + self._climber.legs_len) / 10
                        if v:
                            print("Incline penalty:", b)
                else:
                    reward = -5
                    done = False

            elif climber_action.type == PointType.QUICKDRAW:
                quick_draw_ind = np.where(np.all(self._quickdraws == climber_action.point, axis=1))[0][0]

                if (quick_draw_ind == 0 and not self._clipped[quick_draw_ind] or self._clipped[quick_draw_ind - 1] and not self._clipped[quick_draw_ind]) \
                    and self._climber.is_transition_possible(limb=climber_action.limb, point=climber_action.point):
                    self._clipped[quick_draw_ind] = True
                    self._count_transitions_done += 1
                    reward = np.sum(self._clipped) / len(self._quickdraws)
                    done = False
                else:
                    reward = -5
                    done = False
            else:
                raise Exception("Unknown action")
        
        reward -= self._count_transitions_done / self._max_transitions / 10
        climber_state = ClimberState.make(climber=self._climber, clipped_quickdraws=self._quickdraws[self._clipped])
        info = {
            "climber_state": climber_state,
            "performed_action": climber_action,
            "route_dist": self._route_dist,
            "max_transitions": self._max_transitions,
            "transitions_done": self._count_transitions_done,
            "quickdraws": self._quickdraws,
            "clipped_quickdraws": self._quickdraws[self._clipped]
        }
        return self._states[climber_state], reward, done, info
             

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        self._count_transitions_done = 0
        if self._random_direction:
            self._start_state = random.choice(list(self._states.keys()))
            self._finish_hole = random.choice(self._route)
            self._route_dist = self._dist_to_finish(self._start_state)

        self._climber.set_start_pos(
            left_hand=self._start_state.left_hand,
            right_hand=self._start_state.right_hand,
            left_leg=self._start_state.left_leg,
            right_leg=self._start_state.right_leg,
            support=self._start_state.support
        )
        self._clipped = np.full(self._quickdraws.shape[0], False)
        climber_state = ClimberState.make(climber=self._climber, clipped_quickdraws=self._quickdraws[self._clipped])
        if return_info:
            info = {
                "climber_pos": climber_state,
                "route_dist": self._route_dist,
                "max_transitions": self._max_transitions,
                "quickdraws": self._quickdraws
            }
            return self._states[climber_state], info
        else:
            return self._states[climber_state]


    def render(self, mode='human'):
        if self._render_surface is None:
            width = self._route[self._route[:, 0].argsort()[-1]][0] + 30
            height = self._route[self._route[:, 1].argsort()[-1]][1] + 30
            self._render_surface = pygame.display.set_mode((width * pygame_render.SCALE, height * pygame_render.SCALE))

        self._render_surface.fill(pygame_render.WHITE)
        pygame_render.render_route(self._route, self._render_surface)
        pygame_render.render_hole(self._finish_hole, self._render_surface, color=pygame_render.GREEN)
        pygame_render.render_quickdraws(self._quickdraws, self._clipped, self._climber, self._render_surface)
        pygame_render.render_climber(self._climber, self._render_surface)
        pygame.display.update()


    def close(self):
        pygame.quit()
        

    @staticmethod
    def _generate_climber_states(route: np.ndarray, quickdraws: np.ndarray, climber: climber_model.Climber) -> dict:
        states = {}
        max_distance = climber.hands_len + climber.torso_len + climber.legs_len
        max_distance += max_distance // 2
        possible_states = utils.tetrad_points_dist_divide_and_conquer(route, max_distance, axis=1)
        possible_states = possible_states.union(utils.triple_points_dist_divide_and_conquer(route, max_distance, axis=1))
        counter = 0
        for possible_state in possible_states:
            points = possible_state.points
            if len(points) == 3:
                points = points + (None, )

            limb_options = utils.sequence_without_repetition_4(points)
            for option in limb_options:
                option = list(option)
                for i in range(len(option)):
                    option[i] = np.array(option[i]) if option[i] is not None else None

                for support in [climber_model.RIGHT_HAND_LEFT_LEG, climber_model.LEFT_HAND_RIGH_LEG]:
                    if climber.can_start(left_hand=option[0], right_hand=option[1], left_leg=option[2], right_leg=option[3], support=support):
                        for i in range(len(quickdraws) + 1):
                            state = ClimberState(
                                left_hand=option[0],
                                right_hand=option[1],
                                left_leg=option[2],
                                right_leg=option[3],
                                support=support,
                                clipped_quickdraws=quickdraws[:i]
                            )
                            states[state] = counter
                            counter += 1

                #if climber.can_start(left_hand=option[0], right_hand=option[1], left_leg=option[2], right_leg=option[3], support=climber_model.LEFT_HAND_RIGH_LEG):                
                #    state2 = state1.with_support(climber_model.LEFT_HAND_RIGH_LEG)
                #    states[state2] = counter
                #    counter += 1
                    
        return states


    @staticmethod
    def _generate_climber_actions(route: np.ndarray, quickdraws: np.ndarray) -> list:
        actions = []
        actions.append(CHANGE_SUPPORT_ACTION)
        for hole in route:
            for limb in (climber_model.LEFT_HAND, climber_model.RIGHT_HAND, climber_model.LEFT_LEG, climber_model.RIGHT_LEG):
                actions.append(ClimberAction(limb=limb, point=hole, type=PointType.HOLE))
        
        for quickdraw in quickdraws:
            for limb in (climber_model.LEFT_HAND, climber_model.RIGHT_HAND):
                actions.append(ClimberAction(limb=limb, point=quickdraw, type=PointType.QUICKDRAW))

        return actions


    @staticmethod
    def _init_start_and_finish(states: dict, route: np.ndarray, climb_direction: str):
        states_list = list(filter(lambda s: s.left_leg is not None and s.right_leg is not None and len(s.clipped_quickdraws) == 0, states.keys()))
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

    
    def _dist_to_finish(self, climber_state):
        avg = (climber_state.left_hand + climber_state.right_hand + climber_state.left_leg + climber_state.right_leg) / 4
        return np.linalg.norm(self._finish_hole - avg)



@dataclass(frozen=True, init=False)
class ClimberState:

    _left_hand: tuple
    _right_hand: tuple
    _left_leg: tuple
    _right_leg: tuple
    _support: int
    _clipped_quickdraws: tuple


    def __init__(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, support: int, clipped_quickdraws: np.ndarray):
        object.__setattr__(self, "_left_hand", tuple(left_hand) if left_hand is not None else None)
        object.__setattr__(self, "_right_hand", tuple(right_hand) if right_hand is not None else None)
        object.__setattr__(self, "_left_leg", tuple(left_leg) if left_leg is not None else None)
        object.__setattr__(self, "_right_leg", tuple(right_leg) if right_leg is not None else None)
        object.__setattr__(self, "_support", support)
        object.__setattr__(self, "_clipped_quickdraws", tuple([tuple(q) for q in clipped_quickdraws]))


    @staticmethod
    def make(climber: climber_model.Climber, clipped_quickdraws: np.ndarray):
        return ClimberState(
            left_hand=climber.left_hand_pos,
            right_hand=climber.right_hand_pos,
            left_leg=climber.left_leg_pos,
            right_leg=climber.right_leg_pos,
            support=climber.support,
            clipped_quickdraws=clipped_quickdraws
        )

    @property
    def left_hand(self) -> np.ndarray:
        return np.array(self._left_hand) if self._left_hand is not None else None

    @property
    def right_hand(self) -> np.ndarray:
        return np.array(self._right_hand) if self._right_hand is not None else None

    @property
    def left_leg(self) -> np.ndarray:
        return np.array(self._left_leg) if self._left_leg is not None else None

    @property
    def right_leg(self) -> np.ndarray:
        return np.array(self._right_leg) if self._right_leg is not None else None

    @property
    def support(self) -> np.ndarray:
        return self._support

    @property
    def clipped_quickdraws(self) -> np.ndarray:
        return self._clipped_quickdraws


    #def with_support(self, support: int):
    #    return ClimberState(left_hand=self.left_hand, right_hand=self.right_hand, left_leg=self.left_leg, right_leg=self.right_leg, support=support)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(left_hand={self._left_hand},right_hand={self._right_hand},left_leg={self._left_leg},right_leg={self._right_leg},support={self._support},clipped_quickdraws={self._clipped_quickdraws})"


@unique
class PointType(Enum):
    HOLE = 0
    QUICKDRAW = 1


@dataclass
class ClimberAction:

    limb: int
    point: np.ndarray
    type: PointType

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(limb={self.limb},point={self.point},type={self.type})"


CHANGE_SUPPORT_ACTION = ClimberAction(-1, np.empty(1), PointType.HOLE)