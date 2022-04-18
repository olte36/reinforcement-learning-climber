import enum
from math import sqrt
import pygame
import sys
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import numpy as np


LEFT_HAND  = 0
RIGHT_HAND = 1
LEFT_LEG   = 2
RIGHT_LEG  = 3

LEFT_HAND_RIGH_LEG = 4
RIGHT_HAND_LEFT_LEG = 5

#class SupportPos(enum.Enum):

# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


class Climber:
    
    _left_hand_pos = None
    _left_elbow_pos = None

    _right_hand_pos = None
    _right_elbow_pos = None

    _shoulders_pos = None
    _pelvis_pos = None

    _left_leg_pos = None
    _left_knee_pos = None

    _right_leg_pos = None
    _right_knee_pos = None

    _support = None
    _hands_len = 60
    _torso_len = 50
    _legs_len = 70


    def __init__(self, hands_len = 60, torso_len = 50, legs_len = 70):
        self._hands_len = hands_len
        self._torso_len = torso_len
        self._legs_len = legs_len


    def can_start(self, left_hand: int, right_hand: int, left_leg: int, right_leg: int):
        # return not np.array_equal(left_hand, left_leg) \
        #     and not np.array_equal(left_hand, right_leg) \
        #     and not np.array_equal(right_hand, left_leg) \
        #     and not np.array_equal(right_hand, right_leg) \
        return np.linalg.norm(left_hand - right_hand) <= self._hands_len * 2 \
            and np.linalg.norm(left_leg - right_leg) <= self._legs_len * 2 \
            and np.linalg.norm(left_hand - right_leg) <= self._legs_len + self._torso_len + self._hands_len \
            and np.linalg.norm(left_leg - right_hand) <= self._legs_len + self._torso_len + self._hands_len


    def set_start_pos(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, support: int):
        self._left_hand_pos = left_hand
        self._right_hand_pos = right_hand
        self._left_leg_pos = left_leg
        self._right_leg_pos = right_leg
        self._support = support
        #self._adjust_body()


    def is_transition_possible(self, limb: int, point: np.ndarray):
        if limb == LEFT_HAND:
            # TODO: реализовать отрыв неопорной правой ноги от зацепки
            return self._support != LEFT_HAND_RIGH_LEG \
                and not np.array_equal(point, self._left_hand_pos) \
                and np.linalg.norm(point - self._right_hand_pos) <= self._hands_len * 2 \
                and np.linalg.norm(point - self._left_leg_pos) <= self._legs_len + self._torso_len + self._hands_len \
                and np.linalg.norm(point - self._right_leg_pos) <= self._legs_len + self._torso_len + self._hands_len 
        
        if limb == RIGHT_HAND:
            # TODO: реализовать отрыв неопорной левой ноги от зацепки
            return self._support != RIGHT_HAND_LEFT_LEG \
                and not np.array_equal(point, self._right_hand_pos) \
                and np.linalg.norm(point - self._left_hand_pos) <= self._hands_len * 2 \
                and np.linalg.norm(point - self._right_leg_pos) <= self._legs_len + self._torso_len + self._hands_len \
                and np.linalg.norm(point - self._left_leg_pos) <= self._legs_len + self._torso_len + self._hands_len

        if limb == LEFT_LEG:
            # TODO: реализовать отрыв неопорной правой руки от зацепки
            return self._support != RIGHT_HAND_LEFT_LEG \
                and not np.array_equal(point, self._left_leg_pos) \
                and point[1] < min(self._left_hand_pos[1], self._right_hand_pos[1]) \
                and np.linalg.norm(point - self._right_leg_pos) <= self._legs_len * 2 \
                and np.linalg.norm(point - self._left_hand_pos) <= self._legs_len + self._torso_len + self._hands_len \
                and np.linalg.norm(point - self._right_hand_pos) <= self._legs_len + self._torso_len + self._hands_len

        if limb == RIGHT_LEG:
            # TODO: реализовать отрыв неопорной левой руки от зацепки
            return self._support != LEFT_HAND_RIGH_LEG \
                and not np.array_equal(point, self._right_leg_pos) \
                and point[1] < min(self._left_hand_pos[1], self._right_hand_pos[1]) \
                and np.linalg.norm(point - self._left_leg_pos) <= self._legs_len * 2 \
                and np.linalg.norm(point - self._right_hand_pos) <= self._legs_len + self._torso_len + self._hands_len \
                and np.linalg.norm(point - self._left_hand_pos) <= self._legs_len + self._torso_len + self._hands_len

        raise Exception("Ivalid limb " + limb)


    def do_transition(self, limb: int, point: np.ndarray):
        if limb == LEFT_HAND:
            self._left_hand_pos = point
            #print(np.linalg.norm(self._left_hand_pos - self._right_hand_pos))

        elif limb == RIGHT_HAND:
            self._right_hand_pos = point
            #print(np.linalg.norm(self._right_hand_pos - self._right_hand_pos))

        elif limb == LEFT_LEG:
            self._left_leg_pos = point
            #print(np.linalg.norm(self._left_leg_pos - self._right_leg_pos))
        
        elif limb == RIGHT_LEG:
            self._right_leg_pos = point
            #print(np.linalg.norm(self._right_leg_pos - self._right_leg_pos))
        
        else:
            raise Exception("Ivalid limb " + limb)

        #self._adjust_body()


    def change_support(self):
        if self._support == LEFT_HAND_RIGH_LEG:
            self._support = RIGHT_HAND_LEFT_LEG

        else:
            self._support = LEFT_HAND_RIGH_LEG

        #self._adjust_body()
        return self._support


    def adjust_body(self):
        support_leg = self._left_leg_pos
        support_hand = self._right_hand_pos
        minor_leg = self._right_leg_pos
        minor_hand = self._left_hand_pos
        if self._support == LEFT_HAND_RIGH_LEG:
            support_leg = self._right_leg_pos
            support_hand = self._left_hand_pos
            minor_leg = self._left_leg_pos
            minor_hand = self._right_hand_pos

        # pelvis
        f = lambda x: np.linalg.norm(x - support_leg)
        x0 = support_hand
        res = minimize(f, x0, method='trust-constr', constraints=[
            NonlinearConstraint(lambda x: np.linalg.norm(x - support_leg), 30, self._legs_len),
            NonlinearConstraint(lambda x: np.linalg.norm(x - minor_leg), -np.inf, self._legs_len),
            NonlinearConstraint(lambda x: np.linalg.norm(x - support_hand), -np.inf, self._hands_len + self._torso_len),
            NonlinearConstraint(lambda x: np.linalg.norm(x - minor_hand), -np.inf, self._hands_len + self._torso_len)
        ])
        self._pelvis_pos = np.array(res.x)

        # shoulders
        f = lambda x: -x[1]
        x0 = self._pelvis_pos + np.array((0, 100))
        res = minimize(f, x0, method='trust-constr', constraints=[
            NonlinearConstraint(lambda x: np.linalg.norm(x - minor_hand), -np.inf, self._hands_len),
            NonlinearConstraint(lambda x: np.linalg.norm(x - support_hand), self._hands_len, self._hands_len),
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._pelvis_pos), self._torso_len, self._torso_len)
        ])
        self._shoulders_pos = np.array(res.x)

        # возможно можно свести к линейной оптимизации
        # left elbow
        left_hand_mid = (self._shoulders_pos + self._left_hand_pos) / 2
        f = lambda x: -np.linalg.norm(x - left_hand_mid)
        x0 = left_hand_mid + np.array((0, -10))
        res = minimize(f, x0, method='trust-constr', constraints=[
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._shoulders_pos), self._hands_len // 2, self._hands_len // 2),
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._left_hand_pos), self._hands_len // 2, self._hands_len // 2)
        ])
        self._left_elbow_pos = np.array(res.x)
        
        # right elbow
        right_hand_mid = (self._shoulders_pos + self._right_hand_pos) / 2
        f = lambda x: -np.linalg.norm(x - right_hand_mid)
        x0 = right_hand_mid + np.array((0, -10))
        res = minimize(f, x0, method='trust-constr', constraints=[
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._shoulders_pos), self._hands_len // 2, self._hands_len // 2),
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._right_hand_pos), self._hands_len // 2, self._hands_len // 2)
        ])
        self._right_elbow_pos = np.array(res.x)

        #left knee
        left_leg_mid = (self._pelvis_pos + self._left_leg_pos) / 2
        f = lambda x: -np.linalg.norm(x - left_leg_mid)
        x0 = left_leg_mid + np.array((0, 10))
        res = minimize(f, x0, method='trust-constr', constraints=[
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._pelvis_pos), self._legs_len // 2, self._legs_len // 2),
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._left_leg_pos), self._legs_len // 2, self._legs_len // 2)
        ])
        self._left_knee_pos = np.array(res.x)

        #right knee
        right_leg_mid = (self._pelvis_pos + self._right_leg_pos) / 2
        f = lambda x: -np.linalg.norm(x - right_leg_mid)
        x0 = right_leg_mid + np.array((0, 10))
        res = minimize(f, x0, method='trust-constr', constraints=[
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._pelvis_pos), self._legs_len // 2, self._legs_len // 2),
            NonlinearConstraint(lambda x: np.linalg.norm(x - self._right_leg_pos), self._legs_len // 2, self._legs_len // 2)
        ])
        self._right_knee_pos = np.array(res.x)

        # logs
        # print("Вычисление позиции тела")
        # print("Тело: " + str(np.linalg.norm(self._shoulders_pos - self._pelvis_pos)))
        # print("Левая рука: " + str(np.linalg.norm(self._left_hand_pos - self._shoulders_pos)))
        # print("Правая рука: " + str(np.linalg.norm(self._right_hand_pos - self._shoulders_pos)))
        # print("Левая нога: " + str(np.linalg.norm(self._left_leg_pos - self._pelvis_pos)))
        # print("Правая нога: " + str(np.linalg.norm(self._right_leg_pos - self._pelvis_pos)))

    @property
    def left_hand_pos(self):
        return self._left_hand_pos

    @property    
    def left_elbow_pos(self):   
        return self._left_elbow_pos

    @property
    def right_hand_pos(self):
        return self._right_hand_pos

    @property
    def right_elbow_pos(self):
        return self._right_elbow_pos

    @property
    def shoulders_pos(self):
        return self._shoulders_pos

    @property
    def pelvis_pos(self):    
        return self._pelvis_pos

    @property
    def left_leg_pos(self):
        return self._left_leg_pos
    
    @property
    def left_knee_pos(self):
        return self._left_knee_pos

    @property
    def right_leg_pos(self):
        return self._right_leg_pos

    @property
    def right_knee_pos(self):
        return self._right_knee_pos

    @property
    def support(self):
        return self._support
    
    @property
    def hands_len(self):
        return self._hands_len

    @property
    def torso_len(self):
        return self._torso_len

    @property
    def legs_len(self):
        return self._legs_len

