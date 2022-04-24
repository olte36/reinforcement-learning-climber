from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import numpy as np


LEFT_HAND  = 0
RIGHT_HAND = 1
LEFT_LEG   = 2
RIGHT_LEG  = 3

LEFT_HAND_RIGH_LEG = 4
RIGHT_HAND_LEFT_LEG = 5


class Climber:
   
    def __init__(self, hands_len = 60, torso_len = 50, legs_len = 70, neck_len = 20):
        dim = 2
        self._head_pos = np.zeros(dim)

        self._left_hand_pos = np.zeros(dim)
        self._left_elbow_pos = np.zeros(dim)

        self._right_hand_pos = np.zeros(dim)
        self._right_elbow_pos = np.zeros(dim)

        self._shoulders_pos = np.zeros(dim)
        self._pelvis_pos = np.zeros(dim)

        self._left_leg_pos = np.zeros(dim)
        self._left_knee_pos = np.zeros(dim)

        self._right_leg_pos = np.zeros(dim)
        self._right_knee_pos = np.zeros(dim)

        self._support = LEFT_HAND_RIGH_LEG
        self._hands_len = hands_len
        self._torso_len = torso_len
        self._legs_len = legs_len
        self._neck_len = neck_len
        self._neck_torso_ratio = - (self._torso_len + self._neck_len) / self._neck_len


    def can_start(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray) -> bool:
        return self._is_position_possible(left_hand=left_hand, right_hand=right_hand, left_leg=left_leg, right_leg=right_leg)


    def set_start_pos(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray, support: int):
        self._left_hand_pos = left_hand
        self._right_hand_pos = right_hand
        self._left_leg_pos = left_leg
        self._right_leg_pos = right_leg
        self._support = support


    def is_transition_possible(self, limb: int, point: np.ndarray) -> bool:
        if limb == LEFT_HAND:
            # TODO: реализовать отрыв неопорной правой ноги от зацепки
            return self._support != LEFT_HAND_RIGH_LEG \
                and not np.array_equal(point, self._left_hand_pos) \
                and self._is_position_possible(left_hand=point, right_hand=self._right_hand_pos, left_leg=self._left_leg_pos, right_leg=self._right_leg_pos)
        
        if limb == RIGHT_HAND:
            # TODO: реализовать отрыв неопорной левой ноги от зацепки
            return self._support != RIGHT_HAND_LEFT_LEG \
                and not np.array_equal(point, self._right_hand_pos) \
                and self._is_position_possible(left_hand=self._left_hand_pos, right_hand=point, left_leg=self._left_leg_pos, right_leg=self._right_leg_pos)

        if limb == LEFT_LEG:
            # TODO: реализовать отрыв неопорной правой руки от зацепки
            return self._support != RIGHT_HAND_LEFT_LEG \
                and not np.array_equal(point, self._left_leg_pos) \
                and self._is_position_possible(left_hand=self._left_hand_pos, right_hand=self._right_hand_pos, left_leg=point, right_leg=self._right_leg_pos)

        if limb == RIGHT_LEG:
            # TODO: реализовать отрыв неопорной левой руки от зацепки
            return self._support != LEFT_HAND_RIGH_LEG \
                and not np.array_equal(point, self._right_leg_pos) \
                and self._is_position_possible(left_hand=self._left_hand_pos, right_hand=self._right_hand_pos, left_leg=self._left_leg_pos, right_leg=point)

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


    def change_support(self) -> int:
        if self._support == LEFT_HAND_RIGH_LEG:
            self._support = RIGHT_HAND_LEFT_LEG
        else:
            self._support = LEFT_HAND_RIGH_LEG
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

        # head
        head_pos = []
        for i in range(self._shoulders_pos.size):
            el = (self._pelvis_pos[i] + self._neck_torso_ratio * self._shoulders_pos[i]) / (1 + self._neck_torso_ratio)
            head_pos.append(el)
        self._head_pos = np.array(head_pos)
        # logs
        # print("Вычисление позиции тела")
        # print("Тело: " + str(np.linalg.norm(self._shoulders_pos - self._pelvis_pos)))
        # print("Левая рука: " + str(np.linalg.norm(self._left_hand_pos - self._shoulders_pos)))
        # print("Правая рука: " + str(np.linalg.norm(self._right_hand_pos - self._shoulders_pos)))
        # print("Левая нога: " + str(np.linalg.norm(self._left_leg_pos - self._pelvis_pos)))
        # print("Правая нога: " + str(np.linalg.norm(self._right_leg_pos - self._pelvis_pos)))


    def _is_position_possible(self, left_hand: np.ndarray, right_hand: np.ndarray, left_leg: np.ndarray, right_leg: np.ndarray) -> bool:
        return not np.array_equal(left_hand, right_hand) \
            and not np.array_equal(left_leg, right_leg) \
            and min(left_hand[1], right_hand[1]) > max(left_leg[1], right_leg[1]) \
            and np.linalg.norm(left_hand - right_hand) <= self._hands_len * 2 \
            and np.linalg.norm(left_leg - right_leg) <= self._legs_len * 2 \
            and np.linalg.norm(left_hand - right_leg) <= self._legs_len + self._torso_len + self._hands_len \
            and np.linalg.norm(right_hand - left_leg) <= self._legs_len + self._torso_len + self._hands_len \
            and np.linalg.norm(left_hand - left_leg) <= self._legs_len + self._torso_len + self._hands_len \
            and np.linalg.norm(right_hand - right_leg) <= self._legs_len + self._torso_len + self._hands_len


    @property
    def head_pos(self) -> np.ndarray:
        return self._head_pos

    @property
    def left_hand_pos(self) -> np.ndarray:
        return self._left_hand_pos

    @property    
    def left_elbow_pos(self) -> np.ndarray:   
        return self._left_elbow_pos

    @property
    def right_hand_pos(self) -> np.ndarray:
        return self._right_hand_pos

    @property
    def right_elbow_pos(self) -> np.ndarray:
        return self._right_elbow_pos

    @property
    def shoulders_pos(self) -> np.ndarray:
        return self._shoulders_pos

    @property
    def pelvis_pos(self) -> np.ndarray:    
        return self._pelvis_pos

    @property
    def left_leg_pos(self) -> np.ndarray:
        return self._left_leg_pos
    
    @property
    def left_knee_pos(self) -> np.ndarray:
        return self._left_knee_pos

    @property
    def right_leg_pos(self) -> np.ndarray:
        return self._right_leg_pos

    @property
    def right_knee_pos(self) -> np.ndarray:
        return self._right_knee_pos

    @property
    def support(self) -> int:
        return self._support
    
    @property
    def hands_len(self) -> int:
        return self._hands_len

    @property
    def torso_len(self) -> int:
        return self._torso_len

    @property
    def legs_len(self) -> int:
        return self._legs_len