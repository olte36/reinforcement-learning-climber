import time
from typing import Union
from collections import Counter

import numpy as np
import climber_model
import routes
import sys
import climber_env
import random


class PointsSet:

    def __init__(self, *args: Union[tuple, np.ndarray]) -> None:
        self._points = tuple([tuple(p) for p in args])


    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, PointsSet) and Counter(self._points) == Counter(__o._points)


    def __hash__(self) -> int:
        value = 0
        for p in self._points:
            value += hash(p)
        return value


    def __repr__(self) -> str:
        return str(self._points)


    @property
    def points(self) -> tuple:
        return self._points


def quad_points_dist_set(points: np.ndarray, dist: int):
    quad_set = set()
    rows = points.shape[0]
    for i in range(rows):
        for j in range(i + 1, rows):
            if np.linalg.norm(points[i] - points[j]) <= dist:
                for k in range(j + 1, rows):
                    if np.linalg.norm(points[i] - points[k]) <= dist \
                        and np.linalg.norm(points[j] - points[k]) <= dist:
                        for m in range(k + 1, rows):
                            if np.linalg.norm(points[i] - points[m]) <= dist \
                                and np.linalg.norm(points[j] - points[m]) <= dist \
                                and np.linalg.norm(points[k] - points[m]) <= dist:
                                quad_set.add(PointsSet(points[i], points[j], points[k], points[m]))
                                #quad_set.add((tuple(points[i]), tuple(points[j]), tuple(points[k]), tuple(points[m])))
    return quad_set


def quad_points_dist_set2(points: np.ndarray, dist: int, axis: int):
    points_count = points.shape[0]
    if (points_count <= 8):
        return quad_points_dist_set(points, dist)

    left_part = points[0 : points_count // 2]
    right_part = points[points_count // 2 : points_count]
    
    left_res = quad_points_dist_set2(left_part, dist, axis)
    right_res = quad_points_dist_set2(right_part, dist, axis)
    
    middle_point = points[points_count // 2]
    middle_points = []
    for p in left_part[::-1]:
        if np.linalg.norm(middle_point[axis] - p[axis]) > dist:
            break

        middle_points.append(p.tolist())

    for p in right_part:
        if np.linalg.norm(p[axis] - middle_point[axis]) > dist:
            break

        middle_points.append(p.tolist())
    
    middle_points = np.array(middle_points)

    middle_res = quad_points_dist_set(middle_points, dist)

    return left_res.union(middle_res, right_res)


def sequence_without_repetition_4(seq: np.ndarray) -> np.ndarray:
    res = []
    for el1 in seq:
        for el2 in seq:
            if el2 == el1:
                continue
            for el3 in seq:
                if el3 == el1 or el3 == el2:
                    continue
                for el4 in seq:
                    if el4 == el1 or el4 == el2 or el4 == el3:
                        continue
                    res.append([el1, el2, el3, el4])
    return np.array(res)



if __name__ == '__main__':
    #points = routes.generate_random_route(700, 700, 100)
    points = routes.generate_simple_route(250, 500, step=70)
    # points = np.array([
    #     [1, 1], [3, 1], [3, 3], [1, 3], 
    #     [100, 100], [103, 100], [103, 103], [100, 103],
    #     [200, 200], [203, 200], [203, 203], [200, 203],
    #     [300, 300], [303, 300], [303, 303], [300, 303],
    #     [400, 400], [403, 400], [403, 403], [400, 403],
    #     [500, 500], [503, 500], [503, 503], [500, 503]
    # ])
    
    np.random.shuffle(points)
    points = points[points[:, 1].argsort()]

    dist = 100

    start = time.perf_counter()
    res1 = quad_points_dist_set(points, dist)
    print("Calculated res1 in  " + str(time.perf_counter() - start) + " sec")
    
    start = time.perf_counter()
    res2 = quad_points_dist_set2(points, dist, axis=1)
    print("Calculated res2 in  " + str(time.perf_counter() - start) + " sec")
    #sys.exit()

    #for i in range(res1.shape[2]):
    #    res1 = res1[res1[:, i].argsort()]

    #for i in range(res2.shape[2]):
    #    res2 = res2[res2[:, i].argsort()]

    print(len(res1))
    print(len(res2))

    #seq = sequence_without_repetition_4(res1.pop().points)
    #print(seq.shape)
    #print(res1)
    #print(res2)


    # for quad in res:
    #     for left_hand in quad:
    #         for right_hand in quad:
    #             for left_leg in quad:
    #                 for right_leg in quad:
    #                     if climer.can_start(left_hand=left_hand, right_hand=right_hand, left_leg=left_leg, right_leg=right_leg):
    #                         print("lh: " + str(left_hand) + " rh: " + str(right_hand) + " ll: " + str(left_leg) + " rl: "+ str(right_leg))