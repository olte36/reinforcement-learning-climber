import itertools
import time
import math
import matplotlib.pyplot as plt
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


def triple_points_dist_brute_force(points: np.ndarray, dist: int) -> set:
    quad_set = set()
    rows = points.shape[0]
    for i in range(rows):
        for j in range(i + 1, rows):
            if np.linalg.norm(points[i] - points[j]) <= dist:
                for k in range(j + 1, rows):
                    if np.linalg.norm(points[i] - points[k]) <= dist \
                        and np.linalg.norm(points[j] - points[k]) <= dist:
                        quad_set.add(PointsSet(points[i], points[j], points[k]))

    return quad_set


def tetrad_points_dist_brute_force(points: np.ndarray, dist: int) -> set:
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


def points_dist_divide_and_conquer(points: np.ndarray, dist: int, axis: int, brute_force_func) -> set:
    points_count = points.shape[0]
    if (points_count <= 8 or points[-1][axis] - points[0][axis] <= dist):
        return brute_force_func(points, dist)

    left_part = points[0 : points_count // 2]
    right_part = points[points_count // 2 : points_count]
    
    left_res = points_dist_divide_and_conquer(left_part, dist, axis, brute_force_func)
    right_res = points_dist_divide_and_conquer(right_part, dist, axis, brute_force_func)
    
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

    middle_res = brute_force_func(middle_points, dist)

    return left_res.union(middle_res, right_res)


def tetrad_points_dist_divide_and_conquer(points: np.ndarray, dist: int, axis: int) -> set:
    return points_dist_divide_and_conquer(points, dist, axis, tetrad_points_dist_brute_force)


def triple_points_dist_divide_and_conquer(points: np.ndarray, dist: int, axis: int) -> set:
    return points_dist_divide_and_conquer(points, dist, axis, triple_points_dist_brute_force)


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
    # months = [
    #     "06.20",
    #     "07.20",
    #     "08.20",
    #     "09.20",
    #     "10.20",
    #     "11.20",
    #     "12.20",
    #     "01.21",
    #     "02.21",
    #     "03.21",
    #     "04.21",
    #     "05.21",
    #     "06.21",
    #     "07.21",
    #     "08.21",
    #     "09.21",
    #     "10.21",
    #     "11.21",
    #     "12.21",
    #     "01.22",
    #     "02.22",
    #     "03.22",
    #     "04.22"
    # ]
    # requests = [
    #     24382,
    #     27322,
    #     32262,
    #     45936,
    #     46695,
    #     54782,
    #     41720,
    #     47769,
    #     43736,
    #     52938,
    #     51527,
    #     49653,
    #     43178,
    #     56898,
    #     199391,
    #     73677,
    #     61111,
    #     55852,
    #     61359,
    #     63845,
    #     57952,
    #     61163,
    #     66482
    # ]
    years = [
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2020",
        "2021"
    ]
    papers = [
        743,
        689,
        740,
        692,
        731,
        852,
        1550,
        2820,
        4380,
        6760,
        9320
    ]
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=18)
    plt.xlabel('Год', fontsize=24)
    plt.ylabel('Количество статей', fontsize=24)
    plt.plot(years, papers, c="#000000", linewidth=3)
    plt.show()

    