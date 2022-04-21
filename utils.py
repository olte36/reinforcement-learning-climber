import numpy as np
import climber_model
import routes
import sys
import climber_env
import random

def quad_points_dist_set(points: np.ndarray, dist: int):
    quad_set = set()
    rows = points.shape[0]
    for i in range(rows):
        for j in range(i + 1, rows):
            if np.linalg.norm(points[i] - points[j]) > dist:
                break

            for k in range(j + 1, rows):
                if np.linalg.norm(points[i] - points[k]) > dist or np.linalg.norm(points[j] - points[k]) > dist:
                    break
                
                for m in range(k + 1, rows):
                    if np.linalg.norm(points[i] - points[m]) <= dist \
                        and np.linalg.norm(points[j] - points[m]) <= dist \
                        and np.linalg.norm(points[k] - points[m]) <= dist:
                        quad_set.add((tuple(points[i]), tuple(points[j]), tuple(points[k]), tuple(points[m])))
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
        if middle_point[axis] - p[axis] > dist:
            break

        middle_points.append(p.tolist()) 

    for p in right_part[::-1]:
        if p[axis] - middle_point[axis] > dist:
            break

        middle_points.append(p.tolist())
    
    middle_points = np.array(middle_points)
    for i in range(middle_points.shape[1]):
        middle_points = middle_points[middle_points[:, i].argsort()]

    middle_res = quad_points_dist_set(middle_points, dist)

    return left_res.union(middle_res, right_res)


def quad_points_dist_np(points: np.ndarray, dist: int):
    return np.array(list(quad_points_dist_set(points, dist)))


def quad_points_dist_np2(points: np.ndarray, dist: int, axis=1):
    return np.array(list(quad_points_dist_set2(points, dist, axis)))


if __name__ == '__main__':
    points = routes.generate_simple_route(200, 300)
    state1 = climber_env.ClimberState(np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), 2)
    state2 = climber_env.ClimberState(np.array([2, 2]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), 2)
    
    d = {}
    d[state1] = 0
    d[state2] = 1
    states = list(d.keys())
    states 
    l = [min(st.left_leg[1], st.right_leg[1]) for st in states]
    print(np.array(l).argmin())
    sys.exit()

    np.random.shuffle(points)
    for i in range(points.shape[1]):
        points = points[points[:, i].argsort()]

    #print(points)

    res1 = quad_points_dist_np(points, 200)
    res2 = quad_points_dist_np2(points, 200)
    #sys.exit()

    #for i in range(res1.shape[2]):
    #    res1 = res1[res1[:, i].argsort()]

    #for i in range(res2.shape[2]):
    #    res2 = res2[res2[:, i].argsort()]

    print(res1.shape)
    print(res2.shape)


    # for quad in res:
    #     for left_hand in quad:
    #         for right_hand in quad:
    #             for left_leg in quad:
    #                 for right_leg in quad:
    #                     if climer.can_start(left_hand=left_hand, right_hand=right_hand, left_leg=left_leg, right_leg=right_leg):
    #                         print("lh: " + str(left_hand) + " rh: " + str(right_hand) + " ll: " + str(left_leg) + " rl: "+ str(right_leg))