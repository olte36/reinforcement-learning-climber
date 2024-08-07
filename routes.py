import random

import numpy as np


def generate_random_route(width, heigth, holes_count, min_distance=120):
    route = []
    zone_height = min_distance // 2
    zones_count = heigth // zone_height
    holes_per_zone = holes_count // zones_count
    for i in range(zones_count):
        for j in range(holes_per_zone):
            x = random.randint(0, width)
            y = random.randint(zone_height * i, zone_height * (i + 1))
            route.append((x, y, 0))
    return np.array(route)


def generate_simple_route(width, heigth, step=50):
    route = []
    c = 0
    for i in range(heigth // step):
        y = step // 2 + i * step
        if y > heigth:
            break
        for j in range(width // step):
            x = step // 2 + j * step
            route.append((x, y, 0))
            c += 5
            if x > width:
                break
    return np.array(route)