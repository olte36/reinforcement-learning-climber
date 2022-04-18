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
            route.append((x, y))
    return np.array(route)
