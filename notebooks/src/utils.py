import numpy as np

import math

from src.constraints import ConstraintExtractor


def station_distance_matrix_km(coords_deg):

    coords = np.array(coords_deg, dtype=float)
    n = len(coords)
    dist = np.zeros((n, n))
    R = 6371.0  # Earth mean radius in km
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = math.radians(coords[i, 0]), math.radians(coords[i, 1])
            lat2, lon2 = math.radians(coords[j, 0]), math.radians(coords[j, 1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            )
            d = R * 2 * math.asin(math.sqrt(a))
            dist[i, j] = dist[j, i] = d
    return dist
