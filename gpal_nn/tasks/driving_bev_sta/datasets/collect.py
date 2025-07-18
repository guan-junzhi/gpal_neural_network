import numpy as np
from shapely.geometry import LineString, box

def _fix_pts_interpolate(curve, n_points):
    ls = LineString(curve)
    distances = np.linspace(0, ls.length, n_points)
    curve = np.array([ls.interpolate(distance).coords[0] for distance in distances], dtype=np.float32)
    return curve