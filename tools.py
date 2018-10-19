import numpy as np

def put_angle_into_range(angle):
    if angle > 2 * np.pi:
        n = np.floor(angle / (2 * np.pi))
        angle = angle - n * 2 * np.pi
    elif angle < 0:
        n = np.floor(-angle / (2 * np.pi))
        angle = angle + n * 2 * np.pi
        angle = angle + 2 * np.pi
    return angle