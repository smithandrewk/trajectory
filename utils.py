import numpy as np
def get_axis_angle(dt,angular_velocity):
    norm = np.linalg.norm(angular_velocity)
    axis = angular_velocity/norm
    angle = dt*norm
    return axis,angle