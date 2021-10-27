#!/usr/bin/env python3
import numpy as np
from numpy.linalg.linalg import norm
import pytransform3d.visualizer as pv
from pytransform3d import rotations as pr
from math import pi,sin

angular_velocity = np.zeros((101,3))
X = np.linspace(0,2*pi,101)
for i,x in enumerate(X):
    angular_velocity[i,0] = sin(x)/5
time = np.linspace(0,100,101,dtype=int)
angular_velocity = angular_velocity[1:,:]
time = time[1:]

# def get_axis_angle(t):
#     t = t-1
#     norm = np.linalg.norm(angular_velocity[t,:])
#     print("Norm",norm,"Vel",angular_velocity[t,:])
#     axis = angular_velocity[t,:]/norm
#     angle = time[t]-time[t-1]*norm
#     return axis,angle

# def animation_callback(t, n_frames, frame):
#     axis,angle = get_axis_angle(t)
#     angle = angle
#     print("Step",t,"Axis",axis,"Angle",angle)
#     R = pr.matrix_from_axis_angle((axis[0],axis[1],axis[2],angle))

#     A2B = np.eye(4)
#     A2B[:3, :3] = R
#     frame.set_data(A2B)
#     return frame

# fig = pv.figure(width=500, height=500)
# frame = fig.plot_basis(R=np.eye(3), s=0.5)
# fig.view_init()

# n_frames = 100
# if "__file__" in globals():
#     fig.animate(
#         animation_callback, n_frames, fargs=(n_frames, frame), loop=True)
#     fig.show()
# else:
#     fig.save_image("__open3d_rendered_image.jpg")