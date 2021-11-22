#!/usr/bin/env python3
import numpy as np
from numpy.linalg.linalg import norm
import pytransform3d.visualizer as pv
from pytransform3d import rotations as pr
from math import pi,sin

l = 100
## Creating angular velocity
angular_velocity = np.zeros((l,1))
X = np.linspace(0,2*pi,l)
for i,x in enumerate(X):
    angular_velocity[i] = sin(x)/6
time = np.linspace(0,99,l)

def theta(t):
    if(t==0):
        return 0
    return theta(t-1)+angular_velocity[t]*(time[t]-time[t-1])

def animation_callback(t, n_frames, frame):
    angle = theta(t)
    print("Step",t,"n_Frames",n_frames,"Angle",angle)
    R = pr.matrix_from_angle(0, angle)
    A2B = np.eye(4)
    A2B[:3, :3] = R
    frame.set_data(A2B)
    return frame

fig = pv.figure(width=500, height=500)
frame = fig.plot_basis(R=np.eye(3), s=0.5)  
fig.view_init()

n_frames = 100
if "__file__" in globals():
    fig.animate(
        animation_callback, n_frames, fargs=(n_frames, frame), loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")