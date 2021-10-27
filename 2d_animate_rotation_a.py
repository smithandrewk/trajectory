#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform3d.plot_utils import Frame
from pytransform3d import rotations as pr
from math import pi,sin

l = 100
## Creating angular velocity
angular_velocity = np.zeros((l,1))
X = np.linspace(0,2*pi,l)
for i,x in enumerate(X):
    angular_velocity[i] = sin(x)
time = np.linspace(0,99,l)

def theta(t):
    if(t==0):
        return 0
    return theta(t-1)+angular_velocity[t]*(time[t]-time[t-1])

def update_frame(t, n_frames, frame):
    angle = theta(t)
    print("Step",t,"n_Frames",n_frames,"Angle",angle)
    R = pr.matrix_from_angle(0, angle)
    A2B = np.eye(4)
    A2B[:3, :3] = R
    frame.set_data(A2B)
    return frame


if __name__ == "__main__":
    n_frames = 100

    fig = plt.figure(figsize=(5, 5))
    
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    frame = Frame(np.eye(4), label="rotating frame", s=0.5)
    frame.add_frame(ax)

    anim = animation.FuncAnimation(
        fig, update_frame, n_frames, fargs=(n_frames, frame), interval=50,
        blit=False)

    plt.show()