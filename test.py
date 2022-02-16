#!/usr/bin/env python3
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fileinput
from utils.utils import Plotter,process_line

# Create figure for plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
lim = 1
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
xs = []
ys = []
from pytransform3d.plot_utils import Frame
import numpy as np
last_t = -1
last_ypr = np.zeros(3)

frame = Frame(np.eye(4), label="rotating frame", s=0.5)
frame.add_frame(ax)
f = fileinput.input()
plt.rcParams['image.cmap'] = 'Paired'
from utils.utils import get_rotation_matrix_from_yaw_pitch_roll
from utils.utils import get_tilt_correction_rotation_matrix_from_accelerometer
from utils.utils import get_rotated_basis

# This function is called periodically from FuncAnimation
def animate(i, xs,ys):
    t, omega,acc = process_line(f.readline(),device='watch')
    if(t==0):
        return
    id_basis = np.eye(3)
    origin = np.zeros((3,3))
    global last_t
    delta_t = t-last_t
    last_t = t
    global last_ypr
    yaw_pitch_roll = np.array(omega)*delta_t+last_ypr
    last_ypr = yaw_pitch_roll
    R = get_rotation_matrix_from_yaw_pitch_roll(*yaw_pitch_roll)
    R_tilt = get_tilt_correction_rotation_matrix_from_accelerometer(acc)
    R = R @ R_tilt
    # R = get_rotated_basis(np.eye(3,3),R)
    # a = R[2]
    # b = acc
    # v = np.cross(a,b)
    # s = np.linalg.norm(v)
    # c = np.dot(a,b)
    # I = np.eye(3,3)
    # v_x = np.array([[0,-v[2],v[1]],
    #                 [v[2],0,-v[0]],
    #                 [-v[1],v[0],0]])
    # R_tilt = I + v_x + (v_x @ v_x * (1/(1+c)))
    # R = R @ R_tilt

    print(omega)

    # # Draw x and y lists
    ax.clear()
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))
    colors = ['r','g','b']
    labels = ['x','y','z']
    for i,color in enumerate(colors):
        ax.quiver(*origin[i],*R[i],color=color,label=labels[i])
    plt.legend()
    ax.quiver(0,0,0,*acc)



 

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs,ys), interval=10)
plt.show()
