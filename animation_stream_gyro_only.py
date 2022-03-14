#!/usr/bin/env python3
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fileinput
from utils.utils import Plotter,process_line
import numpy as np
from utils.utils import get_rotation_matrix_from_yaw_pitch_roll
from pytransform3d.rotations import extrinsic_euler_xyz_from_active_matrix,active_matrix_from_extrinsic_euler_xyz
from utils.rotation_utils import get_rotation_matrix_to_rotate_vector_a_to_vector_b
from pytransform3d.plot_utils import Frame
import numpy as np

# Create figure for plotting
fig = plt.figure(figsize=(50, 50))
ax = plt.axes(projection="3d")
lim = 10
ax.set_xlim((-lim, lim))
ax.set_ylim((-lim, lim))
ax.set_zlim((-lim, lim))
last_t = -1
last_ypr = np.zeros(3)
colors = ['r','g','b']
labels = ['x','y','z']
origin = np.zeros((3,3))
last_velocity = np.zeros(3)
last_position = np.zeros(3)

f = fileinput.input()
# This function is called periodically from FuncAnimation
def animate(i):
    global last_t,last_ypr,last_velocity,last_position
    t, omega,acc = process_line(f.readline(),device='watch')
    delta_t = t-last_t  
    if(t==0):
        return
    if(delta_t>1):
        last_t = t
        return
    last_t = t

    yaw_pitch_roll = np.array(omega)*delta_t+last_ypr
    last_ypr = yaw_pitch_roll    
    R = get_rotation_matrix_from_yaw_pitch_roll(*yaw_pitch_roll)
    gravity_estimate = get_rotation_matrix_to_rotate_vector_a_to_vector_b(acc) @ acc
    motion_estimate = gravity_estimate - np.array([0,0,-1])
    if(np.isclose(acc[2],-1,atol=.004) and np.isclose(np.linalg.norm(omega),0,atol=.004)):
        R_tilt = get_rotation_matrix_to_rotate_vector_a_to_vector_b(R.T[2],-np.array(acc))
        R = R_tilt @ R
        last_ypr = extrinsic_euler_xyz_from_active_matrix(R,strict_check=False)
    print("Last v:",last_velocity)
    for i,part in enumerate(motion_estimate):
        if(abs(part)<.004):
            motion_estimate[i] = 0
    print(motion_estimate)
    velocity = motion_estimate*delta_t + last_velocity
    last_velocity = velocity
    position = velocity*delta_t + last_position

    last_position = position
    position = position * 5
    if(np.isclose(np.linalg.norm(acc),1,atol=.004)):
        last_velocity = np.zeros(3)

    # Draw x and y lists
    ax.clear()
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))
    for i,color in enumerate(colors):
        ax.quiver(*position,*R.T[i],color=color,label=labels[i])
    plt.legend()
    # ax.quiver(*position,*acc)
    # ax.quiver(*position,*gravity_estimate,color=(0,0,0))

ani = animation.FuncAnimation(fig, animate, fargs=(), interval=1)
# ani.save('figures/a.mp4', fps=10, bitrate=1000,extra_args=['-vcodec', 'libx264'])
plt.show()    