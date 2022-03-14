#!/usr/bin/env python3
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fileinput
from utils.utils import Plotter,process_line
import numpy as np

def get_rotation_matrix_to_rotate_vector_a_to_vector_b(a,b=np.array([0,0,-1])):
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    I = np.eye(3,3)
    v_x = np.array([[0,-v[2],v[1]],
                    [v[2],0,-v[0]],
                    [-v[1],v[0],0]])
    R = I + v_x + (v_x @ v_x * (1/(1+c)))
    return R
# Create figure for plotting
fig = plt.figure(figsize=(50, 50))
ax = fig.add_subplot(111, projection="3d")
lim = .75
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
from utils.utils import get_rotated_basis
from pytransform3d.rotations import extrinsic_euler_xyz_from_active_matrix,active_matrix_from_extrinsic_euler_xyz


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

    print("Acc:",acc)
    yaw_pitch_roll = np.array(omega)*delta_t+last_ypr
    last_ypr = yaw_pitch_roll
    # R = active_matrix_from_extrinsic_euler_xyz(yaw_pitch_roll)
    
    R = get_rotation_matrix_from_yaw_pitch_roll(*yaw_pitch_roll)

    if(np.isclose(acc[2],-1,atol=.004)):
        print("static")
        # R_tilt = get_rotation_matrix_to_rotate_vector_a_to_vector_b(acc)
        R_tilt = get_rotation_matrix_to_rotate_vector_a_to_vector_b(R.T[2],-np.array(acc))
        R = R_tilt @ R
        last_ypr = extrinsic_euler_xyz_from_active_matrix(R,strict_check=False)
        # R = R_tilt @ R

    A2B = np.eye(4)
    A2B[:3, :3] = R
    if(i==1):
        last_ypr = np.zeros(3)

    # Draw x and y lists
    ax.clear()
    Frame(A2B, s=0.5).add_frame(ax)
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))
    # colors = ['r','g','b']
    # labels = ['x','y','z']
    # for i,color in enumerate(colors):
    #     ax.quiver(*origin[i],*R[i],color=color,label=labels[i])
    # plt.legend()
    ax.quiver(0,0,0,*acc)



 

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs,ys), interval=1)
plt.show()
