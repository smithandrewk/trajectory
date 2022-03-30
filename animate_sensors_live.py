#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fileinput
from lib.utils import process_line
from lib.plot_utils import make_3d_axes, plot_basis, plot_vector
from lib.algorithms import algorithm
from lib.utils import get_rotation_matrix_from_yaw_pitch_roll
import numpy as np
from lib.utils import get_rotation_matrix_to_rotate_vector_a_to_vector_b
from pytransform3d.rotations import extrinsic_euler_xyz_from_active_matrix
lim = 1

fig,ax = make_3d_axes(lim=lim)
f = fileinput.input()
from numpy import array

def animate(i,fig,ax):
    global last_t,last_ypr

    t,omega,acc = process_line(f.readline(),device='phone')
    if(i<10):
        last_t = t
        last_ypr = [0,0,0]
        return
    dt = t-last_t
    yaw_pitch_roll = array(omega)*dt+last_ypr
    R_global = get_rotation_matrix_from_yaw_pitch_roll(*yaw_pitch_roll)
    # print(t,dt,yaw_pitch_roll,R)
    ax.clear()

    if(np.linalg.norm(acc) < 1.002 or np.linalg.norm(acc) > .998):
        # stationary
        print('stationary')
        R_tilt = get_rotation_matrix_to_rotate_vector_a_to_vector_b(R_global.T,-acc)
        # R_global = R_tilt @ R_global
        # last_ypr = extrinsic_euler_xyz_from_active_matrix(R_global,strict_check=False)
    acc_global = R_global @ acc
    plot_basis(ax=ax,R=R_global)
    plot_vector(ax=ax,v=acc,color=(0,0,0))
    plot_vector(ax=ax,v=acc_global,color=(0,.5,.5))

    ## Update Vals
    last_ypr = yaw_pitch_roll
    last_t = t



ani = animation.FuncAnimation(fig, animate, fargs=(fig,ax), interval=1)

plt.show()    
