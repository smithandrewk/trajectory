#!/usr/bin/env python3
"""
==================
Animate Romation
==================

Animates a rotation.
"""
print(__doc__)

import sys
from utils.utils import preprocess_watch_data
from utils.utils import get_yaw_pitch_roll
from utils.utils import get_rotation_matrix_from_yaw_pitch_roll
from utils.utils import get_rotated_basis
import numpy as np

if(len(sys.argv) != 2):
    print("Usage: ./main.py <name of watch file>")
    sys.exit(0)

df = preprocess_watch_data(sys.argv[1],plot=False)
omega = np.array(df[['gyr_x','gyr_y','gyr_z']])
time = df['gyr_t']
yaw_pitch_rolls = [np.zeros(3)]
Rs = [np.eye(3,3)]
bases = [np.eye(3,3)]

for i,_ in enumerate(time):
    if (i==0):
        # don't have delta t, next
        continue
    yaw_pitch_roll = get_yaw_pitch_roll(i,omega,time,yaw_pitch_rolls)
    yaw_pitch_rolls.append(yaw_pitch_roll)
    R = get_rotation_matrix_from_yaw_pitch_roll(*yaw_pitch_roll)
    Rs.append(R)
    bases.append(get_rotated_basis(bases[0],R))



trajectory = np.zeros([3,len(time)])
from math import pi,sin
x = np.linspace(0,2*pi,len(time))
x = np.sin(x)
trajectory[2] = x
trajectory = trajectory.T


from animate_trajectory import animate_trajectory
animate_trajectory(time,bases,trajectory)