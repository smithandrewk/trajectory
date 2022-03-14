#!/usr/bin/env python3
"""
==================
Animate Trajectory
==================

Animates a rotation and translation in 3 dimensions.
"""
print(__doc__)

import sys
from utils.utils import preprocess_watch_data
from utils.utils import get_yaw_pitch_roll
from utils.utils import get_rotation_matrix_from_yaw_pitch_roll,get_tilt_correction_rotation_matrix_from_accelerometer
from utils.utils import get_rotated_basis
import numpy as np

if(len(sys.argv) != 2):
    print("Usage: ./main.py <name of watch file>")
    sys.exit(0)

df = preprocess_watch_data(sys.argv[1],plot=False)
omega = np.array(df[['gyr_x','gyr_y','gyr_z']])
acc = np.array(df[['acc_x','acc_y','acc_z']])
R_tilts = [np.eye(3,3)]
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
    R_tilt = get_tilt_correction_rotation_matrix_from_accelerometer(acc[i])
    R = get_rotation_matrix_from_yaw_pitch_roll(*yaw_pitch_roll)
    Rs.append(R)
    R_tilts.append(R_tilt)
    bases.append(get_rotated_basis(bases[0],R))



from utils.utils import animate_trajectory
animate_trajectory(time,bases,None)