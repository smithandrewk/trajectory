#!/usr/bin/env python3
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df = pd.read_csv("data/0000.csv")
f_s = 100 # sampling frequency in hertz
o = pd.concat([df["gyr_x"],df["gyr_y"],df["gyr_z"]],axis=1) # angular velocity
ax = plt.axes(projection='3d')
ax.plot3D(*np.array(o).T)
from pytransform3d.rotations import quaternion_integrate

Q = quaternion_integrate(np.array(o))
# Q = Q[:100]

from utils import plot
plot(Q,len(Q))