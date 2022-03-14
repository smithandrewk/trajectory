#!/usr/bin/env python3
from utils.utils import Plotter,process_line
import fileinput
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform3d.plot_utils import Frame
from pytransform3d import rotations as pr
from utils.utils import preprocess_watch_data
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from math import sin,cos
from numpy import random
import sys
from utils.utils import preprocess_watch_data

import numpy as np
import pandas as pd
p = Plotter(400, 400, 3)

Rs = [np.eye(3,3)]
R_tilts = [np.eye(3,3)]
bases = [np.eye(3,3)]
velocities = [np.zeros(3)]
positions = [np.zeros(3)]
lim = 2






def animate(i,xs,ys,last_t,last_ypr):
    # acc_global = R @ acc[i]
    # print(acc[i])
    # acc_global = acc_global + np.array([0,0,1])
    # print(acc_global)
    # velocity = acc_global*delta_t + velocities[i-1]
    # velocities.append(velocity)
    # position = velocity*delta_t + positions[i-1]
    # positions.append(position)
