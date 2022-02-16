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
from utils.utils import get_yaw_pitch_roll
from utils.utils import get_rotation_matrix_from_yaw_pitch_roll
from utils.utils import get_rotated_basis
from utils.utils import get_tilt_correction_rotation_matrix_from_accelerometer
import numpy as np
import pandas as pd
p = Plotter(400, 400, 3)
last_t = -1

for line in fileinput.input():
    if(line.split(sep=",")[0]=="loggingTime(txt)"):
        # header
        continue
    t, omega,acc = process_line(line,device='phone')
    delta_t = t-last_t
    if(delta_t==0):
        continue
    Hz = 1/delta_t
    last_t = t
    print({
        't':t,
        'omega':omega,
        'acc':acc,
        'Hz':Hz
    })
    mag=100
    p.multiplot([int(acc[0]*mag), int(acc[1]*mag), int(acc[2]*mag)])
plt.show()