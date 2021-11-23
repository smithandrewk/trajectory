#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from math import sin,cos
from utils.utils import make_cartesian_axes
fig, ax = make_cartesian_axes()
vector = [1,0]
tail = [0,0]
Q = ax.quiver(*tail,*vector,scale=3,color=['r'])



def update_quiver(num, Q):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    print(num)

    num = num/25
    Q.set_UVC(cos(num),sin(num))

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=([Q]),
                               interval=50, blit=False)
fig.tight_layout()
plt.show()