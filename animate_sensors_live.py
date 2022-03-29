#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fileinput
from lib.utils import process_line
from lib.plot_utils import make_3d_axes
from lib.algorithms import algorithm

lim = 1

fig,ax = make_3d_axes(lim=lim)
f = fileinput.input()


def animate(i,fig,ax):
    t,omega,acc = process_line(f.readline(),device='phone')
    if(i==0):
        return
    algorithm(t,omega,acc,fig,ax)




ani = animation.FuncAnimation(fig, animate, fargs=(fig,ax), interval=1)

plt.show()    
