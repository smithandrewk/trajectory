#!/usr/bin/env python3
last_t = -1
from utils.utils import Plotter,process_line
import fileinput

p = Plotter(400, 400, 3)

for line in fileinput.input():
    if(line.split(sep=",")[0]=="loggingTime(txt)"):
        # header
        continue
    t, acc = process_line(line,device='phone')
    delta_t = t-last_t
    if(delta_t==0):
        continue
    Hz = 1/delta_t
    last_t = t
    print({
        't':t,
        'acc':acc,
        'Hz':Hz
    })
    mag = 100
    p.multiplot([int(acc[0]*mag), int(acc[1]*mag), int(acc[2]*mag)])