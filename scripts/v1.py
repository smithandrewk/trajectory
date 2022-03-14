#!/usr/bin/env python3
import math  # for sine and cosine functions
from matplotlib import pyplot as plt, table
from math import sin
import random
import cv2
import time
import matplotlib
import numpy as np


def follow(thefile):
    count = 0
    thefile.seek(0, 2)  # Go to the end of the file
    while True:
        count +=1
        if(count>1000):
            thefile.seek(0, 2)  # Go to the end of the file
            count = 0
        # thefile.seek(0, 2)  # Go to the end of the file
        line = thefile.readline()
        if not line:
            print("noline")
            time.sleep(0.01)  # Sleep briefly
            continue
        yield line






x = []
acc_x = []
acc_y = []
acc_z = []
last_t = -1



from utils.utils import Plotter,process_line

p = Plotter(400, 1000, 3)

f = open('stream', 'r')
for line in follow(f):
    t, acc = process_line(line)
    t = float(t)
    print(acc)
    try:
        acc = list(map(float, acc))
    except:
        print("An exception occurred")

    delta_t = t-last_t
    Hz = 1/delta_t
    # print(t, delta_t, Hz)
    last_t = t
    p.multiplot([int(acc[0]*200), int(acc[1]*200), int(acc[2]*200)])
f.close()