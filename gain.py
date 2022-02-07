#!/usr/bin/env python3
import math  # for sine and cosine functions
from matplotlib import pyplot as plt, table
from math import sin
import random
import cv2
import time
import matplotlib
import numpy as np


def process_sensor_data_from_device(line,device):
    """
    loggingTime(txt),
    loggingSample(N),
    accelerometerTimestamp_sinceReboot(s),
    accelerometerAccelerationX(G),
    accelerometerAccelerationY(G),
    accelerometerAccelerationZ(G),
    gyroTimestamp_sinceReboot(s),
    gyroRotationX(rad/s),
    gyroRotationY(rad/s),
    gyroRotationZ(rad/s),
    magnetometerTimestamp_sinceReboot(s),
    magnetometerX(µT),
    magnetometerY(µT),
    magnetometerZ(µT)
    """
    line = line.strip()
    line = line.split(',')
    if(device=="watch"):
        t = float(line[10])
        acc = line[11:14]
    elif(device=="phone"):
        t = float(line[2])
        acc = list(map(float, line[3:6]))
    else:
        t = None
        acc = None
    return t, acc




x = []
acc_x = []
acc_y = []
acc_z = []
last_t = -1


# Plot values in opencv program
class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 0, 0), (0, 250, 0), (0, 0, 250),
                           (0, 255, 250), (250, 0, 250), (250, 250, 0),
                           (200, 100, 200), (100, 200, 200), (200, 200, 100)]
        self.color = []
        self.val = []
        self.plot = np.ones((self.height, self.width, 3))*255

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])

    # Update new values in plot
    def multiplot(self, val, label="plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    # Show plot using opencv imshow
    def show_plot(self, label):
        self.plot = np.ones((self.height, self.width, 3))*255
        # cv2.line(self.plot, (0, int(self.height/2) ), (self.width, int(self.height/2)), (0,255,0), 1)
        for i in range(len(self.val)-1):
            for j in range(len(self.val[0])):
                cv2.line(self.plot, (i, int(self.height/2) - self.val[i][j]), (i+1, int(
                    self.height/2) - self.val[i+1][j]), self.color[j], 1)

        cv2.imshow(label, self.plot)
        cv2.waitKey(10)


p = Plotter(400, 1000, 3)

# f = open('stream', 'r')

import subprocess

def execute(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(p.stdout.readline, ""):
        yield stdout_line 
    p.stdout.close()
    # return_code = popen.wait()
    while p.poll() is None:
        print("Still working...")
        # sleep a while
    if p.poll():
        raise subprocess.CalledProcessError(p.poll(), cmd)
# def truncate(f, n):
#     '''Truncates/pads a float f to n decimal places without rounding'''
#     s = '{}'.format(f)
#     if 'e' in s or 'E' in s:
#         return '{0:.{1}f}'.format(f, n)
#     i, p, d = s.partition('.')
#     return '.'.join([i, (d+'0'*n)[:n]])

for line in execute(["nc", "-l","65432"]):
    if(line.split(sep=",")[0]=="loggingTime(txt)"):
        # header
        continue
    t, acc = process_line(line,device='phone')
    delta_t = t-last_t
    if(delta_t==0):
        continue
    Hz = 1/delta_t
    print({
        't':t,
        'acc':acc
    })
    last_t = t
    p.multiplot([int(acc[0]*200), int(acc[1]*200), int(acc[2]*200)])
