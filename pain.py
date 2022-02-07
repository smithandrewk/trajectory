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


def process_line(line):
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
    # t = line[2]
    # acc = line[3:6]
    t = line[10]
    acc = line[11:14]
    return t, acc


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


# fig = plt.figure(figsize=(10, 20), dpi=100)
# ax = fig.add_subplot(111)
# ax.set_ylim([-6, 6])
# plt.ion()
# plt.show(block=False)
x = []
acc_x = []
acc_y = []
acc_z = []
last_t = -1


# Plot values in opencv program

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