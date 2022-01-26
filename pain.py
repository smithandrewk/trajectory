import time
def follow(thefile):
    thefile.seek(0,2) # Go to the end of the file
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1) # Sleep briefly
            continue
        yield line
"""
loggingTime(txt),
loggingSample(N),
accelerometerTimestamp_sinceReboot(s),
accel    erometerAccelerationX(G),
accelerometerAccelerationY(G),
accelerometerAccelerat    ionZ(G),
gyroTimestamp_sinceReboot(s),
gyroRotationX(rad/s),
gyroRotationY(rad/s    ),
gyroRotationZ(rad/s),
magnetometerTimestamp_sinceReboot(s),
magnetometerX(µT)    ,
magnetometerY(µT),
magnetometerZ(µT)
"""
def process_line(line):
    line = line.strip()
    line = line.split(',')
    t = line[2]
    acc = line[3:6]
    return t,acc
from matplotlib import pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([-6, 6])
plt.ion()
plt.show(block=False)

x = []
import numpy as np
import matplotlib
acc_x = []
acc_y = []
acc_z = []

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

f = open('../stream.csv','r')
for line in follow(f):
    if(len(x)==100):
        x.pop()
        acc_x.pop()
        acc_y.pop()
        acc_z.pop()
    t,acc = process_line(line)
    print(t,acc)
    print("Length : ",len(x))
    acc = list(map(float,acc))
    print(acc)
    x.append(float(t))
    acc_x.append(acc[0])
    acc_y.append(acc[1])
    acc_z.append(acc[2])

    plt.plot(x,acc_x,'r',markersize=1,linewidth=.5)
    plt.plot(x,acc_y,'g',markersize=1,linewidth=.5)
    plt.plot(x,acc_z,'b',markersize=1,linewidth=.5)
    plt.pause(0.0001) 
    # mypause(.0001)        
