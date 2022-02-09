#!/usr/bin/env python3
from utils.utils import process_line




x = []
acc_x = []
acc_y = []
acc_z = []
last_t = -1


from utils.utils import Plotter

p = Plotter(400, 1000, 3)



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

