#!/usr/bin/env python3
from utils.utils import Plotter
import cv2
plot = Plotter(plot_width=200,plot_height=200,num_plot_values=1)
import numpy as np
cv2.imshow("yomomma",np.ones((200, 200, 3))*0)

cv2.waitKey(0)