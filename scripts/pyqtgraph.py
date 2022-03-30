#! /usr/bin/env python3
from PyQt5 import QtWidgets, QtCore
from scripts.scripts.pyqtgraph import PlotWidget, plot
import scripts.scripts.pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint
import numpy as np
import pandas as pd
import fileinput
from lib.utils import process_line

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setBackground('w')
        self.graphWidget.addLegend()
        self.f = fileinput.input()
        self.x = list(np.zeros(100))
        # self.y = np.zeros(100)
        # self.z = np.zeros(100)
        self.t = list(np.linspace(0,99,100))
        self.xline = self.graphWidget.plot(self.t, self.x,pen=pg.mkPen(color=(255, 0, 0)),name='x')
        # self.yline = self.graphWidget.plot(self.t, self.y,pen=pg.mkPen(color=(0, 255, 0)),name='y')
        # self.zline = self.graphWidget.plot(self.t, self.z,pen=pg.mkPen(color=(0, 0, 255)),name='z')

        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

    def update_plot_data(self):
        t,omega,acc = process_line(self.f.readline(),device='phone')
        if(acc is None):
            return
        # print(acc[0])
        self.t = self.t[1:]  # Remove the first y element.
        self.t.append(self.t[-1] + 1)  # Add a new value 1 higher than the last.

        self.x = self.x[1:]  # Remove the first
        self.x.append(acc[0])  # Add a new random value.

        self.xline.setData(self.t, self.x)  # Update the data.

app = QtWidgets.QApplication(sys.argv)

w = MainWindow()
w.show()
sys.exit(app.exec_())