from math import cos,sin
import numpy as np
class Rotation:
    def __init__(self,axis=[0,0,0],angle=0):
        self.axis=np.array(axis)
        self.angle=angle
    def __str__(self):
        return "Axis: "+str(self.axis)+" Angle: "+str(self.angle)
    def rotate_vector(self,v):
        v = np.array(v)
        v_rot = cos(self.angle)*v+sin(self.angle)*np.cross(self.axis,v)+(1-cos(self.angle))*np.dot(self.axis,v)*self.axis
        return v_rot