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
class Basis:
    def __init__(self,i=[1,0,0],j=[0,1,0],k=[0,0,1]):
        self.i = i
        self.j = j
        self.k = k

def get_rotation_quaternion_from_angular_velocity(angular_velocity):
    if(np.array_equiv(angular_velocity,np.array([0,0,0]))):
        # no rotation
        return [1,0,0,0]
    angle = np.linalg.norm(angular_velocity)
    axis = angular_velocity/angle
    return get_rotation_quaternion_from_axis_angle(angle,axis)


def is_unit_quaternion(quaternion):
    return np.linalg.norm(quaternion)==1

def rotate_basis_by_quaternion(quaternion,basis=np.array([[1,0,0],[0,1,0],[0,0,1]])):
    rotated_basis = []
    for v in basis:
        rotated_basis.append(rotate_vector_by_quaternion(v,quaternion=q)[1:])
    return np.array(rotated_basis).T