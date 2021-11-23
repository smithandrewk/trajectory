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
def get_rotation_matrix_from_yaw_pitch_roll(yaw,pitch,roll):
    ## according to right hand rule,
    # yaw = z rotation
    # pitch = y rotation
    # roll = x rotation
    alpha = yaw
    beta = pitch
    gamma = roll
    R_x = np.array([[1,0,0],[0,cos(gamma),-sin(gamma)],[0,sin(gamma),cos(gamma)]])
    R_y = np.array([[cos(beta),0,sin(beta)],[0,1,0],[-sin(beta),0,cos(beta)]])
    R_z = np.array([[cos(alpha),-sin(alpha),0],[sin(alpha),cos(alpha),0],[0,0,1]])
    R = R_z @ R_y @ R_x
    return R
def get_rotation_quaternion_from_angular_velocity(angular_velocity):
    if(np.array_equiv(angular_velocity,np.array([0,0,0]))):
        # no rotation
        return [1,0,0,0]
    angle = np.linalg.norm(angular_velocity)
    axis = angular_velocity/angle
    return get_rotation_quaternion_from_axis_angle(angle,axis)
def get_rotation_quaternion_from_axis_angle(angle,axis):
    w = cos(angle/2)
    x,y,z = sin(angle/2)*axis
    return np.array([w,x,y,z])
def get_conjugate_quaternion(quaternion):
    ## if q is unit quaternion, q_inv = q_conj
    w,x,y,z = quaternion
    return np.array([w,-x,-y,-z])
def is_unit_quaternion(quaternion):
    return np.linalg.norm(quaternion)==1
def q_mult(q1,q2):
    ## also known as the Hamilton Product
    w = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
    x = q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
    y = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    z = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
    return np.array([w,x,y,z])
def rotate_vector_by_quaternion(vector,quaternion):
    vector = np.array([0,vector[0],vector[1],vector[2]])
    return q_mult(q_mult(quaternion,vector),get_conjugate_quaternion(quaternion))
def rotate_basis_by_quaternion(quaternion,basis=np.array([[1,0,0],[0,1,0],[0,0,1]])):
    rotated_basis = []
    for v in basis:
        rotated_basis.append(rotate_vector_by_quaternion(v,quaternion=q)[1:])
    return np.array(rotated_basis).T