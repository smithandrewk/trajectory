from math import cos,sin
import numpy as np
def get_rotation_quaternion_from_axis_angle(angle,axis):
    w = cos(angle/2)
    x,y,z = sin(angle/2)*axis
    return np.array([w,x,y,z])
def get_conjugate_quaternion(quaternion):
    ## if q is unit quaternion, q_inv = q_conj
    w,x,y,z = quaternion
    return np.array([w,-x,-y,-z])
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