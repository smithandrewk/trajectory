#!/usr/bin/env python3
from rotation import Rotation
from math import pi
r = Rotation(axis=[0,0,1],angle=pi/2)
print(r)
print(r.rotate_vector([1,0,0]))