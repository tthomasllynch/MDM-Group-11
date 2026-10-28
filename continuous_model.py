import numpy as np
from scipy.integrate import solve_bvp

MU = 300000        #Mass per unit height kg/m
G = 9.81        #Gravitational field strength 
H = 0           #Total height
Z = 0           #height measured from base
E = 40e9          #Youngs modulus
I = 0           #Second moment of area 

F = 0           #Constant for now 

# def q(z):
#     """Wind force at height z"""

#     return f 


def P(z):
    """Axial force per unit height"""
    return MU * G * (H-Z)

def U4d(z, U1d, U2d):
    """Fourth derivative"""
    return (F + MU * G * U1d - P(z) * U2d)/(E * I)




print(E)











