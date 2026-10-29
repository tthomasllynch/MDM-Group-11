import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

#modelling 432 Park Avenue

H = 426           #total eight (m)
B = 28.5          #Outer width (m)
D = 28.5          #Outer depth (m)
t = 0.5           #wall thickness (m)

#Material props 
E = 40e9          #young modulus (Pa)
MU = 300000       #Mass per unit height (kg/m)
G = 9.81          #Gravity (ms^-2)

# Uniform wind load (Kyle's part) 
F = 100000    #Distributed lateral load (N/m)

z = np.linspace(0, H, 500)
y = np.zeros((4, len(z)))

def q(z):
    """Wind force at height z"""
    return F

#hollow rectangle sma
I = (B * D**3 - (B - 2*t) * (D - 2*t)**3) / 12

def P(z):
    """Axial force per unit height"""
    return MU * G * (H-z)

def U4d_calc(z, U1d, U2d):
    """Fourth derivative"""
    return (F + MU * G * U1d - P(z) * U2d)/(E * I)

def equations(z, y):
    U1d = y[1]
    U2d = y[2]
    U3d = y[3]

    U4d = U4d_calc(z, U1d, U2d)
    return np.vstack((U1d, U2d, U3d, U4d))

def boundary_conditions(foundation, roof):
    return np.array([foundation[0], foundation[1], roof[2], roof[3]])

# solve_bvp calls equations(z, y) internally
# estimates 4 values at the start for foundation and 4 at the end for roof

solution = solve_bvp(
    equations,
    boundary_conditions,
    z,
    y)

d = solution.sol(z)[0]

plt.plot(d, z)
plt.xlabel('Horizontal deflection (m)')
plt.ylabel('Height (m)')
plt.title("Building deflection")
plt.xlim(-200, 200)
plt.legend(title = (f'Horizontal deflection at roof is {d[-1]} meters',
           f'Horizontal deflection at Midpoint is {d[int(len(d)/2)]} meters'))
plt.plot(np.zeros(len(z)), z)

plt.show()

















