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
MU = 300000       #Mass metre (height) (kg/m)
G = 9.81          #Gravity (ms^-2)

# Uniform wind load (Kyle's part) 
F = 100000    #Lateral force per metre (N/m)

I = (B * D**3 - (B - 2*t) * (D - 2*t)**3) / 12

z = np.linspace(0, H, 500)
y = np.zeros((4, len(z)))

# Non-uniform wind speed, pressure and force for rectangular building
def q(z):
    """Wind force at height z"""
# Non-uniform wind load 
V_REF = 8        # Measured wind speed at reference height (changes with different wind conditions) [m/s]
H_REF = 10        # Reference height h1 (m)
Z0 = 1.6          # Roughness length for urban skyscrapers (m)
RHO = 1.225       # Air density (kg/m^3)
GF = 0.887        # Gust factor

def wind_speed(z):
    """Log wind profile: v(h2) = v(h1) * ln(h2/z0) / ln(h1/z0)"""
    z_safe = np.maximum(z, Z0)   # avoids ln(0) at the foundation
    return V_REF * np.log(z_safe / Z0) / np.log(H_REF / Z0) # Wind speed at z

def Cp(z):
    """Pressure coefficient: rises to 0.8 at 75% height, then falls to 0.4 at the roof"""
    return np.where(
        z <= 0.75 * H,
        0.40 + 0.40 * (z / (0.75 * H)),
        0.80 - 0.40 * ((z - 0.75 * H) / (0.25 * H))
    )

def pressure(z):
    """Wind pressure p(z) in Pa"""
    return 0.5 * RHO * wind_speed(z)**2 * GF * Cp(z) # Pressure Formula

def q(z):
    """Wind force per metre of height (N/m) = pressure x building width"""
    return pressure(z) * B 

#hollow rectangle sma

def P(z):
    """Axial force per unit height"""
    return MU * G * (H-z)

def U4d_calc(z, U1d, U2d):
    """Fourth derivative"""
    return (q(z) + MU * G * U1d - P(z) * U2d)/(E * I)

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
plt.xlim(-100, 100)
plt.legend(title = (f'Horizontal deflection at roof is {d[-1]} meters',
           f'Horizontal deflection at Midpoint is {d[int(len(d)/2)]} meters'))
plt.plot(np.zeros(len(z)), z, label = 'centreline', alpha = 0.5, color = 'grey')

plt.show()

















