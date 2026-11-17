import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "continuous_model"))
import park_avenue as section

#modelling 432 Park Avenue
H = 426           #total weight (m)
MU = 300000       #Mass metre (height) (kg/m)
G = 9.81          #Gravity (ms^-2)

# # Uniform wind load (Kyle's part) 
F = 200_000   #Lateral force per metre (N/m)

#  Assume C50/60 concrete, with mean tensile strength approximately 4.1 MPa. Concrete Society material properties
c_tensile = 4.1e6 #Pa (N/m^2)

EI = section.bending_stiffness()

z = np.linspace(0, H, 500)
y = np.zeros((4, len(z)))

# Non-uniform wind speed, pressure and force for rectangular building
def q(z):
    """Wind force at height z"""
    return F

#hollow rectangle sma

def P(z):
    """Axial force per unit height"""
    return MU * G * (H-z)

def M(z):
    """Bending moment that starts concrete cracking at height z"""
    return (c_tensile + (P(z)/section.CSA))*section.I / (section.B / 2)

def U4d_calc(z, U1d, U2d):
    """Fourth derivative"""
    return (q(z) + MU * G * U1d - P(z) * U2d)/EI

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
curvature = solution.sol(z)[2]

# Total bending moment at each height (N m)
M_total = EI * curvature
cracking_points = M(z)

#Bending moment that causes concrete cracking at that height 

fig, (ax_deflection, ax_moment) = plt.subplots(
    1, 2, figsize=(13, 7), sharey=True, layout='constrained'
)

ax_deflection.plot(d, z, label='Deflection')
ax_deflection.plot(np.zeros(len(z)), z, label='Centreline', alpha=0.5, color='grey')
ax_deflection.set_xlabel('Horizontal deflection (m)')
ax_deflection.set_ylabel('Height (m)')
ax_deflection.set_title('Building deflection')
ax_deflection.set_xlim(-100, 100)
ax_deflection.legend(title=(
    f'Roof deflection: {d[-1]:.3f} m\n'
    f'Midpoint deflection: {d[int(len(d)/2)]:.3f} m'
))

ax_moment.plot(np.abs(M_total) / 1e9, z, label='Total bending moment magnitude')
ax_moment.plot(cracking_points / 1e9, z, '--', label='Approximate cracking threshold')
ax_moment.set_xlabel('Bending moment (GN m)')
ax_moment.set_title('Bending moment and cracking threshold')
ax_moment.legend()
ax_moment.grid(True, alpha=0.3)

plt.show()

















