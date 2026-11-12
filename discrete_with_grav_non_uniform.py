import numpy as np
import matplotlib.pyplot as plt

#modelling 432 Park Avenue

H = 426           # total height (m)
B = 28.5          # Outer width (m)
D = 28.5          # Outer depth (m)
t = 0.5           # wall thickness (m)

# Material props
E = 40e9          # Young's modulus (Pa)
MU = 300000       # Mass per metre of height (kg/m)
g = 9.81          # Gravity (m/s^2)

U = 100000    #Lateral force per metre (N/m)

I = (B * D**3 - (B - 2*t) * (D - 2*t)**3) / 12

####### NON-UNIFORM WIND

V_REF = 8         # Wind speed at reference height [m/s]
H_REF = 10        # Reference height (m)
Z0 = 1.6          # Roughness length (m)
RHO = 1.225       # Air density (kg/m^3)
GF = 0.887        # Gust factor

def wind_speed(z):
    """Log wind profile: v(z) = v_ref * ln(z/z0) / ln(h_ref/z0)"""
    z_safe = np.maximum(z, Z0)
    return V_REF * np.log(z_safe / Z0) / np.log(H_REF / Z0)

def Cp(z):
    """Pressure coefficient: rises to 0.8 at 75% height, falls to 0.4 at the roof"""
    return np.where(
        z <= 0.75 * H,
        0.40 + 0.40 * (z / (0.75 * H)),
        0.80 - 0.40 * ((z - 0.75 * H) / (0.25 * H))
    )

def pressure(z):
    """Wind pressure p(z) in Pa"""
    return 0.5 * RHO * wind_speed(z)**2 * GF * Cp(z)

def q(z):
    """Wind force per metre of height (N/m) = pressure x building width"""
    return pressure(z) * B


##### WITH GRAVITY

def solve_spring_chain_gravity(n=50, load=q):
    """
    n    : number of segments
    load : function z -> force per metre of height (N/m)
    Returns displacements x (n+1), rotations theta (n), segment forces F (n)
    """
    h = H / n
    m = MU * h
    k = E * I / h

    # Spring stiffness matrix
    K = np.zeros((n, n))
    K[0, 0] += k
    for i in range(n - 1):
        K[i, i] += k
        K[i + 1, i + 1] += k
        K[i, i + 1] -= k
        K[i + 1, i] -= k

    # Wind loading (evaluated at segment midpoints)
    z_mid = (np.arange(n) + 0.5) * h             # array of mid points 
    F = load(z_mid) * h
    above = np.cumsum(F[::-1])[::-1] - F
    f = h * (above + 0.5 * F)

    # Gravity geometric stiffness matrix
    Gmat = np.diag(m * g * h * (n - np.arange(n) - 0.5))
    K_effective = K - Gmat

    # Check stability
    if np.linalg.eigvalsh(K_effective)[0] <= 0:
        raise ValueError("Building is unstable under gravity")

    theta = np.linalg.solve(K_effective, f)

    x = np.zeros(n + 1)
    x[1:] = np.cumsum(h * theta)

    return x, theta, F

###############################################

# TEST: Compare wind only with wind + gravity

def uniform_load(z):
    return np.full_like(z, w_equiv)

# Non-uniform wind
x_nu, theta, F = solve_spring_chain_gravity(load=q)
print("Max deflection (non-uniform):", np.max(np.abs(x_nu)), "m")

# Uniform comparison with the same total force
F_total = F.sum()
w_equiv = F_total / H
x_u, _, _ = solve_spring_chain_gravity(load=uniform_load)
print("Max deflection (uniform, same resultant):", np.max(np.abs(x_u)), "m")

# np.full_like(z, w_equiv) creates an array with the same shape as z, with every element set to w_equiv


# Plot
z_nodes = np.linspace(0, H, len(x_nu))
plt.plot(x_nu, z_nodes, label="Non-uniform")
plt.plot(x_u, z_nodes, "--", label="Uniform (same total force)")
plt.xlabel("Horizontal deflection (m)")
plt.ylabel("Height (m)")
plt.legend()
plt.show()