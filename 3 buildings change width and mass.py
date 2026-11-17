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
GF = 1.12         # Gust factor

def wind_speed(z):
    """Log wind profile: v(z) = v_ref * ln(z/z0) / ln(h_ref/z0)"""
    z_safe = np.maximum(z, Z0)
    return V_REF * np.log(z_safe / Z0) / np.log(H_REF / Z0)

def Cp(z):
    """Pressure coefficient: rises to 0.8 at 75% height, falls to 0.4 at the roof"""
    return np.where(
        z <= 0.75 * H,
        0.40 + 0.40 * (z / (0.75 * H))-(-0.7),
        0.80 - 0.40 * ((z - 0.75 * H) / (0.25 * H))-(-0.7)
    )

def cp_tapered(z, H):
    """Cp_w multiplier for tapered buildings, Cp_t(z)."""
    z = np.asarray(z, dtype=float)
    conditions = [
        z <= 0.225 * H,
        z <= 0.475 * H,
        z <= 0.725 * H,
        z <= 0.975 * H,
    ]
    choices = [
        1.03,
        1.03 - 0.03 * ((z - 0.225 * H) / (0.25 * H)),
        1.00 + 0.07 * ((z - 0.475 * H) / (0.25 * H)),
        1.07 - 0.14 * ((z - 0.725 * H) / (0.25 * H)),
    ]
    return np.select(conditions, choices, default=0.93)

def cp_setback(z, H):
    """Cp_w multiplier for setback buildings, Cp_s(z)."""
    z = np.asarray(z, dtype=float)
    conditions = [
        z <= 0.225 * H,
        z <= 0.475 * H,
        z <= 0.725 * H,
        z <= 0.975 * H,
    ]
    choices = [
        0.79,
        0.79 + 0.17 * ((z - 0.225 * H) / (0.25 * H)),
        0.96 - 0.13 * ((z - 0.475 * H) / (0.25 * H)),
        0.83 + 0.42 * ((z - 0.725 * H) / (0.25 * H)),
    ]
    return np.select(conditions, choices, default=1.25)

def pressure(z, type):
    """Wind pressure p(z) in Pa"""
    if type == 'Cuboid':
        return 0.5 * RHO * wind_speed(z)**2 * GF * Cp(z)
    elif type == 'Tapered':
        return 0.5 * RHO * wind_speed(z)**2 * GF * Cp(z) * cp_tapered(z, H)
    elif type == 'Setback':
        return 0.5 * RHO * wind_speed(z)**2 * GF * Cp(z) * cp_setback(z, H)


##### BUILDING WIDTH THAT CHANGES WITH HEIGHT
# One scale factor per shape, used for BOTH the wind width and the stiffness,
# so the two always agree.

def scale_cuboid(z):
    return np.ones_like(z)

def scale_tapered(z, r=0.5):
    """r = roof size as a fraction of the base"""
    return 1 - (1 - r) * z / H

def scale_setback(z, scales=(1.0, 0.8, 0.6)):
    """Three layers, steps at H/3 and 2H/3"""
    return np.where(z < H / 3, scales[0],
           np.where(z < 2 * H / 3, scales[1], scales[2]))

SCALES = {"Cuboid": scale_cuboid, "Tapered": scale_tapered, "Setback": scale_setback}

def width(z, name):
    """Building width at height z (m)"""
    return B * SCALES[name](z)


# Mass per metre scales with the same factor as the width, raised to MASS_EXP:
#   2 -> mass follows floor-plate area (floors, contents, cladding dominate)
#   1 -> mass follows wall perimeter (structure dominates)
#   0 -> constant mass (previous version)
MASS_EXP = 1

def mass_per_metre(z, name):
    """Mass per metre of height (kg/m) at height z"""
    return MU * SCALES[name](z) ** MASS_EXP


def q(z, name):
    """Wind force per metre of height (N/m) = pressure x local building width"""
    return pressure(z, name) * width(z, name)


##### SECOND MOMENT OF AREA THAT CHANGES WITH HEIGHT

def I_hollow_square(s):
    """I of a hollow square tube whose width and depth are B*s (s = scale factor)"""
    Bs = B * s
    return (Bs**4 - (Bs - 2*t)**4) / 12

def I_cuboid(z):
    return I_hollow_square(scale_cuboid(z))

def I_tapered(z):
    return I_hollow_square(scale_tapered(z))

def I_setback(z):
    return I_hollow_square(scale_setback(z))

def build_K(k):
    """Stiffness matrix from one stiffness per spring: k[0] is the base spring"""
    n = len(k)
    K = np.zeros((n, n))
    K[0, 0] += k[0]
    for i in range(n - 1):
        K[i, i]         += k[i + 1]
        K[i + 1, i + 1] += k[i + 1]
        K[i, i + 1]     -= k[i + 1]
        K[i + 1, i]     -= k[i + 1]
    return K


##### WITH GRAVITY

def solve_spring_chain_gravity(n=50, load=q, I_func=None, name='Cuboid'):
    """
    n    : number of segments
    load : function (z, name) -> force per metre of height (N/m)
    I_func : function z -> second moment of area (m^4). None = constant I as before
    Returns displacements x (n+1), rotations theta (n), segment forces F (n)
    """
    h = H / n
    z_mid = (np.arange(n) + 0.5) * h             # array of mid points
    m = mass_per_metre(z_mid, name) * h          # mass of each segment (kg), array of n

    # One stiffness per spring: spring i takes I from the middle of the segment above it
    if I_func is None:
        k = np.full(n, E * I / h)                # constant I, same as before
    else:
        k = E * I_func(z_mid) / h

    # Spring stiffness matrix
    K = build_K(k)

    # Wind loading (evaluated at segment midpoints)
    F = load(z_mid, name) * h
    above = np.cumsum(F[::-1])[::-1] - F
    f = h * (above + 0.5 * F)

    # Gravity geometric stiffness matrix: for rotation i, the weight of everything
    # above segment i plus half of segment i's own weight (same pattern as the wind load).
    # With uniform m this reduces to m*g*h*(n - i - 0.5), as before.
    mass_above = np.cumsum(m[::-1])[::-1] - m
    Gmat = np.diag(g * h * (mass_above + 0.5 * m))
    K_effective = K - Gmat

    # Check stability
    if np.linalg.eigvalsh(K_effective)[0] <= 0:
        raise ValueError("Building is unstable under gravity")

    theta = np.linalg.solve(K_effective, f)

    x = np.zeros(n + 1)
    x[1:] = np.cumsum(h * theta)

    return x, theta, F

###############################################



# same wind, but each shape has its own I(z), width(z) and mass(z)

shapes = {"Cuboid": I_cuboid, "Tapered": I_tapered, "Setback": I_setback}



plt.figure()
for name, I_func in shapes.items():
    x_s, _, _ = solve_spring_chain_gravity(n=100, load=q, I_func=I_func, name=name)
    print(f"{name}: roof deflection {x_s[-1]:.3f} m")
    plt.plot(x_s, np.linspace(0, H, len(x_s)), label=f"{name} (roof {x_s[-1]:.2f} m)")
plt.xlabel("Horizontal deflection (m)")
plt.ylabel("Height (m)")
plt.title("Non-uniform wind + gravity, three building shapes")
plt.legend()
plt.grid()
plt.show()
