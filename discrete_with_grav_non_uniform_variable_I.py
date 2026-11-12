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


##### SECOND MOMENT OF AREA THAT CHANGES WITH HEIGHT

def I_hollow_square(s):
    """I of a hollow square tube whose width and depth are B*s (s = scale factor)"""
    Bs = B * s
    return (Bs**4 - (Bs - 2*t)**4) / 12

def I_cuboid(z):
    return I_hollow_square(np.ones_like(z))

def I_tapered(z, r=0.5):
    """r = roof size as a fraction of the base"""
    return I_hollow_square(1 - (1 - r) * z / H)

def I_setback(z, scales=(1.0, 0.8, 0.6)):
    """Three layers, steps at H/3 and 2H/3"""
    s = np.where(z < H / 3, scales[0],
        np.where(z < 2 * H / 3, scales[1], scales[2]))
    return I_hollow_square(s)

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

def solve_spring_chain_gravity(n=50, load=q, I_func=None):
    """
    n    : number of segments
    load : function z -> force per metre of height (N/m)
    I_func : function z -> second moment of area (m^4). None = constant I as before
    Returns displacements x (n+1), rotations theta (n), segment forces F (n)
    """
    h = H / n
    m = MU * h
    z_mid = (np.arange(n) + 0.5) * h             # array of mid points

    # One stiffness per spring: spring i takes I from the middle of the segment above it
    if I_func is None:
        k = np.full(n, E * I / h)                # constant I, same as before
    else:
        k = E * I_func(z_mid) / h

    # Spring stiffness matrix
    K = build_K(k)

    # Wind loading (evaluated at segment midpoints)
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

# same wind, but each shape has its own I(z)

shapes = {"Cuboid": I_cuboid, "Tapered": I_tapered, "Setback": I_setback}

# Check: I_cuboid must give the same answer as the original constant-I model
x_check, _, _ = solve_spring_chain_gravity(load=q, I_func=I_cuboid)
print("Constant I matches I_cuboid:", np.allclose(x_nu, x_check))

plt.figure()
for name, I_func in shapes.items():
    x_s, _, _ = solve_spring_chain_gravity(n=100, load=q, I_func=I_func)
    print(f"{name}: roof deflection {x_s[-1]:.3f} m")
    plt.plot(x_s, np.linspace(0, H, len(x_s)), label=f"{name} (roof {x_s[-1]:.2f} m)")
plt.xlabel("Horizontal deflection (m)")
plt.ylabel("Height (m)")
plt.title("Non-uniform wind + gravity, three building shapes")
plt.legend()
plt.grid()
plt.show()
