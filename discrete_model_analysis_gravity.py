import numpy as np
import matplotlib.pyplot as plt

######################################################################################################################
#                                                    Method 1                                                        #
######################################################################################################################

def solve_spring_chain(H=10.0, n=10, E=12000.0, I=1.0, w=50.0):
    """
    Method 1: rigid segments joined by rotational springs.

    Parameters
    ----------
    H : total building height (m)
    n : number of rigid segments
    E : Young's modulus (N/m^2)
    I : effective second moment of area (m^4)
    w : uniformly distributed wind load (N/m)

    Returns
    -------
    x     : horizontal deflection at each level, length n+1 (m), x[0] = 0 at ground
    theta : absolute rotation of each segment, length n (rad)
    """
    h = H / n                                # segment height (m)
    k = E * I / h                            # rotational spring stiffness (N m/rad)

    F = np.full(n, w * h)                    # resultant wind force on each segment (N)

    # Rotational stiffness matrix (tridiagonal)
    K = np.zeros((n, n))
    K[0, 0] += k                             # base spring: ground rotation is zero
    for i in range(n - 1):                   # springs between adjacent segments
        K[i, i]         += k
        K[i + 1, i + 1] += k
        K[i, i + 1]     -= k
        K[i + 1, i]     -= k


    # Moment load vector (virtual work)
    above = np.cumsum(F[::-1])[::-1] - F     # sum of F_j for j > i
    f = h * (above + 0.5 * F)

    # Solve K theta = f
    theta = np.linalg.solve(K, f)

    # Horizontal deflection at each level
    x = np.zeros(n + 1)
    x[1:] = np.cumsum(h * theta)

    return x, theta


x, theta = solve_spring_chain()
print(x)
print("Roof displacement:", x[-1])

##### WITH GRAVITY

def solve_spring_chain_gravity(H=10.0, n=10, E=12000.0,
                               I=1.0, w=50.0, mu=10.0, g=9.81):

    # Segment properties
    h = H / n
    m = mu * h
    k = E * I / h

    # Spring stiffness matrix
    K = np.zeros((n, n))
    K[0, 0] += k

    for i in range(n - 1):
        K[i, i] += k
        K[i + 1, i + 1] += k
        K[i, i + 1] -= k
        K[i + 1, i] -= k

    # Wind loading
    F = np.full(n, w * h)
    above = np.cumsum(F[::-1])[::-1] - F
    f = h * (above + 0.5 * F)

    # Gravity geometric stiffness matrix
    G = np.diag(m * g * h * (n - np.arange(n) - 0.5))

    # Effective stiffness
    K_effective = K - G

    # Check stability
    if np.linalg.eigvalsh(K_effective)[0] <= 0:
        raise ValueError("Building is unstable under gravity")

    # Solve for rotations
    theta = np.linalg.solve(K_effective, f)

    # Calculate horizontal displacement
    x = np.zeros(n + 1)
    x[1:] = np.cumsum(h * theta)

    return x, theta


# TEST: Compare wind only with wind + gravity

x_wind, _ = solve_spring_chain(w=0.1)
x_gravity, _ = solve_spring_chain_gravity(w=0.1, mu=1.0)

print("Wind only:", x_wind[-1])
print("Wind + gravity:", x_gravity[-1])


################################# GRAPHS ##############################

# GRAPH 4: Effect of gravity on building deflection

H = 10
n = 10
w = 0.1
mu = 1.0

x_wind, _ = solve_spring_chain(H=H, n=n, w=w)
x_gravity, _ = solve_spring_chain_gravity(H=H, n=n, w=w, mu=mu)

z = np.linspace(0, H, n + 1)

plt.figure()
plt.plot(x_wind, z, label="Wind only")
plt.plot(x_gravity, z, "--", label="Wind + gravity")
plt.xlabel("Horizontal displacement (m)")
plt.ylabel("Height (m)")
plt.title("Effect of gravity on building deflection")
plt.legend()
plt.grid()
plt.show()


# GRAPH 5: Effect of building mass on roof displacement

masses = np.linspace(0, 2, 30)
roof_displacements = []

for mu in masses:
    x, _ = solve_spring_chain_gravity(
        H=10, n=10, E=12000, I=1, w=0.1, mu=mu
    )
    roof_displacements.append(x[-1])

plt.figure()
plt.plot(masses, roof_displacements)
plt.xlabel("Mass per metre of building height (kg/m)")
plt.ylabel("Roof displacement (m)")
plt.title("Effect of building mass on roof displacement")
plt.grid()
plt.show()

################################################### validation ###################################################

# VALIDATION TEST 1: Gravity switched off

x_original, _ = solve_spring_chain(w=0.1)
x_no_gravity, _ = solve_spring_chain_gravity(w=0.1, mu=0)

print("Original model:", x_original[-1])
print("Gravity model with mu=0:", x_no_gravity[-1])

print("Models match:", np.allclose(x_original, x_no_gravity))

# VALIDATION TEST 2: Single-segment analytical solution

H = 10
E = 12000
I = 1
w = 0.1
mu = 1

h = H
m = mu * h
k = E * I / h

# Analytical solution
theta_exact = (w * h**2 / 2) / (k - m * 9.81 * h / 2)
x_exact = h * theta_exact

# Our gravity model
x_model, theta_model = solve_spring_chain_gravity(
    H=H, n=1, E=E, I=I, w=w, mu=mu
)

print("Analytical displacement:", x_exact)
print("Model displacement:", x_model[-1])
print("Models match:", np.isclose(x_exact, x_model[-1]))


# VALIDATION TEST 3: Three-segment gravity matrix

H = 3
n = 3
mu = 2
g = 9.81

h = H / n
m = mu * h

# Gravity matrix used in our model
G_model = np.diag(m * g * h * (n - np.arange(n) - 0.5))

# Expected gravity matrix from our derivation
G_expected = m * g * h * np.diag([2.5, 1.5, 0.5])

print("Gravity matrix:")
print(G_model)

print("Matrices match:", np.allclose(G_model, G_expected))

# VALIDATION TEST 4: Convergence of gravity model

segments = [5, 10, 20, 40, 80, 160]
roof_gravity = []

for N in segments:
    x, _ = solve_spring_chain_gravity(
        H=10, n=N, E=12000, I=1, w=0.1, mu=1
    )
    roof_gravity.append(x[-1])

plt.figure()
plt.plot(segments, roof_gravity, "o-")
plt.xlabel("Number of segments")
plt.ylabel("Roof displacement (m)")
plt.title("Convergence of gravity model")
plt.grid()
plt.show()

print("Roof displacements:", roof_gravity)


######################################################################################################################
#                                                    Method 2                                                        #
######################################################################################################################

def solve_cantilever_beam(n_nodes, EI=1.0, w=10.0, H=None, h=1.0):
    """
        Method 2: discrete cantilever beam.
    
        Parameters
        ----------
        H : total building height (m)
        N : number of rigid segments
        E : Young's modulus (N/m^2)
        I : effective second moment of area (m^4)
        w : uniformly distributed wind load (N/m)
    
        Returns
        -------
        u     : horizontal deflection at each level, length n+1 (m), x[0] = 0 at ground
        u_max : maximum deflection
        i_max : index of max displacement
        theta : absolute rotation of each segment, length n (rad)
        """
    
    N = n_nodes 
    K = np.zeros((N, N))

    if H is not None:
            h = H / (N - 1)

    if N < 5:
        raise ValueError("Need at least 5 nodes.")
    else:

        # Interior rows: standard [1, -4, 6, -4, 1] stencil
        for i in range(2, N - 2):
            K[i, i - 2:i + 3] = [1, -4, 6, -4, 1]

        # Row 1: ghost node u_-1 = u_1 (zero slope at wall), u_0 = 0
        K[1, 1:4] = [6, -4, 1]

        # Row N-2: ghost u_N = 2u_{N-1} - u_{N-2}  (zero moment)
        K[N - 2, N - 4:N] = [1, -4, 5, -2]

        # Row N-1: zero moment + zero shear -> 2 * second difference
        K[N - 1, N - 3:N] = [2, -4, 2]

    rhs = np.full(N, w * h**4 / EI)          # (q a^4 / EI) at every row

    # Remove clamped node 0, solve, then scale
    u = np.zeros(N)
    u[1:] = np.linalg.solve(K[1:, 1:], rhs[1:])

    # Return the unscaled-by-EI/a^3 form too, for reference
    K_matrix = (EI / h**3) * K
    return u, np.max(np.abs(u)), int(np.argmax(np.abs(u)))

u, u_max, i_max = solve_cantilever_beam(n_nodes=10, EI=12000.0, w=50.0, H=10.0)


# Trying to find out angles from displacements
"""sec_theta = np.zeros_like(u)

# Clamped end: zero slope by boundary condition
sec_theta[0] = 0.0


# Interior nodes: central difference
sec_theta[1:-1] = (u[2:] - u[:-2]) / (2 * h)

#print(sec_theta)    
print(u)"""


######################################################################################################################
#                                                    Analytical                                                      #
######################################################################################################################
# use for comparison

def analytical(x, EI, q, H):
    return q * x**2 * (6 * H**2 - 4 * H * x + x**2) / (24 * EI)

#print(analytical(10,12000,50,10))

# LIVS ANALYSIS - Comparing the discrete models

H = 10
E = 12000
I = 1
w = 50
n = 10

# Run both models with the same parameters
x1, theta = solve_spring_chain(H=H, n=n, E=E, I=I, w=w)
x2, _, _ = solve_cantilever_beam(
    n_nodes=n+1, EI=E*I, w=w, H=H
)

# Height coordinates
z = np.linspace(0, H, n+1)

# Exact analytical solution
exact = analytical(z, E*I, w, H)

# GRAPH 1: Deflection against height
plt.figure()
plt.plot(x1, z, label="Spring model")
plt.plot(x2, z, "--", label="Finite difference model")
plt.plot(exact, z, ":", label="Analytical solution")
plt.xlabel("Horizontal displacement (m)")
plt.ylabel("Height (m)")
plt.title("Comparison of building deflection models")
plt.legend()
plt.grid()
plt.show()

# GRAPH 2: Convergence with number of segments
segments = [5, 10, 20, 40, 80]
spring_roof = []
finite_roof = []

for N in segments:
    a, _ = solve_spring_chain(H=H, n=N, E=E, I=I, w=w)
    b, _, _ = solve_cantilever_beam(
        n_nodes=N+1, EI=E*I, w=w, H=H
    )
    spring_roof.append(a[-1])
    finite_roof.append(b[-1])

exact_roof = analytical(H, E*I, w, H)

plt.figure()
plt.plot(segments, spring_roof, "o-", label="Spring model")
plt.plot(segments, finite_roof, "x--", label="Finite difference")
plt.axhline(exact_roof, linestyle=":", color="black",
            label="Analytical solution")
plt.xlabel("Number of segments")
plt.ylabel("Roof displacement (m)")
plt.title("Convergence of discrete models")
plt.legend()
plt.grid()
plt.show()

# GRAPH 3: Percentage error
spring_error = [
    abs(v - exact_roof) / abs(exact_roof) * 100
    for v in spring_roof
]

finite_error = [
    abs(v - exact_roof) / abs(exact_roof) * 100
    for v in finite_roof
]

plt.figure()
plt.plot(segments, spring_error, "o-", label="Spring model")
plt.plot(segments, finite_error, "x--", label="Finite difference")
plt.xlabel("Number of segments")
plt.ylabel("Roof displacement error (%)")
plt.title("Accuracy against analytical solution")
plt.legend()
plt.grid()
plt.show()

print("Analytical roof displacement:", exact_roof)
print("Spring roof displacement:", x1[-1])
print("Finite difference roof displacement:", x2[-1])
