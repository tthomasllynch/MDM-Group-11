import numpy as np

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

