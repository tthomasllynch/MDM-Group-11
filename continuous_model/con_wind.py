import numpy as np

# Common wind conditions and gravity. Building properties come from section files.
V_REF = 50       # Wind speed at H_REF (m/s).
H_REF = 10      # Reference height (m).
Z0 = 1.6        # Roughness length (m).
RHO = 1.225     # Air density (kg/m^3).
GF = 0.887      # Assumed gust factor.
G = 9.81        # Gravity (m/s^2).


def wind_speed(z):
    z_safe = np.maximum(z, Z0)
    return V_REF * np.log(z_safe/Z0) / np.log(H_REF/Z0)


def Cp(z, H):
    """Assumed pressure coefficient profile, common to every shape."""
    return np.where(z <= 0.75*H,
                    0.40 + 0.40*z/(0.75*H),
                    0.80 - 0.40*(z-0.75*H)/(0.25*H))


def pressure(z, H):
    """Effective wind pressure (Pa)."""
    return 0.5 * RHO * wind_speed(z)**2 * GF * Cp(z, H)


def q(z, D, H):
    """Lateral force per metre of height for wind along x (N/m)."""
    return pressure(z, H) * D


def P(z, MU, G, H):
    """Axial compression from weight above z (N)."""
    return MU * G * (H-z)


def M(CSA, I, B, P, c_tensile):
    """Approximate first-cracking moment magnitude (N m)."""
    return (c_tensile + P/CSA) * I / (B/2)


def U4d_calc(U1d, U2d, P, q, EI, MU):
    return (q + MU*G*U1d - P*U2d)/EI


def equations(z, y, section):
    axial_force = P(z, section.MU, G, section.H)
    wind_load = q(z, section.D, section.H)
    U4d = U4d_calc(y[1], y[2], axial_force, wind_load,
                   section.bending_stiffness(), section.MU)
    return np.vstack((y[1], y[2], y[3], U4d))


def boundary_conditions(foundation, roof):
    """Fixed base, free roof."""
    return np.array([foundation[0], foundation[1], roof[2], roof[3]])


