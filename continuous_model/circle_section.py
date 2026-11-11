import math

# BUILDING DIMENSIONS AND MATERIALS (same for every shape)
H = 500
MU = 300000
floor_area = 28.5**2
# C50/60 mean tensile strength (Pa): Concrete Society, Basic properties.
# https://www.concrete.org.uk/fingertips/basic-properties/
c_tensile = 4.1e6
E_c = 40e9
E_s = 200e9
rho = 0.02
E_RC = E_c*(1-rho) + E_s*rho

# HOLLOW CENTRAL CORE: REINFORCED CONCRETE
core_B = core_D = 9.144
core_t = 0.762
I_core = (core_D*core_B**3 - (core_D-2*core_t)*(core_B-2*core_t)**3)/12
A_core = core_B*core_D - (core_B-2*core_t)*(core_D-2*core_t)

# OUTER WALL AND EMBEDDED REINFORCED-CONCRETE COLUMNS
wall_t = 0.762
column_width = column_depth = 0.75
A_column = column_width*column_depth
n_columns = 24

R = math.sqrt(floor_area/math.pi)
B = D = 2*R
inner_R = R-wall_t
A_outer_wall = math.pi*(R**2-inner_R**2)
I_outer = math.pi*(R**4-inner_R**4)/4
# Square columns follow the tangent, centred within the wall thickness.
column_radius = R-wall_t/2
column_angles = [2*math.pi*i/n_columns for i in range(n_columns)]
columns = [(column_radius*math.cos(angle),column_radius*math.sin(angle))
           for angle in column_angles]

# Bending about y: wind and deflection in x.
# Square column local I is unchanged when rotated in the floor plan.
I_columns_full = sum(
    column_depth*column_width**3/12 + A_column*x**2
    for x, y in columns
)
I_plain_wall = I_outer - I_columns_full
I = I_plain_wall + I_columns_full + I_core
EI = E_c*I_plain_wall + E_RC*(I_columns_full + I_core)
CSA = A_outer_wall + A_core


def bending_stiffness():
    return EI


# Symmetry gives equal stiffness about x and y for these three layouts.
# These aliases support the comparison script without changing the calculation.
I_x = I_y = I
EI_x = EI_y = EI
