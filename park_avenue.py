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

B = D = math.sqrt(floor_area)
inner_B = B-2*wall_t
inner_D = D-2*wall_t
A_outer_wall = B*D-inner_B*inner_D
I_outer = (B*D**3-inner_B*inner_D**3)/12
n_side = n_columns//4 + 1
half_x = B/2-column_width/2
half_y = D/2-column_depth/2
xs = [-half_x+2*half_x*i/(n_side-1) for i in range(n_side)]
ys = [-half_y+2*half_y*i/(n_side-1) for i in range(n_side)]
columns = ([(x,ys[0]) for x in xs]+[(x,ys[-1]) for x in xs]
           +[(xs[0],y) for y in ys[1:-1]]+[(xs[-1],y) for y in ys[1:-1]])

column_angles = [0.0]*len(columns)

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
