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

# Vertex on +x; regular hexagon area = 3 sqrt(3) R^2 / 2.
R = math.sqrt(2*floor_area/(3*math.sqrt(3)))
apothem = math.sqrt(3)*R/2
B = 2*R
D = 2*apothem
inner_apothem = apothem-wall_t
inner_R = inner_apothem/math.cos(math.pi/6)
vertices = [(R*math.cos(i*math.pi/3),R*math.sin(i*math.pi/3)) for i in range(6)]
inner_vertices = [(inner_R*math.cos(i*math.pi/3),inner_R*math.sin(i*math.pi/3)) for i in range(6)]
A_outer_wall = 3*math.sqrt(3)*(R**2-inner_R**2)/2
I_outer = 5*math.sqrt(3)*(R**4-inner_R**4)/16
# Four column regions per face, rotated with the face and away from corners.
columns_per_side = n_columns//6
columns = []
column_angles = []
for i in range(6):
    angle = (i+0.5)*math.pi/3
    nx,ny = math.cos(angle),math.sin(angle)
    tx,ty = -ny,nx
    for j in range(columns_per_side):
        along = ((j+0.5)/columns_per_side-0.5)*R
        radial = apothem-wall_t/2
        columns.append((radial*nx+along*tx,radial*ny+along*ty))
        column_angles.append(angle)

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
