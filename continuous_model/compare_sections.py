from scipy.integrate import solve_bvp
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

import con_wind as CM
import circle_section
import park_avenue  
import hexagon_section 


assert park_avenue.H == circle_section.H == hexagon_section.H 
z = np.linspace(0, park_avenue.H, 500)
y = np.zeros((4, len(z)))

EI_HEX = hexagon_section.bending_stiffness()
EI_SQU = park_avenue.bending_stiffness()
EI_CIR = circle_section.bending_stiffness()


def solve_section(section):
    solution = solve_bvp(
        lambda z, y: CM.equations(z, y, section),
        CM.boundary_conditions, z, y.copy(), tol=1e-5, max_nodes=10000
    )
    return solution


# individual section stiffness
solution_hex = solve_section(hexagon_section)
solution_squ = solve_section(park_avenue)
solution_cir = solve_section(circle_section)

BM_total_hex = EI_HEX * solution_hex.sol(z)[2]
BM_total_squ = EI_SQU * solution_squ.sol(z)[2]
BM_total_cir = EI_CIR * solution_cir.sol(z)[2]

# cracking points 
max_BM_hex = CM.M(
hexagon_section.CSA, hexagon_section.I, hexagon_section.B,
CM.P(z, hexagon_section.MU, CM.G, hexagon_section.H), hexagon_section.c_tensile
)

max_BM_circ = CM.M(
circle_section.CSA, circle_section.I, circle_section.B,
CM.P(z, circle_section.MU, CM.G, circle_section.H), circle_section.c_tensile
)

max_BM_squ = CM.M(park_avenue.CSA, park_avenue.I, park_avenue.B,
CM.P(z, park_avenue.MU, CM.G, park_avenue.H), park_avenue.c_tensile
)

# bending moment and cracking threshold for each shape.
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True)

# square
axes[0].plot(np.abs(BM_total_squ) / 1e9, z, label='Bending moment')
axes[0].plot(max_BM_squ / 1e9, z, '--', label='Cracking threshold')
axes[0].set_title('Square')
axes[0].set_xlabel('Bending moment (GN m)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# circle
axes[1].plot(np.abs(BM_total_cir) / 1e9, z, label='Bending moment')
axes[1].plot(max_BM_circ / 1e9, z, '--', label='Cracking threshold')
axes[1].set_title('Circle')
axes[1].set_xlabel('Bending moment (GN m)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Hhxagon
axes[2].plot(np.abs(BM_total_hex) / 1e9, z, label='Bending moment')
axes[2].plot(max_BM_hex / 1e9, z, '--', label='Cracking threshold')
axes[2].set_title('Hexagon')
axes[2].set_xlabel('Bending moment (GN m)')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

axes[0].set_ylabel('Height (m)')
plt.tight_layout()


# Compare different reference wind speeds for each shape.
reference_speeds = [20, 30, 40, 50]  # m/s at CM.H_REF.
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
original_speed = CM.V_REF

fig_speeds, axes_speeds = plt.subplots(
    1, 3, figsize=(12, 7), sharex=True, sharey=True
)

for ax, name, section, cracking in zip(
    axes_speeds,
    ['Square', 'Circle', 'Hexagon'],
    [park_avenue, circle_section, hexagon_section],
    [max_BM_squ, max_BM_circ, max_BM_hex]
):
    for speed, colour in zip(reference_speeds, colours):
        CM.V_REF = speed
        wind_solution = solve_section(section)
        bending = section.bending_stiffness() * wind_solution.sol(z)[2]
        idx = np.argmax(np.abs(bending))
        print(
            f"{name}, reference wind {speed} m/s: "
            f"maximum moment = {abs(bending[idx])/1e9:.3f} GN m "
            f"at height {z[idx]:.2f} m"
        )
        ax.plot(np.abs(bending) / 1e9, z, color=colour,
                label=f'{speed} m/s')

    moments = np.abs(bending)

    # Moments scale with wind speed squared; scale to the first cracking point.
    ratio = np.max(moments / cracking)
    cracking_speed = speed / np.sqrt(ratio)
    print(f"{name}: maximum moment at first cracking = "
          f"{moments.max() / ratio / 1e9:.3f} GN m "
          f"(reference wind {cracking_speed:.2f} m/s)\n")

    ax.plot(cracking / 1e9, z, '--', color='black', label='Cracking threshold')
    ax.set_title(name, fontsize=24, pad=12)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)

CM.V_REF = original_speed
axes_speeds[0].set_ylabel('Height (m)', fontsize=22, labelpad=10)
fig_speeds.supxlabel('Bending moment (GN m)', fontsize=22, y=0.19)
fig_speeds.suptitle(f'Wind speeds at reference height {CM.H_REF} m', fontsize=21, y=0.99)
handles, labels = axes_speeds[0].get_legend_handles_labels()
fig_speeds.legend(handles, labels, loc='lower center',
                  bbox_to_anchor=(0.5, 0.005), ncol=3,
                  fontsize=17, frameon=False, columnspacing=1.8)
fig_speeds.subplots_adjust(left=0.10, right=0.98, bottom=0.30,
                           top=0.86, wspace=0.10)
fig_speeds.savefig(Path(__file__).resolve().parent / 'comparingCS_with_WS.png',
                    dpi=400, bbox_inches='tight', facecolor='white')
plt.show()
