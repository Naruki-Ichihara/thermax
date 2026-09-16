"""2-D thermal isolation with fibre orientation as a design field, NO thermal strain.

    left/right edges   free and insulated -- a single standalone support
    top edge           T = T_HOT,  clamped, passive solid band -- fuselage frame
    bottom edge        T = T_COLD, loaded,  passive solid band -- LH2 tank

The rectangle is sized from the outer LH2-tank mounting bracket of Aerospace
2026, 13, 388, Table 8 -- 150 mm wide by 165 mm tall. See the Domain block for
what the two edges correspond to and for why only the aspect ratio reaches the
answer.

The frame is the reaction and sits near ambient; the tank hangs off the bracket
and sits near 20 K, so heat leaks DOWN the bracket into the tank and the load
is applied at the tank interface. T_HOT/T_COLD are 1/0 rather than the physical
temperatures because the conduction energy is quadratic in the temperature
difference and every reported quantity is normalised, so only the ratio of the
two ever reaches the answer.

    design per node    rho, t1, t2 (orientation a_2; a_4 from a closure)
    minimise           heat / heat_ref  +  W_MASS * V
    subject to         C / C_ref <= ENERGY_ALLOW

This is thermal_isolation_2d_aniso.py with the thermal expansion removed
(alpha_par = alpha_perp = 0). What that deletes, and why it was worth deleting:

    eps_el = sym(grad u) - alpha (T - T_ref)   ->   eps_el = sym(grad u)

so the temperature no longer enters the mechanical problem at all. The two
physics are then INDEPENDENT linear problems that share only the design field
(rho, t1, t2). The staggered rho -> T -> u bridge, the cooldown-only drift
measure, and the drift term in the objective all go with it.

Why the coupled version could drop it
-------------------------------------
Measured on the coupled problem, U splits EXACTLY as U = U_mech + U_therm --
the cross term is zero identically, because the cooldown solution carries no
external load and is therefore orthogonal in C to every admissible displacement.
The thermal share of that sum is:

    rho(design)   U_therm / U
       1.00          1.09 %
       0.50          0.04 %
       0.30          0.01 %      <- where C = 10 C_ref, the operating point
       0.15          0.00 %

U_mech grows as material is removed and U_therm shrinks, so the thermal part of
the constraint vanishes along the path the optimiser actually takes. At the
operating point it is 1e-4 of the constraint value. Dropping it changes the
answer by that much.

What is NOT captured, and cannot be recovered here
--------------------------------------------------
The top edge still moves about 5e-3 on cooldown, and that number does not fall
as the structure thins (4.9477e-03 at rho = 1, 4.9604e-03 at rho = 0.08) -- the
displacement of a freely contracting body does not depend on its stiffness. No
strain-energy measure can see it. If the thermal DISPLACEMENT is a requirement,
this file is the wrong formulation; use the coupled one with W_DRIFT > 0.

Consequences of alpha = 0
-------------------------
U reduces exactly to half the compliance, so the constraint is the standard
normalised compliance constraint and the argument for U over int t.u no longer
applies. The OPTIMUM is also invariant to the traction magnitude: U and U_ref
both scale as lambda^2 and the reported stresses both scale as lambda, so every
normalised quantity is unchanged. Sizing the load therefore does not move the
design -- what it buys is that the stresses come out in real MPa, against real
allowables, which is what a stress constraint would need.

Load
----
Table 11 of the same paper gives the peak resultant load on the critical
bracket, per configuration and per certification load case (CS-25.561 /
CS-25.963, a 1100 kg tank):

               9 g fwd   6 g down   3 g up   3 g lat        critical bracket
    A (8)       16.69      13.15     6.58     20.00         1st row
    B (8)       15.30      12.69     6.39      9.76         1st row
    C (16)      11.22       8.87     4.43      5.27         1st row inner/OUTER
    D (12)      12.80       9.94     4.92      6.04         1st row inner/OUTER

kN. Dividing by mass x load factor / bracket count gives peak-to-average ratios
of 1.26 to 2.60, which is the uneven distribution the paper describes; the one
outlier is A under lateral load at 4.94, the case the paper singles out as
unacceptable.

Configuration C is the one the paper recommends, and in C the OUTER bracket
modelled here is critical under the 3 g lateral case at 5.27 kN. A bracket is
two L-joints and this domain is one L-joint web, so

    TRACTION = 5270 N / 2 / (150 mm x 4 mm) = 4.39 MPa

Table 11 gives resultants, not the axial/transverse split (that is only in the
figures), so applying the whole resultant as axial tension is conservative -- and
axial is the load path the paper says to design for, since its debonding
failures were driven by bracket bending.

For scale: at that load the frame-side bond sees 2635 N / 5640 mm^2 = 0.47 MPa
against the FM300 adhesive's 53 MPa normal strength (Table 2), so the intact
bracket is nowhere near strength-limited. That is why stress here is reported
and not constrained.

Material
--------
AS4/PEEK UD LAMINA, from Table 1 of the same paper -- the ply properties, NOT
the [0/45/90/-45]4S laminate the paper's bracket is built from. The lay-up is
what this problem designs: the orientation field (a_2, and a_4 through a
closure) IS the stacking sequence, resolved per node, so feeding it laminate
properties would be assuming the answer.

    E11 = 129 GPa    E22 = 9.4 GPa    G12 = 4.5 GPa    nu12 = 0.32

The paper's own lay-up is still in here, as the reference. The orientation
average at a_2 = I/2 (t1 = t2 = 0) reproduces the classical quasi-isotropic
laminate EXACTLY -- 49.79 GPa and nu = 0.3156 by both routes, 0.00% apart -- so
every "x reference" figure this script prints is "x the [0/45/90/-45]4S bracket
of the paper".

The paper reports no thermal properties -- it is a structural study -- so the
lamina conductivity is built from the constituents:

    k_par  = Vf kf_ax + (1-Vf) km                      (rule of mixtures)
    k_perp = km [kf_tr(1+Vf) + km(1-Vf)]
                / [kf_tr(1-Vf) + km(1+Vf)]             (Rayleigh / Hasselman-
                                                        Johnson, perfect interface)

with AS4 fibre kf_ax = 6.9, kf_tr = 2.6 W/(m K), PEEK km = 0.25 W/(m K) at 23 C
and Vf = 0.61 (APC-2 nominal), giving

    k_par = 4.31 W/(m K)     k_perp = 0.756 W/(m K)     ratio 5.70

The same two formulas on AS4/3501-6, where measured values exist (same fibre,
epoxy matrix, km = 0.20, Vf = 0.63), give 4.42 against a measured 4.9 (-10%) and
0.670 against a measured 0.69 (-3%). Pushing kf_tr to 3.0 moves k_perp to 0.783
and the ratio to 5.50, so the ratio is good to roughly +-5%.

Conductivity is a second-order average and depends on a_2 alone; stiffness is a
fourth-order average and depends on a_2 and a_4. So a fibre lying along an
isotherm delivers E11 to whatever structural duty that direction carries while
presenting only k_perp to the heat.

WHICH WAY THE FIBRES SHOULD POINT, AND WHY IT IS NOT WHAT THE SQUARE-CELL
EXAMPLE FOUND. With the thermal strain gone the orientation field has a two-way
conflict: heat wants the fibres transverse (k_yy = k_perp), y-direction
stiffness wants them axial (E_yy = E11). For a member carrying the y load,
conductance at fixed stiffness goes as k_yy / E_yy, relative to the best:

    a11   fibres      E_yy [GPa]   k_yy [W/(m K)]   k_yy/E_yy
    0.00  || y           129.00         4.307         1.000   <- best
    0.25                  67.75         3.419         1.512
    0.50  a2 = I/2 (QI)   49.79         2.531         1.523
    0.75                  18.94         1.644         2.600
    0.88                   8.46         1.200         4.247   <- worst
    1.00  || x             9.40         0.756         2.409

The comparison is E1/E2 = 13.72 against k_par/k_perp = 5.70. The stiffness is
the MORE anisotropic of the two, so turning the fibres transverse gives up more
modulus than it saves conductance, and the axial state wins by 2.41x over the
transverse one. Every intermediate state is worse than axial too, so the field
should drive to a11 = 0: fibres along the heat path.

That is the opposite of what the original square-cell example concluded, and
the reason is only that it carried a placeholder k_par/k_perp = 20. At that
ratio transverse wins by 1.46x. The real material is on the other side of the
threshold, so the "lay the fibres along an isotherm and get E11 for free" move
does not pay for AS4/PEEK.

Room-temperature conductivity is used across a 20-293 K span, where the true
values fall substantially. The absolute watt figures are therefore indicative;
the ratio, which is what sets the orientation, is the more robust of the two.

Both interpolations are RAMP, with opposite signs of q, because grey is
penalised on whichever side of linear is unfavourable and that flips with the
direction of the objective: stiffness is wanted (q > 0), conductance is not
(q < 0).

Passive bands
-------------
feax applies a surface traction to the boundary faces regardless of density, so
on a fully loaded edge the traction also lands on void, where E_MIN makes the
compliance integral meaningless. The bands make the load land on material; they
also model the shell walls a skirt is welded between.

Outputs in outputs/thermal_isolation_2d_aniso_uncoupled_C<..>_m<W_MASS>/.
"""
from __future__ import annotations

import collections
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as onp
import nlopt

import feax as fe
import feax.gene as gene
from feax.mechanics.orientation import principal_direction
from feax.utils import save_sol

import thermax as tx

# ── Domain ─────────────────────────────────────────────────────────────────
# Bracket envelope from Gyftos, Sioutis & Lampeas, "Investigation of Liquid
# Hydrogen Tank Structural Integration Concepts for Regional Aircraft",
# Aerospace 2026, 13, 388, Table 8 (structural integrator geometric details):
#
#     bracket width           150 mm      (both inner and outer)
#     outer bracket height    165 mm      (the inner one is 115 mm)
#     bracket thickness         4 mm
#     bond area on frame per L-joint   5572 / 5640 mm^2
#
# Each bracket there is a pair of L-joints bonded between a fuselage frame and
# the LH2 tank. 5572 mm^2 / 150 mm = 37.1 mm, so the 150 mm is the extruded
# width of the L and ~37 mm is the bonded foot. The design domain used here is
# the FACE of that plate, viewed along the 4 mm laminate thickness, so the
# rectangle is (bracket width) x (bracket height) and the bottom and top edges
# are the two bonded interfaces.
#
# Only the ASPECT RATIO reaches the answer. Every reported quantity is
# normalised by the solid-block reference, the material constants are
# non-dimensional, and with alpha = 0 the result is invariant to the traction
# magnitude, so LX = 150, LY = 165 gives exactly what LX = 1, LY = 1.1 gives.
# The millimetres are here to keep the model traceable to the paper.
# The OUTER bracket is the one modelled: the paper reports the longer brackets
# as the higher-stress ones (increased bending), and it is also the longer heat
# path of the two.
LX = 150.0                         # mm, bracket width
LY = 115.0                         # mm, inner bracket height
ELEM_SIZE = 1.0                    # mm, target element edge -> 100 x 110
NX = int(round(LX / ELEM_SIZE))
NY = int(round(LY / ELEM_SIZE))
# Bonded band at each interface, kept at the 5% of the height the square cell
# used. The paper's ~37 mm foot is the other in-plane direction, so it does not
# set this; raise it if the adhesive doubler should be modelled as deeper.
PASSIVE_THICKNESS = 0.05 * LY      # mm, non-design solid band on each edge

# ── Material ───────────────────────────────────────────────────────────────
# AS4/PEEK UD lamina, Aerospace 2026, 13, 388, Table 1 (their source is
# Gillespie et al., ref [33]). Units are mm, N, MPa throughout, consistent with
# the millimetre geometry. These are PLY properties: the orientation field is
# the lay-up, so the paper's [0/45/90/-45]4S is the REFERENCE state (a2 = I/2),
# not the input. That equivalence is exact -- see the header.
E1, E2, G12, NU12 = 129000.0, 9400.0, 4500.0, 0.32
# NOT from the paper, which reports no thermal properties. Built from the
# constituents (rule of mixtures / Rayleigh, AS4 + PEEK at Vf = 0.61) and
# validated on measured AS4/3501-6 to within 3% transverse, 10% axial -- see
# the header. W/(mm K), i.e. W/(m K) / 1000, for the mm/N/MPa/K/W set.
# k_par/k_perp = 5.70 against E1/E2 = 13.72, so AXIAL fibres win.
K_PAR, K_PERP = 4.307e-3, 7.560e-4
# Relative floors: both RAMP weights are called with a solid value of 1.0, so
# these are fractions of the full property, not absolute moduli.
E_MIN, K_MIN = 1e-6, 1e-4
RAMP_Q_E = 4.0                     # stiffness, q > 0
RAMP_Q = -0.8                      # conductivity, q < 0
SGN_BETA = 10.0

# Real temperatures: the fuselage frame sits near ambient, the tank at LH2
# boiling point. Only the difference reaches the answer -- the conduction energy
# is quadratic in it -- but with real k it is what puts the heat leak in watts.
T_HOT, T_COLD = 293.0, 20.0        # K, frame and LH2 tank
# Out-of-plane laminate thickness, Table 8. The 2-D energy integral is per unit
# thickness, so this converts it to a real heat rate:
#     E [W K / mm] * THICKNESS [mm] = E_tot [W K],  Q = 2 E_tot / dT  [W]
# because a two-terminal conduction energy is E_tot = 1/2 G dT^2 and Q = G dT.
# Checked against the 1-D value for the solid reference plate: k_yy = 2.53e-3,
# G = k A / L = 9.21e-3 W/K, Q = 2.513 W.
THICKNESS = 4.0                    # mm
HEAT_TO_WATT = 2.0 * THICKNESS / (T_HOT - T_COLD)
# No thermal expansion: this is what makes the two physics independent.
ALPHA_PAR = ALPHA_PERP = 0.0
# Peak bracket resultant load, Table 11: Configuration C (the paper's
# recommended one), 3 g lateral, 1st row OUTER -- the case in which the outer
# bracket modelled here is the critical one. A bracket is two L-joints and this
# domain is one L-joint web, so the load per web is half. See the header for the
# full table and for why the resultant is applied as pure axial tension.
#
# The OPTIMUM does not depend on this number (U and U_ref both scale as its
# square), but the reported stresses do, and they come out in real MPa.
BRACKET_LOAD = 5270.0              # N, Table 11, config C / 3 g lateral / outer
L_JOINTS_PER_BRACKET = 2
TRACTION = BRACKET_LOAD / L_JOINTS_PER_BRACKET / (LX * THICKNESS)   # MPa
# (name, traction on the top edge, weight in the compliance term). Only cases
# with a positive weight are built and solved.
LOAD_CASES = (
    # The traction is applied to the BOTTOM edge (tank interface) and the TOP
    # edge (frame) is the clamp. feax's surface map is added to the residual, so
    # a POSITIVE component acts in the NEGATIVE coordinate direction: on this
    # edge (0, +T) pulls the tank interface DOWN, away from the clamp, which is
    # tension. Verified: (0, +1) gives mean uy(bottom) = -2.20e+01 and (0, -1)
    # gives +2.20e+01.
    #
    # For pure linear elasticity the sign changes nothing -- C, U and the
    # optimum are identical (both signs give U = 1.649426e+03 on the reference
    # block) -- and with the thermal strain gone that is the whole story.
    ("tension", (0.0, TRACTION), 1.0),
    ("shear",   (TRACTION, 0.0), 0.0),
)

# ── Objective and constraint ───────────────────────────────────────────────
ENERGY_ALLOW = 0.5     # C / C_ref must stay below this
W_MASS = 2                      # weight on volume, against heat
# Mass is in the objective rather than capped by a constraint: a cap does not
# minimise it, the design simply grows to meet it. And it has to be priced,
# because heat alone does not price it once orientation is a design field. Heat
# is monotone increasing in density AT FIXED ORIENTATION, but the design has
# another move -- turn the fibres transverse, which cuts k_yy by k_par/k_perp
# and E_yy by E1/E2, then thicken to put the stiffness back. That lowers heat
# while RAISING volume, and nothing in a heat-only objective stands in its way.
#
# V is the design-region volume fraction, so both terms sit around 0.05-0.10 at
# the operating point and W_MASS reads directly as their relative importance.

# ── Optimisation ───────────────────────────────────────────────────────────
N_FIELDS = 3                       # rho, t1, t2  (a_4 comes from a closure)
FILTER_RADIUS = 2.0 * LX / NX
FILTER_RADIUS_ORI = 2.0 * LX / NX
# Heaviside continuation. The projection is what drives the design to 0/1 -- the
# RAMP pair alone was measured to leave it at 68% non-discrete -- and the ladder
# is what makes it survivable, since a sharp projection cannot undo a large step.
HEAVISIDE_BETAS = (2, 4, 8)
ITER_PER_STAGE = 50
# The continuation stages run a fixed count -- their job is to hand the next
# beta a warm start, not to converge. The FINAL stage instead stops on relative
# change, with ITER_PER_STAGE_FINAL only as a cap.
#
# The test is on a trailing WINDOW, not on consecutive steps. MMA does not
# descend monotonically here -- measured, J bounced between 2.34 and 2.78 over
# ten consecutive evaluations while still drifting downwards -- so any criterion
# of the form "|dJ|/|J| small N times in a row" either never fires or fires on a
# coincidence. The window measures the spread instead:
#
#     (max J - min J) over the last FINAL_WINDOW iterations
#     ----------------------------------------------------  <  FINAL_FTOL_REL
#                     |mean J| over the same
#
# which cannot be satisfied by an oscillation and cannot be satisfied by a slow
# drift either, since both widen the spread.
#
# Feasibility is checked alongside, because nlopt's own ftol does not: a run
# that stalls while still violating the compliance limit would otherwise be
# reported as converged.
ITER_PER_STAGE_FINAL = 100         # hard cap on the final stage
FINAL_WINDOW = 20                  # trailing outer iterations tested
FINAL_FTOL_REL = 1e-3              # relative spread of J over that window
FINAL_FEAS_TOL = 1e-3              # and C/ENERGY_ALLOW - 1 must be <= this
# An x-uniform design is a stationary point of this cell, so a uniform start
# leaves the symmetric branch only through round-off. A perturbation makes that
# transition reproducible; > 0 to use it.
SEED, PERTURB = 0, 0.0

# Snapshots. The design fields are written every SNAPSHOT_EVERY outer
# iterations to snapshots.h5, incrementally, with every constant above
# stored as a root attribute -- so a sweep's files say what produced them
# and an interrupted run still leaves everything up to the last snapshot.
SNAPSHOT_EVERY = 10
PRINT_EVERY = 1                    # iterations between progress lines

# Stress is reported, not constrained: sigma = rho**q C0:eps with 0 < q < PENAL
# (q > 0 relaxes the singularity phenomenon), aggregated by a volume-weighted
# p-norm.
STRESS_P = 12.0
STRESS_RELAX_Q = 2.0

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs",
    f"thermal_isolation_2d_aniso_uncoupled_C{ENERGY_ALLOW:g}_m{W_MASS:g}")
os.makedirs(OUTDIR, exist_ok=True)

TOL = 1e-6 * max(LX, LY)
bottom_edge = lambda p: jnp.isclose(p[1], 0.0, atol=TOL)
top_edge = lambda p: jnp.isclose(p[1], LY, atol=TOL)

mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                              ele_type="QUAD4")
n_nodes = mesh.points.shape[0]
pts = onp.asarray(mesh.points)

passive_mask = jnp.asarray((pts[:, 1] <= PASSIVE_THICKNESS + TOL)
                           | (pts[:, 1] >= LY - PASSIVE_THICKNESS - TOL))
design_mask = ~passive_mask
n_design = int(design_mask.sum())

CASES = [(name, tvec, w) for name, tvec, w in LOAD_CASES if w > 0.0]
W_SUM = sum(w for _n, _v, w in CASES)

print(f"2-D thermal isolation, uncoupled   outer bracket "
      f"{LX:g} x {LY:g} mm (aspect {LX/LY:.3f}), {ELEM_SIZE:g} mm elements")
print(f"  top = fuselage frame (T_HOT, clamped), "
      f"bottom = LH2 tank (T_COLD, loaded)")
print(f"  {NX}x{NY} QUAD4  {n_nodes:,} nodes; "
      f"passive band {PASSIVE_THICKNESS:g} mm at each interface")
print(f"  AS4/PEEK UD lamina E1/E2 = {E1/E2:.2f}, "
      f"k_par/k_perp = {K_PAR/K_PERP:.0f}; "
      f"RAMP q_E = {RAMP_Q_E:g}, q_k = {RAMP_Q:g};  alpha = 0")
print(f"  load {BRACKET_LOAD/1000:g} kN per bracket / {L_JOINTS_PER_BRACKET} "
      f"L-joints / ({LX:g} x {THICKNESS:g} mm) = {TRACTION:.3f} MPa; "
      + ", ".join(f"{n} w={w:g}" for n, _v, w in CASES))
print(f"  design {N_FIELDS} x {n_nodes:,} = {N_FIELDS*n_nodes:,} variables; "
      f"{n_design:,} design nodes, passive {float(passive_mask.mean())*100:.1f}%")
print(f"  minimise heat/reference + {W_MASS:g}*V,  "
      f"C/reference <= {ENERGY_ALLOW:g}")

# ── Problems and solvers ───────────────────────────────────────────────────
thermal = tx.anisotropic_diffusion_problem(
    mesh, k_par=K_PAR, k_perp=K_PERP, k_min=K_MIN, ramp_q=RAMP_Q,
    sgn_beta=SGN_BETA, ele_type="QUAD4")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(top_edge, "all", T_HOT),        # fuselage frame, ambient
    fe.DirichletBCSpec(bottom_edge, "all", T_COLD),    # LH2 tank, cryogenic
]).create_bc(thermal)
ts_th = fe.TracedStructure.from_problem(thermal)

direct = fe.DirectSolverOptions()
_o, _z = jnp.ones((n_nodes,)), jnp.zeros((n_nodes,))
solve_T = fe.create_solver(thermal, bc_th, solver_options=direct,
                           adjoint_solver_options=direct, linear=True,
                           traced_structure=ts_th,
                           traced_params=fe.TracedParams(volume_vars=(_o, _z, _z)))
guess_T = fe.zero_like_initial_guess(thermal, bc_th)
conduction_energy = tx.create_conduction_energy_fn(thermal)

# thermax's anisotropic elasticity takes (rho, t1, t2, T) because its energy
# density carries the thermal strain. With alpha = 0 that term is identically
# zero, so T is passed as a CONSTANT zero field: it is not a function of the
# design, no adjoint runs through it, and psi reduces to
# 1/2 w_E(rho) sym(grad u) : C : sym(grad u).
T_UNUSED = _z

stress_at_quad = tx.anisotropic_stress_measure(
    E1=E1, E2=E2, G12=G12, nu12=NU12, q=STRESS_RELAX_Q,
    alpha_par=ALPHA_PAR, alpha_perp=ALPHA_PERP, t_ref=0.0, sgn_beta=SGN_BETA)
mech, solvers, energies, vonmises_fns = [], [], [], []
bc_me = ts_me = None
for name, tvec, _w in CASES:
    p = tx.anisotropic_elasticity_problem(
        mesh, E1=E1, E2=E2, G12=G12, nu12=NU12, e_min=E_MIN, ramp_q=RAMP_Q_E,
        alpha_par=ALPHA_PAR, alpha_perp=ALPHA_PERP, t_ref=0.0,
        sgn_beta=SGN_BETA, ele_type="QUAD4",
        location_fns=(bottom_edge,), tractions=(tvec,))
    if bc_me is None:
        bc_me = fe.DirichletBCConfig([
            fe.DirichletBCSpec(top_edge, "all", 0.0)]).create_bc(p)
        ts_me = fe.TracedStructure.from_problem(p)
    mech.append(p)
    solvers.append(fe.create_solver(
        p, bc_me, solver_options=direct, adjoint_solver_options=direct,
        linear=True, traced_structure=ts_me,
        traced_params=fe.TracedParams(volume_vars=(_o, _z, _z, _z))))
    energies.append(fe.create_energy_fn(p))
    vonmises_fns.append(tx.create_vonmises_fn(p, stress_at_quad))
guess_u = fe.zero_like_initial_guess(mech[0], bc_me)

filter_rho = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS)
filter_ori = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS_ORI)


# ── Design fields and responses ────────────────────────────────────────────
def process(x, beta):
    """Raw design vector -> physical (rho, t1, t2).

    Density is filtered, Heaviside-projected (``normalize=True``, so 0 -> 0 and
    1 -> 1 exactly and void really is void), then forced to 1 on the passive
    nodes -- ``where`` also kills their gradient, so those variables are inert.
    Orientation is only filtered: projection has no meaning for a field with no
    void state.
    """
    f = tuple(x[i * n_nodes:(i + 1) * n_nodes] for i in range(N_FIELDS))
    rho = gene.heaviside_projection(filter_rho(f[0]), beta=beta, threshold=0.5,
                                    normalize=True)
    rho = jnp.where(passive_mask, 1.0, rho)
    return (rho, filter_ori(f[1]), filter_ori(f[2]))


def heat(fields):
    """Conduction energy of the solved temperature field.

    For a two-terminal problem this is (T_hot - T_cold)/2 times the transmitted
    heat, so minimising it minimises the leak exactly.
    """
    tp = fe.TracedParams(volume_vars=fields)
    return conduction_energy(solve_T(tp, guess_T, traced_structure=ts_th), tp)


def mech_params(fields):
    """The same design fields, padded with the unused temperature slot."""
    return fe.TracedParams(volume_vars=fields + (T_UNUSED,))


def stored_energy(tp_me):
    """Weighted MEAN over load cases of U = 1/2 int eps : C : eps, each
    normalised by its own reference, so the reference design scores 1.0.

    With alpha = 0 this is exactly half the compliance int t.u, so the constraint
    is the standard normalised compliance constraint.
    """
    return sum(w * energies[i](
        solvers[i](tp_me, guess_u, traced_structure=ts_me).flat, tp_me)
        / energy_ref[i] for i, (_n, _v, w) in enumerate(CASES)) / W_SUM


def stress_pnorm(tp_me):
    """Aggregated von Mises over every load case at once: one constraint-sized
    number instead of one per quadrature point, and the pooled maximum is still
    the maximum."""
    sig, wts = [], []
    for i in range(len(CASES)):
        s, jxw = vonmises_fns[i](
            solvers[i](tp_me, guess_u, traced_structure=ts_me), tp_me)
        sig.append(s.ravel())
        wts.append(jxw.ravel())
    return tx.pnorm(jnp.concatenate(sig), jnp.concatenate(wts), P=STRESS_P)


# ── Reference: solid cell, isotropic composite (a2 = I/2, random-planar a4) ─
ref = (_o, _z, _z)
_tp_ref = mech_params(ref)
heat_ref = float(heat(ref))
energy_ref = [float(energies[i](
    solvers[i](_tp_ref, guess_u, traced_structure=ts_me).flat, _tp_ref))
    for i in range(len(CASES))]
stress_ref = float(stress_pnorm(_tp_ref)[1])
heat_ref_W = heat_ref * HEAT_TO_WATT
print(f"  reference: heat={heat_ref_W:.3f} W  "
      + "  ".join(f"U_{n}={c:.4e}" for (n, _v, _w), c in zip(CASES, energy_ref))
      + f"  peak von Mises={stress_ref:.3f} MPa")
print("-" * 78)

_snap = tx.SnapshotWriter(
    os.path.join(OUTDIR, "snapshots.h5"), mesh,
    ("rho", "t1", "t2", "a11", "director"),
    attrs=dict(
        LX=LX, LY=LY, ELEM_SIZE=ELEM_SIZE, THICKNESS=THICKNESS, dim=2, PASSIVE_THICKNESS=PASSIVE_THICKNESS,
        E1=E1, E2=E2, G12=G12, NU12=NU12, K_PAR=K_PAR, K_PERP=K_PERP,
        E_MIN=E_MIN, K_MIN=K_MIN, RAMP_Q_E=RAMP_Q_E, RAMP_Q=RAMP_Q,
        SGN_BETA=SGN_BETA, T_HOT=T_HOT, T_COLD=T_COLD,
        BRACKET_LOAD=BRACKET_LOAD, TRACTION=TRACTION,
        ENERGY_ALLOW=ENERGY_ALLOW, W_MASS=W_MASS,
        FILTER_RADIUS=FILTER_RADIUS, FILTER_RADIUS_ORI=FILTER_RADIUS_ORI,
        HEAVISIDE_BETAS=list(HEAVISIDE_BETAS), ITER_PER_STAGE=ITER_PER_STAGE,
        ITER_PER_STAGE_FINAL=ITER_PER_STAGE_FINAL,
        FINAL_WINDOW=FINAL_WINDOW, FINAL_FTOL_REL=FINAL_FTOL_REL,
        FINAL_FEAS_TOL=FINAL_FEAS_TOL, SNAPSHOT_EVERY=SNAPSHOT_EVERY,
        SEED=SEED, PERTURB=PERTURB, STRESS_P=STRESS_P,
        STRESS_RELAX_Q=STRESS_RELAX_Q,
        heat_ref=heat_ref, heat_ref_W=heat_ref_W, stress_ref=stress_ref,
        energy_ref=list(energy_ref),
        load_cases=[n for n, _v, _w in CASES]))


def snapshot(xx, beta, scalars):
    """One HDF5 row: the PHYSICAL fields, i.e. after filtering and projection,
    which is what a reader wants to plot. The raw design vector is recoverable
    from nothing else, but it is also not what anyone looks at."""
    f = process(jnp.asarray(xx), jnp.float64(beta))
    # a11 and the director travel too: t1/t2 are box coordinates and mean
    # nothing in a viewer, whereas a11 is "how much fibre lies along x" and the
    # director is what a glyph filter draws.
    a2 = jax.vmap(lambda a, b: tx.unit_trace_a2(a, b, SGN_BETA))(f[1], f[2])
    d = principal_direction(a2)
    _snap.append(it[0], [onp.asarray(a) for a in
                         (f[0], f[1], f[2], a2[:, 0, 0], d)], scalars)


@jax.jit
def evaluate(x, beta):
    def _h(z):
        return heat(process(z, beta)) / heat_ref

    def _u(z):
        return stored_energy(mech_params(process(z, beta)))

    h, gh = jax.value_and_grad(_h)(x)
    c, gc = jax.value_and_grad(_u)(x)
    v, gv = jax.value_and_grad(
        lambda z: jnp.sum(process(z, beta)[0] * design_mask) / n_design)(x)
    f = process(x, beta)
    spn, smax = stress_pnorm(mech_params(f))
    a11f = 0.5 * (f[1] + 1.0)
    solid = f[0] * design_mask
    return (h, gh, c, gc, v, gv, spn, smax,
            jnp.sum(a11f * solid) / jnp.maximum(jnp.sum(solid), 1e-12),
            jnp.sum(a11f * passive_mask) / jnp.sum(passive_mask),
            400.0 * jnp.mean(f[0] * (1.0 - f[0])))


# ── MMA ────────────────────────────────────────────────────────────────────
_cache: dict = {}
_beta = [HEAVISIDE_BETAS[0]]


def _cached(xx):
    key = (_beta[0], xx.tobytes())
    if key not in _cache:
        _cache.clear()
        _cache[key] = [onp.asarray(a) for a in
                       evaluate(jnp.asarray(xx), jnp.float64(_beta[0]))]
    return _cache[key]


history = {k: [] for k in ("iteration", "beta", "heat", "compliance", "volume",
                           "objective", "a11_struts", "a11_bands", "mnd",
                           "spn", "smax")}
it = [0]
stage_ends: list = []
# Final-stage convergence state, reset per stage.
_final = [False]
_conv = {"window": collections.deque(maxlen=FINAL_WINDOW), "stopped": None}
_last_x = [None]


def objective(xx, grad):
    h, gh, c, gc, v, gv, spn, smax, a11s, a11b, mnd = _cached(xx)
    if grad.size > 0:
        grad[:] = gh + W_MASS * gv
    it[0] += 1
    j = float(h) + W_MASS * float(v)
    for k, val in (("iteration", it[0]), ("beta", _beta[0]),
                   ("heat", float(h)), ("compliance", float(c)),
                   ("volume", float(v)), ("objective", j),
                   ("a11_struts", float(a11s)), ("a11_bands", float(a11b)),
                   ("mnd", float(mnd)), ("spn", float(spn) / stress_ref),
                   ("smax", float(smax) / stress_ref)):
        history[k].append(val)
    if it[0] % PRINT_EVERY == 0 or it[0] == 1:
        print(f"  it {it[0]:4d} b={_beta[0]:4.1f} heat={float(h):.3e} "
              f"C={float(c):6.2f} V={float(v):.4f} "
              f"a11[str={float(a11s):.2f} bnd={float(a11b):.2f}] "
              f"grey={float(mnd):4.1f}% sMAX={float(smax)/stress_ref:6.1f} "
              f"(lim {ENERGY_ALLOW:g})  J={j:.4f} "
              f"[h={float(h):.4f} m={W_MASS*float(v):.4f}]")
    # Relative-change stop, final stage only. nlopt has no way to report "the x
    # I just evaluated", so keep it here; a ForcedStop propagates out of
    # optimize() and the caller falls back to this copy.
    if it[0] % SNAPSHOT_EVERY == 0 or it[0] == 1:
        snapshot(xx, _beta[0], dict(
            beta=_beta[0], heat=float(h), compliance=float(c),
            volume=float(v), objective=j, mnd=float(mnd),
            a11_struts=float(a11s), a11_bands=float(a11b),
            spn=float(spn), smax=float(smax)))
    _last_x[0] = onp.array(xx)
    if _final[0]:
        margin = float(c) / ENERGY_ALLOW - 1.0
        w = _conv["window"]
        w.append(j)
        if len(w) == w.maxlen:
            spread = (max(w) - min(w)) / max(abs(sum(w) / len(w)), 1e-12)
            if spread < FINAL_FTOL_REL and margin <= FINAL_FEAS_TOL:
                _conv["stopped"] = (it[0], spread, margin)
                raise nlopt.ForcedStop
    return j


def compliance_constraint(xx, grad):
    _, _, c, gc, *_ = _cached(xx)
    if grad.size > 0:
        grad[:] = gc / ENERGY_ALLOW
    return float(c) / ENERGY_ALLOW - 1.0


n_total = N_FIELDS * n_nodes
lower, upper = onp.full(n_total, -1.0), onp.ones(n_total)
lower[:n_nodes] = 0.0
x_opt = onp.zeros(n_total)
x_opt[:n_nodes] = 0.1
if PERTURB > 0.0:
    rng = onp.random.default_rng(SEED)
    x_opt[:n_nodes] = onp.clip(
        x_opt[:n_nodes] + PERTURB * rng.standard_normal(n_nodes), 0.0, 1.0)

t0 = time.perf_counter()
for stage, beta in enumerate(HEAVISIDE_BETAS):
    _beta[0] = beta
    _cache.clear()
    _final[0] = (stage == len(HEAVISIDE_BETAS) - 1)
    _conv["window"].clear()
    _conv["stopped"] = None
    cap = ITER_PER_STAGE_FINAL if _final[0] else ITER_PER_STAGE
    opt = nlopt.opt(nlopt.LD_MMA, n_total)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(compliance_constraint, 1e-8)
    opt.set_maxeval(cap)
    print(f"  -- stage {stage+1}/{len(HEAVISIDE_BETAS)}: beta = {beta:g}, "
          + (f"stop when J spreads < {FINAL_FTOL_REL:g} over "
             f"{FINAL_WINDOW} iterations (cap {cap}) --"
             if _final[0] else f"{cap} iterations --"))
    try:
        x_opt = opt.optimize(x_opt)
    except nlopt.ForcedStop:
        x_opt = _last_x[0]
        n_it, spread, margin = _conv["stopped"]
        print(f"     converged at it {n_it}: J spread over the last "
              f"{FINAL_WINDOW} iterations = {spread:.2e} < {FINAL_FTOL_REL:g}, "
              f"constraint margin {margin:+.2e}")
    else:
        if _final[0]:
            w = _conv["window"]
            spread = ((max(w) - min(w)) / max(abs(sum(w) / len(w)), 1e-12)
                      if w else float("nan"))
            print(f"     NOT converged: hit the {cap}-evaluation cap with the "
                  f"J spread at {spread:.2e}, still above {FINAL_FTOL_REL:g} "
                  f"(nlopt result {opt.last_optimize_result()})")
    stage_ends.append(it[0])
dt = time.perf_counter() - t0

(h_f0, _, c_f0, _, v_f0, _, spn_f0, smax_f0, s_a, s_b, mnd_f0) = _cached(x_opt)
snapshot(x_opt, _beta[0], dict(
    beta=_beta[0], heat=float(h_f0), compliance=float(c_f0), volume=float(v_f0),
    objective=float(h_f0) + W_MASS * float(v_f0), mnd=float(mnd_f0),
    **{"a11_struts": float(s_a), "a11_bands": float(s_b)},
    spn=float(spn_f0), smax=float(smax_f0)))
_snap.close()

fields = [onp.asarray(f) for f in process(jnp.asarray(x_opt),
                                          jnp.float64(_beta[0]))]
rho = fields[0]
(h_f, _, c_f, _, v_f, _, spn_f, smax_f, a11s_f, a11b_f, mnd_f) = _cached(x_opt)

print("-" * 78)
print(f"objective   : {history['objective'][0]:.4f} -> "
      f"{history['objective'][-1]:.4f}   "
      f"[heat {float(h_f):.4f}  mass {W_MASS*float(v_f):.4f}]")
print(f"heat        : {history['heat'][0]:.4e} -> {float(h_f):.4e} x reference"
      f"   = {float(h_f)*heat_ref_W:.3f} W "
      f"(reference plate {heat_ref_W:.3f} W, {THICKNESS:g} mm thick, "
      f"dT = {T_HOT-T_COLD:g} K)")
print(f"compliance  : {float(c_f):.3f} x reference (limit {ENERGY_ALLOW:g}, "
      f"{'ACTIVE' if float(c_f) > 0.98 * ENERGY_ALLOW else 'SLACK'})")
print(f"volume      : design {float(v_f):.4f}   whole cell {rho.mean():.4f}   "
      f"solid (rho>0.5) {(rho > 0.5).mean()*100:.1f}%")
print(f"discreteness: {float(mnd_f):.1f}% non-discrete")
print(f"stress      : p-norm {float(spn_f):.2f} MPa ({float(spn_f)/stress_ref:.2f}x), "
      f"peak {float(smax_f):.2f} MPa ({float(smax_f)/stress_ref:.2f}x of the "
      f"reference plate) -- reported only, not constrained")
print(f"orientation : a11 = 0 fibres along y (heat/load path), 1 along x "
      f"(isothermal, and the direction a band spreads load)")
print(f"              struts {float(a11s_f):.3f}   bands {float(a11b_f):.3f}")
_iy = onp.rint(pts[:, 1] / LY * NY).astype(int)
_a11n = 0.5 * (fields[1] + 1.0)
_rows = [(j / NY, float((_a11n[_iy == j] * rho[_iy == j]).sum()
                        / max(rho[_iy == j].sum(), 1e-12)))
         for j in range(0, NY + 1, max(1, NY // 20))]
print("                y   " + " ".join(f"{y:5.2f}" for y, _ in _rows))
print("                a11 " + " ".join(f"{a:5.2f}" for _, a in _rows))
print(f"evaluations : {it[0]}   wall time {dt:.1f}s")

# ── Outputs ────────────────────────────────────────────────────────────────
a2 = onp.asarray(jax.vmap(lambda a, b: tx.unit_trace_a2(a, b, SGN_BETA))(
    jnp.asarray(fields[1]), jnp.asarray(fields[2])))
director = onp.asarray(principal_direction(a2))
onp.save(os.path.join(OUTDIR, "density.npy"), rho.astype(onp.float32))
onp.save(os.path.join(OUTDIR, "orientation.npy"), a2.astype(onp.float32))
with open(os.path.join(OUTDIR, "history.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    keys = list(history.keys())
    w.writerow(keys)
    for i in range(len(history["iteration"])):
        w.writerow([history[k][i] for k in keys])

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    tpc = axes[0].tripcolor(pts[:, 0], pts[:, 1], rho, cmap="gray_r",
                            vmin=0, vmax=1, shading="gouraud")
    sl = slice(None, None, 7)
    axes[0].quiver(pts[sl, 0], pts[sl, 1], director[sl, 0] * rho[sl],
                   director[sl, 1] * rho[sl], color="tab:orange", pivot="mid",
                   headwidth=0, headlength=0, headaxislength=0, width=0.003,
                   scale=28)
    axes[0].set_title("density + fibre director")
    axes[0].set_aspect("equal")
    fig.colorbar(tpc, ax=axes[0])

    axes[1].plot(history["iteration"], history["objective"], "k-", label="J")
    axes[1].plot(history["iteration"], history["heat"], "r-", label="heat")
    axes[1].plot(history["iteration"],
                 [W_MASS * v for v in history["volume"]], "b-", label="mass")
    for xb in stage_ends[:-1]:          # actual boundaries: stages can end early
        axes[1].axvline(xb, color="0.8", lw=0.8)
    axes[1].set_xlabel("evaluation")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("objective and its two terms")

    axes[2].plot(history["iteration"], history["compliance"], label="C / ref")
    axes[2].axhline(ENERGY_ALLOW, color="r", ls="--", label="limit")
    ax2 = axes[2].twinx()
    ax2.plot(history["iteration"], history["mnd"], "m:", label="non-discrete %")
    ax2.plot(history["iteration"], history["a11_struts"], "g:", label="a11 struts")
    ax2.set_ylim(0, 100)
    axes[2].set_xlabel("evaluation")
    axes[2].legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="center right")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("constraint / discreteness / orientation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "design.png"), dpi=150)
    plt.close()
except ImportError:
    pass

try:
    # Everything needed to check the answer by eye in ParaView. The stress is
    # the TRUE material stress (q = 0), not the relaxed measure the p-norm uses.
    jf = tuple(jnp.asarray(f) for f in fields)
    tp_me = mech_params(jf)
    T = onp.asarray(jnp.asarray(
        fe.TracedParams.node_var_from_solution(
            thermal, solve_T(fe.TracedParams(volume_vars=jf), guess_T,
                             traced_structure=ts_th))))
    a2j = jnp.asarray(a2)
    k_yy = onp.asarray(jax.vmap(
        lambda a: tx.orientation_conductivity(a, K_PAR, K_PERP)[1, 1])(a2j))
    true_stress = tx.anisotropic_stress_measure(
        E1=E1, E2=E2, G12=G12, nu12=NU12, q=0.0, alpha_par=0.0,
        alpha_perp=0.0, t_ref=0.0, sgn_beta=SGN_BETA)

    points = [("density", rho), ("temperature", T.reshape(-1)),
              ("a11", a2[:, 0, 0]), ("a12", a2[:, 0, 1]), ("k_yy", k_yy),
              ("director", onp.column_stack([director, onp.zeros(n_nodes)]))]
    cells = []

    def _add(problem, solver, ts, label):
        sol = solver(tp_me, guess_u, traced_structure=ts)
        u = onp.asarray(sol.field(0))
        points.append((f"u_{label}", onp.column_stack([u, onp.zeros(len(u))])))
        sig, _ = tx.create_stress_tensor_fn(problem, true_stress)(sol, tp_me)
        sig = onp.asarray(sig).mean(axis=1)          # quadrature -> cell
        vm = onp.sqrt(onp.maximum(
            sig[:, 0, 0]**2 - sig[:, 0, 0] * sig[:, 1, 1] + sig[:, 1, 1]**2
            + 3.0 * sig[:, 0, 1]**2, 0.0))
        cells.extend([(f"vm_{label}", vm), (f"sxx_{label}", sig[:, 0, 0]),
                      (f"syy_{label}", sig[:, 1, 1]), (f"sxy_{label}", sig[:, 0, 1])])

    for i, (name, _v, _w) in enumerate(CASES):
        _add(mech[i], solvers[i], ts_me, name)

    save_sol(mesh, os.path.join(OUTDIR, "final.vtu"),
             point_infos=points, cell_infos=cells)
except Exception as e:
    print(f"(no vtu: {e})")

print(f"snapshots  : {_snap._n} rows in snapshots.h5 "
      f"(every {SNAPSHOT_EVERY} iterations, plus the first and the last)")
print(f"wrote density.npy / orientation.npy / history.csv / design.png / "
      f"final.vtu to {os.path.relpath(OUTDIR, os.getcwd())}/")
