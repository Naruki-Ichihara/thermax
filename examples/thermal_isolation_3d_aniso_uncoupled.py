"""3-D thermal isolation with fibre orientation as a design field, NO thermal strain.

    top face       T = T_HOT,  clamped, passive solid band -- fuselage frame
    bottom face    T = T_COLD, loaded,  passive solid band -- LH2 tank
    other faces    free and insulated

    design per node    rho, t1, t2, t12, t13  (a_2 in 3-D; a_4 from a closure)
    minimise           heat / heat_ref  +  W_MASS * V
    subject to         C / C_ref <= ENERGY_ALLOW

The 3-D counterpart of thermal_isolation_2d_aniso_uncoupled.py. Same bracket
(Aerospace 2026, 13, 388), same AS4/PEEK UD lamina, same objective, same
convergence scheme -- but a real 50 mm block rather than a plane-stress face,
and the fibre director is free to point anywhere on the sphere rather than in
one plane.

What is genuinely different
---------------------------
1. THE ORIENTATION PARAMETERISATION. Four box variables, not two:

       (a11, a22) = box_to_triangle(t1, t2),  a33 = 1 - a11 - a22
       s12 = smooth_sgn(t12),  s13 = smooth_sgn(t13),  s23 = s12 s13
       a_ij = sqrt(a_ii a_jj) s_ij

   The tie ``s23 = s12 s13`` is not a simplification, it is what makes the
   tensor admissible. With the three signs free -- the six-variable Nomura
   construction -- 97.7% of 20000 uniform samples of the box came out with a
   NEGATIVE eigenvalue, down to -1/3, because a rank-1 state ``v v^T`` forces
   ``s12 s13 s23 = +1`` and an odd number of minus signs corresponds to no fibre
   direction at all. Tied, the same sample gives a minimum eigenvalue of
   +3.7e-14 and nothing negative. See thermax.orientation.unit_trace_a2_3d.

2. THE CLOSURE IS WEAKER HERE, AND THE REFERENCE STATE HAD TO MOVE. The hybrid
   closure blends linear and quadratic by ``f = 1 - 27 det(a_2)``, and in 3-D
   that measure cannot tell "fully aligned" from "planar random": both have
   ``det = 0``, so both get the pure quadratic closure, which is right only for
   the first. Measured in-plane modulus for ``a_2 = diag(1/2, 1/2, 0)``:

       hybrid 16.02 GPa   linear 107.79 GPa   (2-D plane-stress truth 49.79)

   and for rank-1 the linear closure returns -294 GPa, so neither is usable
   alone. The one state where it is exact besides rank-1 is 3-D isotropy:
   ``det(I/3) = 1/27`` gives ``f = 0``, the pure linear closure, which is the
   exact average for a spatially random ODF.

   So the reference here is the 3-D ISOTROPIC solid (fibres random in space,
   E = 29.72 GPa), reached at ``t1 = t2 = T_ISO``, and NOT the paper's
   [0/45/90/-45]4S laminate that the 2-D version could use exactly. "x
   reference" therefore means "x a solid block of spatially random AS4/PEEK".
   A 50 mm block is not a laminate anyway.

   The closure being inaccurate in the planar interior does not make the FE
   problem ill-posed: over 4000 samples of the design box the Voigt stiffness
   had a minimum eigenvalue of +3.5e3 MPa and none was negative. And the state
   the design should converge to -- rank-1 -- is where the closure is exact.

3. ``nu23`` is needed and has no 2-D counterpart: the plane-stress stiffness
   never asks for the transverse-plane Poisson ratio, the 3-D invariants do.
   0.35 for AS4/PEEK, Table 1 of the paper.

4. THE HEAT IS A TRUE VOLUME INTEGRAL. In 2-D the conduction energy came out
   per unit thickness and had to be multiplied by 4 mm; here it is already in
   W K, so ``Q = 2 E / dT`` with no thickness factor. Checked against the 1-D
   conductance of the solid isotropic block: FE 24.070 W, hand calculation
   ``k_yy A / L dT`` = 24.070 W.

Everything else -- RAMP pair with opposite q, Heaviside continuation, the
windowed relative-spread stop on the final stage, mass in the objective and
stiffness as a constraint -- is carried over unchanged, and the reasoning for
each is in the 2-D file.

Outputs in outputs/thermal_isolation_3d_aniso_uncoupled_C<..>_m<W_MASS>/.
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
# Outer bracket of Aerospace 2026, 13, 388, Table 8: 150 mm wide, 165 mm tall.
# The 4 mm of Table 8 is the laminate thickness of the real L-joint; here the
# third dimension is a design dimension, so it is set by choice, not by the
# paper.
LX = 100.0                         # mm, bracket width
LY = 50.0                         # mm, bracket height (tank -> frame)
LZ = 100.0                          # mm, block thickness
ELEM_SIZE = 2                    # mm -> 30 x 33 x 10 HEX8, 34,782 DOF
# NX/NY/NZ are read back off the mesh, not computed here. box_mesh divides with
# int(L / mesh_size), so a size that does not divide the edge exactly rounds
# DOWN, and predicting it with round() puts the node-row binning used for the
# a22 profile out by one, which shows up as an empty bin.
PASSIVE_THICKNESS = 0.05 * LY      # mm, bonded band at each interface

# ── Material ───────────────────────────────────────────────────────────────
# AS4/PEEK UD lamina, Table 1. mm / N / MPa / K / W throughout.
E1, E2, G12, NU12 = 129000.0, 9400.0, 4500.0, 0.32
NU23 = 0.35                        # 3-D only; plane stress never needs it
# Built from the constituents (rule of mixtures / Rayleigh, AS4 + PEEK at
# Vf = 0.61), validated on measured AS4/3501-6 to 3% transverse, 10% axial.
# W/(mm K). k_par/k_perp = 5.70 against E1/E2 = 13.72, so AXIAL fibres win.
K_PAR, K_PERP = 4.307e-3, 7.560e-4
E_MIN, K_MIN = 1e-6, 1e-4          # relative floors on the RAMP weights
RAMP_Q_E = 4.0                     # stiffness, q > 0
RAMP_Q = -0.8                      # conductivity, q < 0
SGN_BETA = 10.0

T_HOT, T_COLD = 293.0, 20.0        # K, frame and LH2 tank
# 3-D: the conduction energy is already a volume integral, so no thickness
# factor. E = 1/2 G dT^2 and Q = G dT for a two-terminal problem.
HEAT_TO_WATT = 2.0 / (T_HOT - T_COLD)
# t1 = t2 = T_ISO gives a_2 = I/3 exactly -- the one non-rank-1 state where the
# hybrid closure is exact, and the reference this file normalises by.
T_ISO = -0.1547008478

# Peak bracket resultant load, Table 11: Configuration C (recommended), 3 g
# lateral, 1st row OUTER. A bracket is two L-joints and this domain is one of
# them, so the load per block is half, spread over the loaded face LX x LZ.
# The optimum does not depend on the magnitude (U and U_ref both scale as its
# square); the reported stresses do, and come out in real MPa.
BRACKET_LOAD = 5270.0              # N
L_JOINTS_PER_BRACKET = 2
TRACTION = BRACKET_LOAD / L_JOINTS_PER_BRACKET / (LX * LZ)   # MPa
LOAD_CASES = (
    # Traction on the BOTTOM face (tank), TOP face (frame) clamped. feax adds
    # the surface map to the residual, so a positive component acts in the
    # negative coordinate direction: on this face (0, +T, 0) pulls the tank
    # interface DOWN, away from the clamp, which is tension.
    ("tension", (0.0, TRACTION, 0.0), 1.0),
    ("shear",   (TRACTION, 0.0, 0.0), 0.0),
)

# ── Objective and constraint ───────────────────────────────────────────────
ENERGY_ALLOW = 1                 # C / C_ref must stay below this
W_MASS = 2                         # weight on volume, against heat

# ── Optimisation ───────────────────────────────────────────────────────────
N_FIELDS = 5                       # rho, t1, t2, t12, t13
FILTER_RADIUS = 2.0 * ELEM_SIZE
FILTER_RADIUS_ORI = 2.0 * ELEM_SIZE
HEAVISIDE_BETAS = (2, 4, 8)
ITER_PER_STAGE = 50
ITER_PER_STAGE_FINAL = 50         # hard cap on the final stage
FINAL_WINDOW = 10                  # trailing outer iterations tested
FINAL_FTOL_REL = 1e-3              # relative spread of J over that window
FINAL_FEAS_TOL = 1e-3              # and C/ENERGY_ALLOW - 1 must be <= this
SEED, PERTURB = 0, 0.0

# Snapshots. The design fields are written every SNAPSHOT_EVERY outer
# iterations to snapshots.h5, incrementally, with every constant above
# stored as a root attribute -- so a sweep's files say what produced them
# and an interrupted run still leaves everything up to the last snapshot.
SNAPSHOT_EVERY = 10
PRINT_EVERY = 1                    # iterations between progress lines

STRESS_P = 12.0
STRESS_RELAX_Q = 2.0

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs",
    f"thermal_isolation_3d_aniso_uncoupled_C{ENERGY_ALLOW:g}_m{W_MASS:g}")
os.makedirs(OUTDIR, exist_ok=True)

TOL = 1e-6 * max(LX, LY, LZ)
bottom_face = lambda p: jnp.isclose(p[1], 0.0, atol=TOL)
top_face = lambda p: jnp.isclose(p[1], LY, atol=TOL)

mesh = fe.mesh.box_mesh(size=(LX, LY, LZ), mesh_size=ELEM_SIZE,
                        element_type="HEX8")
n_nodes = mesh.points.shape[0]
pts = onp.asarray(mesh.points)
NX, NY, NZ = (len(onp.unique(pts[:, d])) - 1 for d in range(3))

passive_mask = jnp.asarray((pts[:, 1] <= PASSIVE_THICKNESS + TOL)
                           | (pts[:, 1] >= LY - PASSIVE_THICKNESS - TOL))
design_mask = ~passive_mask
n_design = int(design_mask.sum())

CASES = [(name, tvec, w) for name, tvec, w in LOAD_CASES if w > 0.0]
W_SUM = sum(w for _n, _v, w in CASES)

print(f"3-D thermal isolation, uncoupled   outer bracket "
      f"{LX:g} x {LY:g} x {LZ:g} mm, {ELEM_SIZE:g} mm elements")
print(f"  top = fuselage frame (T_HOT, clamped), "
      f"bottom = LH2 tank (T_COLD, loaded)")
print(f"  {NX}x{NY}x{NZ} HEX8  {mesh.cells.shape[0]:,} cells  {n_nodes:,} nodes "
      f"({3*n_nodes:,} elastic DOF)")
print(f"  AS4/PEEK UD lamina E1/E2 = {E1/E2:.2f}, nu23 = {NU23:g}, "
      f"k_par/k_perp = {K_PAR/K_PERP:.2f}; RAMP q_E = {RAMP_Q_E:g}, "
      f"q_k = {RAMP_Q:g}")
print(f"  load {BRACKET_LOAD/1000:g} kN per bracket / {L_JOINTS_PER_BRACKET} "
      f"L-joints / ({LX:g} x {LZ:g} mm) = {TRACTION:.4f} MPa; "
      + ", ".join(f"{n} w={w:g}" for n, _v, w in CASES))
print(f"  design {N_FIELDS} x {n_nodes:,} = {N_FIELDS*n_nodes:,} variables; "
      f"{n_design:,} design nodes, passive {float(passive_mask.mean())*100:.1f}%")
print(f"  minimise heat/reference + {W_MASS:g}*V,  "
      f"C/reference <= {ENERGY_ALLOW:g}")

# ── Problems and solvers ───────────────────────────────────────────────────
thermal = tx.anisotropic_diffusion_problem_3d(
    mesh, k_par=K_PAR, k_perp=K_PERP, k_min=K_MIN, ramp_q=RAMP_Q,
    sgn_beta=SGN_BETA, ele_type="HEX8")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(top_face, "all", T_HOT),         # fuselage frame
    fe.DirichletBCSpec(bottom_face, "all", T_COLD),     # LH2 tank
]).create_bc(thermal)
ts_th = fe.TracedStructure.from_problem(thermal)

direct = fe.DirectSolverOptions()
_o, _z = jnp.ones((n_nodes,)), jnp.zeros((n_nodes,))
_TH_TEMPLATE = fe.TracedParams(volume_vars=(_o, _z, _z, _z, _z))
solve_T = fe.create_solver(thermal, bc_th, solver_options=direct,
                           adjoint_solver_options=direct, linear=True,
                           traced_structure=ts_th,
                           traced_params=_TH_TEMPLATE)
guess_T = fe.zero_like_initial_guess(thermal, bc_th)
conduction_energy = tx.create_conduction_energy_fn(thermal)

stress_at_quad = tx.anisotropic_stress_measure_3d(
    E1=E1, E2=E2, G12=G12, nu12=NU12, nu23=NU23, q=STRESS_RELAX_Q,
    sgn_beta=SGN_BETA)
mech, solvers, energies, vonmises_fns = [], [], [], []
bc_me = ts_me = None
for name, tvec, _w in CASES:
    p = tx.anisotropic_elasticity_problem_3d(
        mesh, E1=E1, E2=E2, G12=G12, nu12=NU12, nu23=NU23, e_min=E_MIN,
        ramp_q=RAMP_Q_E, sgn_beta=SGN_BETA, ele_type="HEX8",
        location_fns=(bottom_face,), tractions=(tvec,))
    if bc_me is None:
        bc_me = fe.DirichletBCConfig([
            fe.DirichletBCSpec(top_face, "all", 0.0)]).create_bc(p)
        ts_me = fe.TracedStructure.from_problem(p)
    mech.append(p)
    solvers.append(fe.create_solver(
        p, bc_me, solver_options=direct, adjoint_solver_options=direct,
        linear=True, traced_structure=ts_me, traced_params=_TH_TEMPLATE))
    energies.append(fe.create_energy_fn(p))
    vonmises_fns.append(tx.create_vonmises_fn(p, stress_at_quad))
guess_u = fe.zero_like_initial_guess(mech[0], bc_me)

filter_rho = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS)
filter_ori = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS_ORI)


# ── Design fields and responses ────────────────────────────────────────────
def process(x, beta):
    """Raw design vector -> physical (rho, t1, t2, t12, t13).

    Density is filtered, Heaviside-projected and forced to 1 on the passive
    nodes. Orientation is only filtered: projection has no meaning for a field
    with no void state, and the two sign variables in particular must stay free
    to pass through zero, which is where the off-diagonals vanish.
    """
    f = tuple(x[i * n_nodes:(i + 1) * n_nodes] for i in range(N_FIELDS))
    rho = gene.heaviside_projection(filter_rho(f[0]), beta=beta, threshold=0.5,
                                    normalize=True)
    rho = jnp.where(passive_mask, 1.0, rho)
    return (rho,) + tuple(filter_ori(f[i]) for i in range(1, N_FIELDS))


def heat(fields):
    """Conduction energy of the solved temperature field, W K."""
    tp = fe.TracedParams(volume_vars=fields)
    return conduction_energy(solve_T(tp, guess_T, traced_structure=ts_th), tp)


def stored_energy(tp):
    """Weighted mean of U = 1/2 int eps : C : eps over the load cases, each
    normalised by its own reference. With no thermal strain this is exactly
    half the compliance."""
    return sum(w * energies[i](
        solvers[i](tp, guess_u, traced_structure=ts_me).flat, tp)
        / energy_ref[i] for i, (_n, _v, w) in enumerate(CASES)) / W_SUM


def stress_pnorm(tp):
    """Aggregated 3-D von Mises over every load case at once."""
    sig, wts = [], []
    for i in range(len(CASES)):
        s, jxw = vonmises_fns[i](
            solvers[i](tp, guess_u, traced_structure=ts_me), tp)
        sig.append(s.ravel())
        wts.append(jxw.ravel())
    return tx.pnorm(jnp.concatenate(sig), jnp.concatenate(wts), P=STRESS_P)


# ── Reference: solid block, 3-D isotropic composite (a2 = I/3) ─────────────
_ti = jnp.full((n_nodes,), T_ISO)
ref = (_o, _ti, _ti, _z, _z)
_tp_ref = fe.TracedParams(volume_vars=ref)
heat_ref = float(heat(ref))
heat_ref_W = heat_ref * HEAT_TO_WATT
energy_ref = [float(energies[i](
    solvers[i](_tp_ref, guess_u, traced_structure=ts_me).flat, _tp_ref))
    for i in range(len(CASES))]
stress_ref = float(stress_pnorm(_tp_ref)[1])
print(f"  reference (a2 = I/3 solid): heat={heat_ref_W:.3f} W  "
      + "  ".join(f"U_{n}={c:.4e}" for (n, _v, _w), c in zip(CASES, energy_ref))
      + f"  peak von Mises={stress_ref:.3f} MPa")
print("-" * 78)

_snap = tx.SnapshotWriter(
    os.path.join(OUTDIR, "snapshots.h5"), mesh,
    ("rho", "t1", "t2", "t12", "t13", "a11", "a22", "a33", "director"),
    attrs=dict(
        LX=LX, LY=LY, LZ=LZ, ELEM_SIZE=ELEM_SIZE, dim=3,
        NU23=NU23, T_ISO=T_ISO, PASSIVE_THICKNESS=PASSIVE_THICKNESS,
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
    # The four box coordinates mean nothing in a viewer; a11/a22/a33 and the
    # director do. a22 is the one to colour by -- fibre along the heat path.
    a2 = jax.vmap(lambda a, b, c, e: tx.unit_trace_a2_3d(a, b, c, e, SGN_BETA))(
        f[1], f[2], f[3], f[4])
    d = principal_direction(a2)
    _snap.append(it[0], [onp.asarray(a) for a in
                         f + (a2[:, 0, 0], a2[:, 1, 1], a2[:, 2, 2], d)], scalars)


@jax.jit
def evaluate(x, beta):
    def _h(z):
        return heat(process(z, beta)) / heat_ref

    def _u(z):
        return stored_energy(fe.TracedParams(volume_vars=process(z, beta)))

    h, gh = jax.value_and_grad(_h)(x)
    c, gc = jax.value_and_grad(_u)(x)
    v, gv = jax.value_and_grad(
        lambda z: jnp.sum(process(z, beta)[0] * design_mask) / n_design)(x)
    f = process(x, beta)
    spn, smax = stress_pnorm(fe.TracedParams(volume_vars=f))
    # a22 is the fraction of fibre lying along y, the heat path and the load
    # path: 1 = all axial, 0 = none, 1/3 = spatially random.
    a22 = jax.vmap(lambda t1, t2, t12, t13: tx.unit_trace_a2_3d(
        t1, t2, t12, t13, SGN_BETA)[1, 1])(f[1], f[2], f[3], f[4])
    solid = f[0] * design_mask
    return (h, gh, c, gc, v, gv, spn, smax,
            jnp.sum(a22 * solid) / jnp.maximum(jnp.sum(solid), 1e-12),
            jnp.sum(a22 * passive_mask) / jnp.sum(passive_mask),
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
                           "objective", "a22_struts", "a22_bands", "mnd",
                           "spn", "smax")}
it = [0]
stage_ends: list = []
_final = [False]
_conv = {"window": collections.deque(maxlen=FINAL_WINDOW), "stopped": None}
_last_x = [None]


def objective(xx, grad):
    h, gh, c, gc, v, gv, spn, smax, a22s, a22b, mnd = _cached(xx)
    if grad.size > 0:
        grad[:] = gh + W_MASS * gv
    it[0] += 1
    j = float(h) + W_MASS * float(v)
    for k, val in (("iteration", it[0]), ("beta", _beta[0]),
                   ("heat", float(h)), ("compliance", float(c)),
                   ("volume", float(v)), ("objective", j),
                   ("a22_struts", float(a22s)), ("a22_bands", float(a22b)),
                   ("mnd", float(mnd)), ("spn", float(spn) / stress_ref),
                   ("smax", float(smax) / stress_ref)):
        history[k].append(val)
    if it[0] % PRINT_EVERY == 0 or it[0] == 1:
        print(f"  it {it[0]:4d} b={_beta[0]:4.1f} heat={float(h):.3e} "
              f"C={float(c):6.2f} V={float(v):.4f} "
              f"a22[str={float(a22s):.2f} bnd={float(a22b):.2f}] "
              f"grey={float(mnd):4.1f}% sMAX={float(smax)/stress_ref:6.1f} "
              f"(lim {ENERGY_ALLOW:g})  J={j:.4f} "
              f"[h={float(h):.4f} m={W_MASS*float(v):.4f}]")
    if it[0] % SNAPSHOT_EVERY == 0 or it[0] == 1:
        snapshot(xx, _beta[0], dict(
            beta=_beta[0], heat=float(h), compliance=float(c),
            volume=float(v), objective=j, mnd=float(mnd),
            a22_struts=float(a22s), a22_bands=float(a22b),
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
# Start at the isotropic orientation, the same state the reference uses, so the
# run begins with no direction preferred and no bound active.
x_opt[n_nodes:3 * n_nodes] = T_ISO
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
    **{"a22_struts": float(s_a), "a22_bands": float(s_b)},
    spn=float(spn_f0), smax=float(smax_f0)))
_snap.close()

fields = [onp.asarray(f) for f in process(jnp.asarray(x_opt),
                                          jnp.float64(_beta[0]))]
rho = fields[0]
(h_f, _, c_f, _, v_f, _, spn_f, smax_f, a22s_f, a22b_f, mnd_f) = _cached(x_opt)

print("-" * 78)
print(f"objective   : {history['objective'][0]:.4f} -> "
      f"{history['objective'][-1]:.4f}   "
      f"[heat {float(h_f):.4f}  mass {W_MASS*float(v_f):.4f}]")
print(f"heat        : {history['heat'][0]:.4e} -> {float(h_f):.4e} x reference"
      f"   = {float(h_f)*heat_ref_W:.3f} W "
      f"(reference solid block {heat_ref_W:.3f} W, dT = {T_HOT-T_COLD:g} K)")
print(f"compliance  : {float(c_f):.3f} x reference (limit {ENERGY_ALLOW:g}, "
      f"{'ACTIVE' if float(c_f) > 0.98 * ENERGY_ALLOW else 'SLACK'})")
print(f"volume      : design {float(v_f):.4f}   whole block {rho.mean():.4f}   "
      f"solid (rho>0.5) {(rho > 0.5).mean()*100:.1f}%")
print(f"discreteness: {float(mnd_f):.1f}% non-discrete")
print(f"stress      : p-norm {float(spn_f):.2f} MPa ({float(spn_f)/stress_ref:.2f}x), "
      f"peak {float(smax_f):.2f} MPa ({float(smax_f)/stress_ref:.2f}x of the "
      f"reference block) -- reported only, not constrained")
print(f"orientation : a22 = fraction of fibre along y (the heat and load path). "
      f"1 = fully axial, 1/3 = spatially random, 0 = none")
print(f"              struts {float(a22s_f):.3f}   bands {float(a22b_f):.3f}")

a2 = onp.asarray(jax.vmap(lambda a, b, c, d: tx.unit_trace_a2_3d(
    a, b, c, d, SGN_BETA))(*[jnp.asarray(f) for f in fields[1:]]))
_iy = onp.rint(pts[:, 1] / LY * NY).astype(int)
_rows = [(j / NY, float((a2[_iy == j, 1, 1] * rho[_iy == j]).sum()
                        / max(rho[_iy == j].sum(), 1e-12)))
         for j in range(0, NY + 1, max(1, NY // 16))]
print("                y   " + " ".join(f"{y:5.2f}" for y, _ in _rows))
print("                a22 " + " ".join(f"{a:5.2f}" for _, a in _rows))
print(f"evaluations : {it[0]}   wall time {dt:.1f}s")

# ── Outputs ────────────────────────────────────────────────────────────────
director = onp.asarray(principal_direction(jnp.asarray(a2)))
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(history["iteration"], history["objective"], "k-", label="J")
    axes[0].plot(history["iteration"], history["heat"], "r-", label="heat")
    axes[0].plot(history["iteration"],
                 [W_MASS * v for v in history["volume"]], "b-", label="mass")
    for xb in stage_ends[:-1]:
        axes[0].axvline(xb, color="0.8", lw=0.8)
    axes[0].set_xlabel("evaluation")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("objective and its two terms")

    axes[1].plot(history["iteration"], history["compliance"], label="C / ref")
    axes[1].axhline(ENERGY_ALLOW, color="r", ls="--", label="limit")
    ax2 = axes[1].twinx()
    ax2.plot(history["iteration"], history["mnd"], "m:", label="non-discrete %")
    ax2.plot(history["iteration"], [100 * a for a in history["a22_struts"]],
             "g:", label="a22 struts x100")
    ax2.set_ylim(0, 100)
    axes[1].set_xlabel("evaluation")
    axes[1].legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="center right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("constraint / discreteness / orientation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "history.png"), dpi=150)
    plt.close()
except ImportError:
    pass

try:
    # The stress written here is the TRUE material stress (q = 0), not the
    # relaxed measure the p-norm uses.
    jf = tuple(jnp.asarray(f) for f in fields)
    tp = fe.TracedParams(volume_vars=jf)
    T = onp.asarray(jnp.asarray(fe.TracedParams.node_var_from_solution(
        thermal, solve_T(tp, guess_T, traced_structure=ts_th))))
    a2j = jnp.asarray(a2)
    k_yy = onp.asarray(jax.vmap(
        lambda a: tx.orientation_conductivity(a, K_PAR, K_PERP)[1, 1])(a2j))
    true_stress = tx.anisotropic_stress_measure_3d(
        E1=E1, E2=E2, G12=G12, nu12=NU12, nu23=NU23, q=0.0, sgn_beta=SGN_BETA)

    points = [("density", rho), ("temperature", T.reshape(-1)),
              ("a11", a2[:, 0, 0]), ("a22", a2[:, 1, 1]), ("a33", a2[:, 2, 2]),
              ("k_yy", k_yy), ("director", director)]
    cells = []

    def _add(problem, solver, label):
        sol = solver(tp, guess_u, traced_structure=ts_me)
        points.append((f"u_{label}", onp.asarray(sol.field(0))))
        sig, _ = tx.create_stress_tensor_fn(problem, true_stress)(sol, tp)
        sig = onp.asarray(sig).mean(axis=1)          # quadrature -> cell
        dev = sig - onp.eye(3) * (onp.trace(sig, axis1=1, axis2=2)[:, None, None] / 3.0)
        vm = onp.sqrt(onp.maximum(1.5 * (dev * dev).sum(axis=(1, 2)), 0.0))
        cells.extend([(f"vm_{label}", vm), (f"sxx_{label}", sig[:, 0, 0]),
                      (f"syy_{label}", sig[:, 1, 1]), (f"szz_{label}", sig[:, 2, 2]),
                      (f"sxy_{label}", sig[:, 0, 1])])

    for i, (name, _v, _w) in enumerate(CASES):
        _add(mech[i], solvers[i], name)

    save_sol(mesh, os.path.join(OUTDIR, "final.vtu"),
             point_infos=points, cell_infos=cells)
except Exception as e:
    print(f"(no vtu: {e})")

print(f"snapshots  : {_snap._n} rows in snapshots.h5 "
      f"(every {SNAPSHOT_EVERY} iterations, plus the first and the last)")
print(f"wrote density.npy / orientation.npy / history.csv / history.png / "
      f"final.vtu to {os.path.relpath(OUTDIR, os.getcwd())}/")
