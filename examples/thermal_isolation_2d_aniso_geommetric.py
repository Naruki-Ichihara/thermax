"""2-D thermal isolation with fibre orientation as a design field, NO thermal strain.
"""
from __future__ import annotations

import collections
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
import feax.gene as gene
from feax.experimental.mma import GCMMA
from feax.mechanics.orientation import principal_direction
from feax.utils import save_sol

import thermax as tx

LX = 150.0                         # mm, bracket width
LY = 115.0                         # mm, inner bracket height
ELEM_SIZE = 1.0                    # mm, target element edge -> 100 x 110
NX = int(round(LX / ELEM_SIZE))
NY = int(round(LY / ELEM_SIZE))
PASSIVE_THICKNESS = 0.05 * LY      # mm, non-design solid band on each edge

E1, E2, G12, NU12 = 129000.0, 9400.0, 4500.0, 0.32
K_PAR, K_PERP = 4.307e-3, 7.560e-4
E_MIN, K_MIN = 1e-6, 1e-4
RAMP_Q_E = 8.0                     # stiffness, q > 0
RAMP_Q = -0.8                      # conductivity, q < 0
SGN_BETA = 10.0

T_HOT, T_COLD = 293.0, 20.0        # K, frame and LH2 tank
THICKNESS = 4.0                    # mm
HEAT_TO_WATT = 2.0 * THICKNESS / (T_HOT - T_COLD)
# No thermal expansion: this is what makes the two physics independent.
ALPHA_PAR = ALPHA_PERP = 0.0
BRACKET_LOAD = 5270.0              # N, Table 11, config C / 3 g lateral / outer
L_JOINTS_PER_BRACKET = 2
TRACTION = BRACKET_LOAD / L_JOINTS_PER_BRACKET / (LX * THICKNESS)   # MPa
# (name, traction on the top edge, weight in the compliance term). Only cases
# with a positive weight are built and solved.
LOAD_CASES = (
    ("tension", (0.0, TRACTION), 1.0),
    ("shear",   (TRACTION, 0.0), 0.0),
)

# ── Objective and constraint ───────────────────────────────────────────────
ENERGY_ALLOW = 2      # C / C_ref must stay below this
W_MASS = 1                      # weight on volume, against heat

# ── Optimisation ───────────────────────────────────────────────────────────
N_FIELDS = 3                       # rho, t1, t2  (a_4 comes from a closure)
ANISOTROPY = False

# ── Minimum length scale ───────────────────────────────────────────────────
# Geometric constraints of Zhou, Lazarov, Wang & Sigmund, CMAME 293 (2015)
# 266-282. The filtered field is smooth, so the centreline of a thin member is
# a stationary point of it; the constraints mark those points and require the
# filtered field to have reached eta_e (solid) or dropped below eta_d (void)
# there. See feax.gene.length_scale.
#
# The filter sets the enforced width, so MIN_WIDTH sets FILTER_RADIUS and not
# the other way round. The kernel is the paper's linear hat, so its calibration
# (c = r**4, the eps curve, the width relation) transfers verbatim; it is also
# 60x cheaper per call than the Helmholtz filter used for the orientation.
MIN_WIDTH = 6                    # mm, target minimum member width and gap
LS_ETA_D, LS_ETA_E = 0.3, 0.7      # dilated / eroded thresholds, blueprint 0.5
FILTER_RADIUS = gene.radius_for_minimum_width(MIN_WIDTH, LS_ETA_E)
FILTER_RADIUS_ORI = 2.0 * LX / NX
# Relaxation of Eq. (12). Derived from r = R/h, not pinned: a mesh twice as
# coarse needs roughly twenty times the slack.
LS_EPS = gene.epsilon_for_radius(FILTER_RADIUS / ELEM_SIZE)
# Which measure the CONSTRAINT uses. Both are reported either way.
#
#   "geometric"     -- Zhou et al. Eqs. (10-11). Polices the stationary points
#                      of the filtered field, which is what their topology
#                      argument needs but is not the same as a minimum width:
#                      a member that tapers has no stationary point, its
#                      indicator is killed by exp(-c |grad|^2), and it may thin
#                      without limit. Measured here, a strut tapering to 2 mm
#                      against an 8 mm target left g_s inside its bound.
#   "morphological" -- opening/closing of the blueprint. Asks the width
#                      question directly, so tapering is caught, at the cost of
#                      saying nothing about threshold robustness.
LS_METHOD = "morphological"
# Erosion diameter, one element below MIN_WIDTH -- see the function's docstring
# for why the target itself would otherwise read as a violation.
LS_MORPH_WIDTH = MIN_WIDTH - ELEM_SIZE
LS_MORPH_SHARPNESS = 80.0
LS_MORPH_P = 12.0
LS_MORPH_EPS = 3.5e-2
HEAVISIDE_BETAS = (2, 4, 8, 12, 16)
LS_INDICATOR_BETA = 32.0
LS_FROM_BETA = 12.0
LS_FROM_STAGE = next(
    (i for i, b in enumerate(HEAVISIDE_BETAS) if b >= LS_FROM_BETA),
    len(HEAVISIDE_BETAS) - 1)

ITER_PER_STAGE = (6, 6, 10, 20)
assert len(ITER_PER_STAGE) == len(HEAVISIDE_BETAS) - 1

ITER_PER_STAGE_FINAL = 50         # hard cap on the final stage
FINAL_WINDOW = 10                  # trailing outer iterations tested
MMA_MAX_INNER = 5
MMA_MOVE = None
MMA_C_COMPLIANCE = 1e7
MMA_C_LENGTH_SCALE = 1e6
FINAL_FTOL_REL = 1e-3              # relative spread of J over that window
FINAL_FEAS_TOL = 1e-3              # and C/ENERGY_ALLOW - 1 must be <= this
SEED, PERTURB = 0, 0.

SNAPSHOT_EVERY = 2
PRINT_EVERY = 1                    # iterations between progress lines

# p-norm.
STRESS_P = 12.0
STRESS_RELAX_Q = 2.0
STRAIN_ALLOW = 5.0

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs",
    f"thermal_isolation_2d_aniso_uncoupled_C{ENERGY_ALLOW:g}_m{W_MASS:g}"
    f"{'' if ANISOTROPY else '_iso'}"
    f"{'' if PERTURB == 0.0 else f'_s{SEED}p{PERTURB:g}'}")
os.makedirs(OUTDIR, exist_ok=True)

TOL = 1e-6 * max(LX, LY)
bottom_edge = lambda p: jnp.isclose(p[1], 0.0, atol=TOL)
top_edge = lambda p: jnp.isclose(p[1], LY, atol=TOL)

mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                              ele_type="QUAD4")
n_nodes = mesh.points.shape[0]
pts = onp.asarray(mesh.points)
cells = onp.asarray(mesh.cells)
n_cells = cells.shape[0]
centroids = pts[cells].mean(axis=1)
_node_hits = onp.bincount(cells.reshape(-1), minlength=n_nodes).astype(float)


def cell_to_node(cell_field):
    """Average an element field onto the nodes, for OUTPUT only -- the physics
    reads the element field directly."""
    acc = onp.bincount(cells.reshape(-1),
                       weights=onp.repeat(onp.asarray(cell_field), cells.shape[1]),
                       minlength=n_nodes)
    return acc / onp.maximum(_node_hits, 1.0)

# Element masks for the density, node masks for the orientation diagnostics.
passive_cell = jnp.asarray((centroids[:, 1] <= PASSIVE_THICKNESS + TOL)
                           | (centroids[:, 1] >= LY - PASSIVE_THICKNESS - TOL))
design_cell = ~passive_cell
n_design = int(design_cell.sum())
passive_mask = jnp.asarray((pts[:, 1] <= PASSIVE_THICKNESS + TOL)
                           | (pts[:, 1] >= LY - PASSIVE_THICKNESS - TOL))
design_mask = ~passive_mask

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
print(f"  design rho on {n_cells:,} elements"
      + (f" + 2 x {n_nodes:,} orientation nodes" if ANISOTROPY
         else "  (orientation FIXED isotropic, a2 = I/2)")
      + f" = {n_cells + (2*n_nodes if ANISOTROPY else 0):,} variables; "
      + f"{n_design:,} design elements, "
      + f"passive {float(passive_cell.mean())*100:.1f}%")
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
_o, _z = jnp.ones((n_cells,)), jnp.zeros((n_nodes,))
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
# The same measure with q = 0: sigma = C0 : eps, so no density weighting on the
# field. A strain measure wearing stress units. See strain_pnorm.
strain_at_quad = tx.anisotropic_stress_measure(
    E1=E1, E2=E2, G12=G12, nu12=NU12, q=0.0,
    alpha_par=0.0, alpha_perp=0.0, t_ref=0.0, sgn_beta=SGN_BETA)
mech, solvers, energies, vonmises_fns, strain_fns = [], [], [], [], []
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
    strain_fns.append(tx.create_vonmises_fn(p, strain_at_quad))
guess_u = fe.zero_like_initial_guess(mech[0], bc_me)

# Density (linear hat) filter for rho -- the paper's kernel, see MIN_WIDTH.
# The orientation field keeps the Helmholtz filter: it has no void state, no
# length scale is imposed on it, and its radius is small enough that the CG
# solve is cheap.
# Filtered on the element centroids, so the filter maps (n_cells,) to itself.
filter_rho = gene.create_density_filter(mesh, radius=FILTER_RADIUS,
                                        weight_type="cone", points=centroids)
filter_ori = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS_ORI)

# Restricted to the design region: the passive bands are solid by construction,
# so including them would only add a constant violation.
length_scale = gene.create_geometric_length_scale_fn(
    mesh, filter_radius=FILTER_RADIUS,
    ele_type="QUAD4", dim=2, eta_e=LS_ETA_E, eta_d=LS_ETA_D,
    element_size=ELEM_SIZE, mask=design_cell.astype(jnp.float64),
    field="cell")
morph_length_scale = gene.create_morphological_length_scale_fn(
    centroids, min_width=LS_MORPH_WIDTH, sharpness=LS_MORPH_SHARPNESS,
    p=LS_MORPH_P, mask=design_cell.astype(jnp.float64))
LS_BOUND = LS_MORPH_EPS if LS_METHOD == "morphological" else LS_EPS


# ── Design fields and responses ────────────────────────────────────────────
def process(x, beta):
    """Raw design vector -> physical (rho, t1, t2).

    Density is filtered, Heaviside-projected, then forced to 1 on the passive
    nodes -- ``where`` also kills their gradient, so those variables are inert.
    Orientation is only filtered: projection has no meaning for a field with no
    void state.
    """
    return design_fields(x, beta)[1]


def design_fields(x, beta):
    """``(rho_tilde, (rho, t1, t2))`` -- as ``process``, but the filtered
    density comes out too.  The length-scale constraints need it: they test
    where ``rho_tilde`` is stationary, which the projected field cannot say."""
    rho_raw = x[:n_cells]
    rho_tilde = filter_rho(rho_raw)
    rho = gene.heaviside_projection(rho_tilde, beta=beta, threshold=0.5)
    rho = jnp.where(passive_cell, 1.0, rho)
    if ANISOTROPY:
        t1 = x[n_cells:n_cells + n_nodes]
        t2 = x[n_cells + n_nodes:]
        return rho_tilde, (rho, filter_ori(t1), filter_ori(t2))
    # t1 = t2 = 0 is a2 = I/2, the random-planar composite: isotropic in plane.
    return rho_tilde, (rho, _z, _z)


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


def strain_pnorm(tp_me):
    """Aggregated deformation: the measure that sees a weak load path.

    The relaxed stress measure cannot -- its ``rho**q`` crushes the value
    exactly where the density is low, so a bracket cut in two scores LOWER than
    an intact one. With q = 0 the density leaves the field and what remains is
    proportional to strain, which grows with how broken the structure is.
    """
    sig, wts = [], []
    rho_q = tp_me.volume_vars[0][:, None]          # element density, per quad pt
    for i in range(len(CASES)):
        s, jxw = strain_fns[i](
            solvers[i](tp_me, guess_u, traced_structure=ts_me), tp_me)
        sig.append(s.ravel())
        # Weight the AGGREGATION by density, never the field. Unweighted, 52%
        # of the p-norm came from elements below rho = 0.01 -- it was measuring
        # how freely the void deforms. Weighting the field by rho**q instead is
        # the opposite failure (see the docstring). This keeps the field at
        # C0 : eps, so a bridge at rho = 0.31 reports its full value while the
        # void beside it is scaled out.
        wts.append((jnp.broadcast_to(rho_q, s.shape) * jxw).ravel())
    return tx.pnorm(jnp.concatenate(sig), jnp.concatenate(wts), P=STRESS_P)


# ── Reference: solid cell, isotropic composite (a2 = I/2, random-planar a4) ─
ref = (_o, _z, _z)
_tp_ref = mech_params(ref)
heat_ref = float(heat(ref))
energy_ref = [float(energies[i](
    solvers[i](_tp_ref, guess_u, traced_structure=ts_me).flat, _tp_ref))
    for i in range(len(CASES))]
stress_ref = float(stress_pnorm(_tp_ref)[1])
strain_ref = float(strain_pnorm(_tp_ref)[0])
heat_ref_W = heat_ref * HEAT_TO_WATT
print(f"  reference: heat={heat_ref_W:.3f} W  "
      + "  ".join(f"U_{n}={c:.4e}" for (n, _v, _w), c in zip(CASES, energy_ref))
      + f"  peak von Mises={stress_ref:.3f} MPa"
      + f"  strain p-norm={strain_ref:.3f} (limit {STRAIN_ALLOW:g}x)")
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
        ANISOTROPY=int(ANISOTROPY),
        MIN_WIDTH=MIN_WIDTH, LS_ETA_D=LS_ETA_D, LS_ETA_E=LS_ETA_E,
        LS_EPS=LS_EPS, LS_BOUND=LS_BOUND, LS_METHOD=LS_METHOD,
        LS_MORPH_WIDTH=LS_MORPH_WIDTH, LS_MORPH_EPS=LS_MORPH_EPS,
        LS_MORPH_P=LS_MORPH_P,
        LS_FROM_STAGE=LS_FROM_STAGE,
        LS_FROM_BETA=LS_FROM_BETA, LS_INDICATOR_BETA=LS_INDICATOR_BETA,
        HEAVISIDE_BETAS=list(HEAVISIDE_BETAS), ITER_PER_STAGE=list(ITER_PER_STAGE),
        ITER_PER_STAGE_FINAL=ITER_PER_STAGE_FINAL,
        FINAL_WINDOW=FINAL_WINDOW, FINAL_FTOL_REL=FINAL_FTOL_REL,
        FINAL_FEAS_TOL=FINAL_FEAS_TOL, SNAPSHOT_EVERY=SNAPSHOT_EVERY,
        SEED=SEED, PERTURB=PERTURB, STRESS_P=STRESS_P,
        STRESS_RELAX_Q=STRESS_RELAX_Q, STRAIN_ALLOW=STRAIN_ALLOW,
        strain_ref=strain_ref,
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
    # rho is per element; the writer and every viewer downstream want nodes.
    rho_n = cell_to_node(onp.asarray(f[0]))
    # The director is masked by density too: a glyph filter otherwise draws a
    # fibre direction across the whole void, where there is no fibre.
    dm = onp.asarray(d) * onp.clip(rho_n, 0.0, 1.0)[:, None]
    _snap.append(it[0], [rho_n] + [onp.asarray(a) for a in
                                   (f[1], f[2], a2[:, 0, 0])] + [dm], scalars)


# ── What the optimiser is handed ───────────────────────────────────────────
# One function returns the objective, its gradient, the constraints and their
# Jacobian, and every logged diagnostic via the optimiser's `has_aux` channel.
# One jit matters: stored_energy and the two p-norms share a mechanical solve,
# and XLA only eliminates the duplicate inside a single trace.
def _length_scale_both(x, beta):
    """``(g_s, g_v)`` from BOTH measures: ``(geo_s, geo_v, mor_s, mor_v)``.

    The geometric pair's solid/void mask uses LS_INDICATOR_BETA, not the
    continuation's beta, so it measures geometry rather than how far the
    projection has sharpened.
    """
    rho_tilde, fields = design_fields(x, beta)
    rho_sharp = jnp.where(passive_cell, 1.0, gene.heaviside_projection(
        rho_tilde, beta=LS_INDICATOR_BETA, threshold=0.5))
    geo_s, geo_v = length_scale(rho_tilde, rho_sharp)
    mor_s, mor_v = morph_length_scale(fields[0])
    return geo_s, geo_v, mor_s, mor_v


def _length_scale_raw(x, beta):
    """The pair the CONSTRAINT uses, picked by LS_METHOD."""
    geo_s, geo_v, mor_s, mor_v = _length_scale_both(x, beta)
    return ((mor_s, mor_v) if LS_METHOD == "morphological"
            else (geo_s, geo_v))


def _scaled_length_scale(g, ls_on):
    """``g <= LS_BOUND`` as an O(1) constraint, switchable off.

    ``g - LS_BOUND`` would need a multiplier many orders above the objective's
    scale.  ``ls_on = 0`` returns a constant -1: satisfied, zero gradient, and
    no retrace when a stage turns the constraints on.
    """
    return ls_on * (g / LS_BOUND - 1.0) - (1.0 - ls_on)


@jax.jit
def value_and_grad(x, beta, ls_on):
    def _h(z):
        return heat(process(z, beta)) / heat_ref

    def _u(z):
        return stored_energy(mech_params(process(z, beta)))

    def _v(z):
        return jnp.sum(process(z, beta)[0] * design_cell) / n_design

    def _e(z):
        """Deformation per unit of compliance: how localised, not how much."""
        tp = mech_params(process(z, beta))
        return (strain_pnorm(tp)[0] / strain_ref) / stored_energy(tp)

    h, gh = jax.value_and_grad(_h)(x)
    c, gc = jax.value_and_grad(_u)(x)
    v, gv = jax.value_and_grad(_v)(x)
    e, ge = jax.value_and_grad(_e)(x)
    gs, gv_ls = _length_scale_raw(x, beta)
    geo_s, geo_v, mor_s, mor_v = _length_scale_both(x, beta)
    ls_jac = jax.jacrev(lambda z: jnp.stack(_length_scale_raw(z, beta)))(x)
    f = process(x, beta)
    spn, smax = stress_pnorm(mech_params(f))
    # a11 is nodal, rho is per element: average a11 onto the elements before
    # weighting by density.
    a11f = 0.5 * (f[1] + 1.0)
    a11_cell = jnp.mean(a11f[cells], axis=1)
    solid = f[0] * design_cell
    aux = dict(
        heat=h, compliance=c, volume=v, spn=spn, smax=smax, gs=gs, gv=gv_ls,
        geo_s=geo_s, geo_v=geo_v, mor_s=mor_s, mor_v=mor_v,
        strain=e,
        a11_struts=jnp.sum(a11_cell * solid) / jnp.maximum(jnp.sum(solid), 1e-12),
        a11_bands=jnp.sum(a11f * passive_mask) / jnp.sum(passive_mask),
        mnd=400.0 * jnp.mean(f[0] * (1.0 - f[0])))
    fval = jnp.array([c / ENERGY_ALLOW - 1.0,
                      e / STRAIN_ALLOW - 1.0,
                      _scaled_length_scale(gs, ls_on),
                      _scaled_length_scale(gv_ls, ls_on)])
    dfdx = jnp.stack([gc / ENERGY_ALLOW,
                      ge / STRAIN_ALLOW,
                      ls_on * ls_jac[0] / LS_BOUND,
                      ls_on * ls_jac[1] / LS_BOUND])
    return h + W_MASS * v, gh + W_MASS * gv, fval, dfdx, aux


@jax.jit
def values_only(x, beta, ls_on):
    """J and the constraints without gradients, for GCMMA's inner loop: the
    conservativity test needs values only, so no adjoint and no stress."""
    rho_tilde, f = design_fields(x, beta)
    h = heat(f) / heat_ref
    c = stored_energy(mech_params(f))
    v = jnp.sum(f[0] * design_cell) / n_design
    tp_e = mech_params(f)
    e = (strain_pnorm(tp_e)[0] / strain_ref) / stored_energy(tp_e)
    rho_sharp = jnp.where(passive_cell, 1.0, gene.heaviside_projection(
        rho_tilde, beta=LS_INDICATOR_BETA, threshold=0.5))
    gs, gv = (morph_length_scale(f[0]) if LS_METHOD == "morphological"
              else length_scale(rho_tilde, rho_sharp))
    return h + W_MASS * v, jnp.array([c / ENERGY_ALLOW - 1.0,
                                      e / STRAIN_ALLOW - 1.0,
                                      _scaled_length_scale(gs, ls_on),
                                      _scaled_length_scale(gv, ls_on)])


# ── GCMMA ──────────────────────────────────────────────────────────────────
history = {k: [] for k in ("iteration", "beta", "heat", "compliance", "volume",
                           "objective", "a11_struts", "a11_bands", "mnd",
                           "spn", "smax", "gs", "gv", "strain",
                           "geo_s", "geo_v", "mor_s", "mor_v")}
it = [0]
stage_ends: list = []

# [ rho on n_cells | t1 on n_nodes | t2 on n_nodes ], the orientation half
# present only when ANISOTROPY is on.
n_total = n_cells + (2 * n_nodes if ANISOTROPY else 0)
lower, upper = onp.full(n_total, -1.0), onp.ones(n_total)
lower[:n_cells] = 0.0
x_opt = onp.zeros(n_total)
x_opt[:n_cells] = 0.1
if PERTURB > 0.0:
    rng = onp.random.default_rng(SEED)
    x_opt[:n_cells] = onp.clip(
        x_opt[:n_cells] + 2.0 * PERTURB * (rng.random(n_cells) - 0.5), 0.0, 1.0)

# One optimiser for the whole run. beta is a traced ARGUMENT, not a closure
# constant, so the continuation stages reuse a single compiled step instead of
# retracing per stage.
# Per-constraint elastic penalty. MMA buys constraint violations at price c and
# abandons any constraint whose multiplier exceeds it. The objective here WANTS
# the bracket cut in two, so the physical constraints are priced high enough
# that they are never the cheap thing to give up; the geometric pair is a
# manufacturing preference and stays lower.
MMA_C = onp.array([MMA_C_COMPLIANCE, MMA_C_COMPLIANCE,
                   MMA_C_LENGTH_SCALE, MMA_C_LENGTH_SCALE])

opt = GCMMA(value_and_grad, lower, upper, n_constraints=4,
            value_fn=values_only, has_aux=True, c=MMA_C,
            max_inner=MMA_MAX_INNER, move=MMA_MOVE)
step = jax.jit(opt.step)


def record(state, beta, ls_on):
    """One history row, progress line and snapshot, off the state the optimiser
    already produced."""
    a = state.aux
    it[0] += 1
    h, c, v = float(a["heat"]), float(a["compliance"]), float(a["volume"])
    spn, smax = float(a["spn"]), float(a["smax"])
    a11s, a11b = float(a["a11_struts"]), float(a["a11_bands"])
    mnd, j = float(a["mnd"]), float(state.f0)
    gs, gv = float(a["gs"]), float(a["gv"])
    strain = float(a["strain"])
    geo_s, geo_v = float(a["geo_s"]), float(a["geo_v"])
    mor_s, mor_v = float(a["mor_s"]), float(a["mor_v"])
    for k, val in (("iteration", it[0]), ("beta", beta), ("heat", h),
                   ("compliance", c), ("volume", v), ("objective", j),
                   ("a11_struts", a11s), ("a11_bands", a11b), ("mnd", mnd),
                   ("spn", spn / stress_ref), ("smax", smax / stress_ref),
                   ("gs", gs), ("gv", gv), ("strain", strain),
                   ("geo_s", geo_s), ("geo_v", geo_v),
                   ("mor_s", mor_s), ("mor_v", mor_v)):
        history[k].append(val)
    if it[0] % PRINT_EVERY == 0 or it[0] == 1:
        print(f"  it {it[0]:4d} b={beta:4.1f} heat={h:.3e} "
              f"C={c:6.2f} V={v:.4f} "
              f"a11[str={a11s:.2f} bnd={a11b:.2f}] "
              f"grey={mnd:4.1f}% eps={strain:6.1f} "
              f"(lim {STRAIN_ALLOW:g})  J={j:.4f} "
              f"[h={h:.4f} m={W_MASS*v:.4f}] "
              f"ls[{'on ' if ls_on else 'off'} s={gs/LS_BOUND:6.2f} "
              f"v={gv/LS_BOUND:6.2f}] in={int(state.inner_iterations)}"
              + (f"  ABANDONED{onp.flatnonzero(onp.asarray(state.sol.y) > 1e-9).tolist()}"
                 if onp.any(onp.asarray(state.sol.y) > 1e-9) else ""))
    if it[0] % SNAPSHOT_EVERY == 0 or it[0] == 1:
        snapshot(onp.asarray(state.x), beta, dict(
            beta=beta, heat=h, compliance=c, volume=v, objective=j, mnd=mnd,
            a11_struts=a11s, a11_bands=a11b, spn=spn, smax=smax,
            gs=gs, gv=gv, strain=strain,
            geo_s=geo_s, geo_v=geo_v, mor_s=mor_s, mor_v=mor_v))
    return j, c


t0 = time.perf_counter()
state = None
for stage, beta in enumerate(HEAVISIDE_BETAS):
    beta_t = jnp.float64(beta)
    ls_on = 1.0 if stage >= LS_FROM_STAGE else 0.0
    ls_t = jnp.float64(ls_on)
    final = (stage == len(HEAVISIDE_BETAS) - 1)
    cap = ITER_PER_STAGE_FINAL if final else ITER_PER_STAGE[stage]
    window: collections.deque = collections.deque(maxlen=FINAL_WINDOW)
    stopped = None
    print(f"  -- stage {stage+1}/{len(HEAVISIDE_BETAS)}: beta = {beta:g}, "
          + (f"stop when J spreads < {FINAL_FTOL_REL:g} over "
             f"{FINAL_WINDOW} iterations (cap {cap}) --"
             if final else f"{cap} iterations --")
          + (f"  [min length scale ON, target {MIN_WIDTH:g} mm]" if ls_on
             else "  [min length scale off]"))
    # A new beta is a new problem, so the asymptotes restart from the warm
    # start. init() is the one evaluation there, and it is this stage's first
    # history row -- matching what an evaluation-counting driver reports.
    state = opt.init(jnp.asarray(x_opt), beta_t, ls_t)
    record(state, beta, ls_on)
    for _ in range(cap - 1):
        state = step(state, beta_t, ls_t)
        j, c = record(state, beta, ls_on)
        if final:
            margin = c / ENERGY_ALLOW - 1.0
            window.append(j)
            if len(window) == window.maxlen:
                spread = ((max(window) - min(window))
                          / max(abs(sum(window) / len(window)), 1e-12))
                if spread < FINAL_FTOL_REL and margin <= FINAL_FEAS_TOL:
                    stopped = (it[0], spread, margin)
                    break
    x_opt = onp.asarray(state.x)
    if final:
        if stopped is not None:
            n_it, spread, margin = stopped
            print(f"     converged at it {n_it}: J spread over the last "
                  f"{FINAL_WINDOW} iterations = {spread:.2e} < "
                  f"{FINAL_FTOL_REL:g}, constraint margin {margin:+.2e}")
        else:
            spread = ((max(window) - min(window))
                      / max(abs(sum(window) / len(window)), 1e-12)
                      if window else float("nan"))
            print(f"     NOT converged: hit the {cap}-evaluation cap with the "
                  f"J spread at {spread:.2e}, still above {FINAL_FTOL_REL:g} "
                  f"(KKT residual {float(state.kkt_norm):.2e})")
    stage_ends.append(it[0])
dt = time.perf_counter() - t0

# The last state already holds x_opt and everything measured there.
beta_f = float(HEAVISIDE_BETAS[-1])
_a = state.aux
h_f, c_f, v_f = (float(_a["heat"]), float(_a["compliance"]),
                 float(_a["volume"]))
spn_f, smax_f = float(_a["spn"]), float(_a["smax"])
a11s_f, a11b_f = float(_a["a11_struts"]), float(_a["a11_bands"])
mnd_f = float(_a["mnd"])
gs_f, gv_f = float(_a["gs"]), float(_a["gv"])
geo_s_f, geo_v_f = float(_a["geo_s"]), float(_a["geo_v"])
mor_s_f, mor_v_f = float(_a["mor_s"]), float(_a["mor_v"])
strain_f = float(_a["strain"])
snapshot(x_opt, beta_f, dict(
    beta=beta_f, heat=h_f, compliance=c_f, volume=v_f,
    objective=h_f + W_MASS * v_f, mnd=mnd_f,
    a11_struts=a11s_f, a11_bands=a11b_f, spn=spn_f, smax=smax_f,
    gs=gs_f, gv=gv_f, strain=strain_f,
    geo_s=geo_s_f, geo_v=geo_v_f, mor_s=mor_s_f, mor_v=mor_v_f))
_snap.close()

fields = [onp.asarray(f) for f in process(jnp.asarray(x_opt),
                                          jnp.float64(beta_f))]
rho = fields[0]                      # per element, what the physics saw
rho_node = cell_to_node(rho)         # per node, for plots and files

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
      f"solid (rho>0.5) {(rho > 0.5).mean()*100:.1f}% of elements")
print(f"deformation : {strain_f:.2f} x the concentration a solid plate has "
      f"at the same compliance (limit {STRAIN_ALLOW:g}, "
      f"{'SATISFIED' if strain_f <= STRAIN_ALLOW else 'VIOLATED'}) "
      f"-- this is what forbids a weak load path")
print(f"discreteness: {float(mnd_f):.1f}% non-discrete")
print(f"length scale: target {MIN_WIDTH:g} mm, constrained on "
      f"{LS_METHOD.upper()}")
print(f"  geometric   : g_s/eps = {geo_s_f/LS_EPS:8.2f}  "
      f"g_v/eps = {geo_v_f/LS_EPS:8.2f}   (stationary points only -- blind to "
      f"a tapering member)")
print(f"  morphologic : g_s/eps = {mor_s_f/LS_MORPH_EPS:8.2f}  "
      f"g_v/eps = {mor_v_f/LS_MORPH_EPS:8.2f}   (opening/closing of the "
      f"blueprint)")
_act = ((mor_s_f, mor_v_f, LS_MORPH_EPS) if LS_METHOD == "morphological"
        else (geo_s_f, geo_v_f, LS_EPS))
print(f"  -> {'SATISFIED' if max(_act[0], _act[1]) <= _act[2] else 'VIOLATED'}"
      f" (limit 1.00)")
print(f"stress      : p-norm {float(spn_f):.2f} MPa ({float(spn_f)/stress_ref:.2f}x), "
      f"peak {float(smax_f):.2f} MPa ({float(smax_f)/stress_ref:.2f}x of the "
      f"reference plate) -- reported only, not constrained")
print(f"orientation : a11 = 0 fibres along y (heat/load path), 1 along x "
      f"(isothermal, and the direction a band spreads load)")
print(f"              struts {float(a11s_f):.3f}   bands {float(a11b_f):.3f}")
_iy = onp.rint(pts[:, 1] / LY * NY).astype(int)
_a11n = 0.5 * (fields[1] + 1.0)
_rows = [(j / NY, float((_a11n[_iy == j] * rho_node[_iy == j]).sum()
                        / max(rho_node[_iy == j].sum(), 1e-12)))
         for j in range(0, NY + 1, max(1, NY // 20))]
print("                y   " + " ".join(f"{y:5.2f}" for y, _ in _rows))
print("                a11 " + " ".join(f"{a:5.2f}" for _, a in _rows))
print(f"evaluations : {it[0]}   wall time {dt:.1f}s")

# ── Outputs ────────────────────────────────────────────────────────────────
a2 = onp.asarray(jax.vmap(lambda a, b: tx.unit_trace_a2(a, b, SGN_BETA))(
    jnp.asarray(fields[1]), jnp.asarray(fields[2])))
director = onp.asarray(principal_direction(a2))
onp.save(os.path.join(OUTDIR, "density.npy"), rho.astype(onp.float32))
onp.save(os.path.join(OUTDIR, "density_nodal.npy"), rho_node.astype(onp.float32))
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
    tpc = axes[0].tripcolor(pts[:, 0], pts[:, 1], rho_node, cmap="gray_r",
                            vmin=0, vmax=1, shading="gouraud")
    sl = slice(None, None, 7)
    axes[0].quiver(pts[sl, 0], pts[sl, 1], director[sl, 0] * rho_node[sl],
                   director[sl, 1] * rho_node[sl], color="tab:orange", pivot="mid",
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

    # Fields the solve produces everywhere, masked by density before they are
    # written. The state equations are posed on the whole domain -- the void is
    # a very weak material, not an absence -- so a raw temperature or stress
    # field fills the void with values that are real solutions and completely
    # uninformative: the stress written here is the TRUE material stress
    # C0 : eps, with no rho weighting, so the void's unrestrained strain shows
    # up as a large stress exactly where there is nothing. Masked fields go out
    # under the plain names and the raw ones keep a `_raw` suffix.
    #
    # This is presentation only. The objective and the constraints are already
    # weighted by k(rho) and E(rho): deleting the structure and keeping only
    # the void leaves 0.4% of the heat and sends C/ref to 4e5.
    mask_n = onp.clip(rho_node, 0.0, 1.0)
    mask_c = onp.clip(rho, 0.0, 1.0)

    points = [("density", rho_node),
              ("temperature", T.reshape(-1) * mask_n),
              ("temperature_raw", T.reshape(-1)),
              ("a11", a2[:, 0, 0]), ("a12", a2[:, 0, 1]), ("k_yy", k_yy),
              ("director", onp.column_stack([director, onp.zeros(n_nodes)])
               * mask_n[:, None])]
    cells = []

    def _add(problem, solver, ts, label):
        sol = solver(tp_me, guess_u, traced_structure=ts)
        u = onp.asarray(sol.field(0))
        points.append((f"u_{label}",
                       onp.column_stack([u, onp.zeros(len(u))]) * mask_n[:, None]))
        sig, _ = tx.create_stress_tensor_fn(problem, true_stress)(sol, tp_me)
        sig = onp.asarray(sig).mean(axis=1)          # quadrature -> cell
        vm = onp.sqrt(onp.maximum(
            sig[:, 0, 0]**2 - sig[:, 0, 0] * sig[:, 1, 1] + sig[:, 1, 1]**2
            + 3.0 * sig[:, 0, 1]**2, 0.0))
        cells.extend([(f"vm_{label}", vm * mask_c),
                      (f"vm_{label}_raw", vm),
                      (f"sxx_{label}", sig[:, 0, 0] * mask_c),
                      (f"syy_{label}", sig[:, 1, 1] * mask_c),
                      (f"sxy_{label}", sig[:, 0, 1] * mask_c)])

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
