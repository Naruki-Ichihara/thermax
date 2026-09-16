"""2-D thermal isolation with fibre orientation as a design field.

    left/right edges   free and insulated -- a single standalone support
    bottom edge        T = T_HOT,  clamped,  passive solid band
    top edge           T = T_COLD, loaded,   passive solid band

    design per node    rho, t1, t2 (orientation a_2; a_4 from a closure)
    minimise           heat / heat_ref  +  W_MASS * V
                       + W_DRIFT * (drift / drift_ref)^2
    subject to         U / U_ref <= ENERGY_ALLOW

Stiffness is a requirement, not a preference, so it is a constraint and heat is
the objective. That also makes the operating point well defined: heat is
monotone increasing in density -- adding material can only add conductance --
so the design shrinks until the compliance constraint binds, and the constraint
is active by construction.

Mass is in the objective rather than capped by a constraint: a cap does not
minimise it, the design simply grows to meet it. And it has to be priced
somewhere, because heat alone does not price it once orientation is a design
field. Heat is monotone increasing in density AT FIXED ORIENTATION, but the
design has another move -- turn the fibres transverse, which cuts k_yy by
k_par/k_perp and E_yy by E1/E2, then thicken to put the stiffness back. That
lowers heat while RAISING volume; measured at the trunk root, 2.2x the thickness
against a 14x loss of axial modulus. Nothing in a heat-only objective stands in
the way of it.

V is the design-region volume fraction, so both terms sit around 0.05-0.10 at
the operating point and W_MASS reads directly as their relative importance.

Material
--------
Transversely isotropic lamina, orientation-averaged: ``a_2`` from ``t1, t2`` and
the closure-free ``a_4`` from ``t4, t5``. Conductivity is a second-order average
and depends on ``a_2`` alone; stiffness is a fourth-order average and depends on
both. So a fibre lying along an isotherm delivers E1 to whatever structural duty
that direction carries while presenting only k_perp to the heat.

Both interpolations are RAMP, with opposite signs of q, because grey is
penalised on whichever side of linear is unfavourable and that flips with the
direction of the objective: stiffness is wanted (q > 0), conductance is not
(q < 0). SIMP is wrong for the conductivity here -- it makes intermediate
density the cheapest insulator available and fills the design with grey -- and
wrong for the stiffness once the conductivity is RAMP, because dE/drho = 3 rho^2
vanishes at rho = 0 while dk/drho does not, which makes the void state a
one-way trap the compliance term cannot pull back from.

Passive bands
-------------
feax applies a surface traction to the boundary faces regardless of density, so
on a fully loaded edge the traction also lands on void, where E_MIN makes the
compliance integral meaningless and every concentration of material is punished.
The bands make the load land on material; they also model the shell walls a
skirt is welded between, which are solid and not designable.

Outputs in outputs/thermal_isolation_2d_aniso_C<..>_m<W_MASS>/.
"""
from __future__ import annotations

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
LX, LY = 1.0, 1.0
NX, NY = 100, 100
PASSIVE_THICKNESS = 0.05           # non-design solid band on each edge

# ── Material ───────────────────────────────────────────────────────────────
E1, E2, G12, NU12 = 20.0, 1.0, 0.5, 0.3      # CFRP-like lamina
# 20:1. Note this is PAST the threshold that decides which way fibres should
# point: aligning them with the load path wins only while E1/E2 > k_par/k_perp,
# and here 14 < 20, so the transverse orientation should now be preferred.
K_PAR, K_PERP = 20.0, 1.0
E_MIN, K_MIN = 1e-6, 1e-4
RAMP_Q_E = 4.0                     # stiffness, q > 0
RAMP_Q = -0.8                      # conductivity, q < 0
SGN_BETA = 10.0

T_HOT, T_COLD = 1.0, 0.0
# Thermal expansion of the lamina, orientation-averaged like the conductivity
# (both are second-order tensor properties, so both depend on a_2 alone). For a
# carbon lamina the two constants have OPPOSITE SIGNS, so orientation can
# reverse the direction of the contraction rather than merely scale it.
# T_REF is the stress-free (assembly) temperature: the hot side.
ALPHA_PAR, ALPHA_PERP = -4.0e-4, 2.0e-2
T_REF = T_HOT
# Scaled so the mechanical and thermal strain energies are comparable.
# U is quadratic in the traction, and at TRACTION = 1 the reference block gave
# U_mech = 6.07e-02 against U_thermal = 1.19e-04 -- the thermal term was 0.2% of
# the total and the cross term was negligible, so the coupling did nothing.
# sqrt(1.19e-4 / 6.07e-2) = 0.044 balances them; 0.05 leaves the mechanical
# term about 1.3x the thermal one.
TRACTION = 0.05
# (name, traction on the top edge, weight in the compliance term). Only cases
# with a positive weight are built and solved.
LOAD_CASES = (
    # NOTE THE SIGN. feax's surface map is added to the residual, so a POSITIVE
    # component acts in the NEGATIVE coordinate direction: (0, +T) pushes the
    # top edge DOWN. Verified on a plain isotropic cell -- (0, +1) gives
    # uy(top) = -4.44e-02 and (0, -1) gives +4.44e-02. Tension therefore needs
    # -TRACTION, which is what is written here.
    #
    # For pure linear elasticity the sign changes nothing: C, U and the optimum
    # are identical. It matters now only because of the thermal strain, which
    # pulls the top edge DOWN on cooldown. Tension OPPOSES that contraction;
    # compression follows it. The cross term in U changes sign accordingly.
    ("tension", (0.0, -TRACTION), 1.0),
    ("shear",   (TRACTION, 0.0), 0.0),
)

# ── Objective and constraint ───────────────────────────────────────────────
ENERGY_ALLOW = 5     # U / U_ref must stay below this
W_MASS = 2                      # weight on volume, against heat
# Weight on the THERMAL DRIFT: the mean u_y of the top edge in a load case with
# no traction at all, so the contraction is the only thing driving it. Squared
# and normalised by the reference block's drift, so the term is 1.0 there and 0
# for a structure that does not move on cooldown.
#
# This is the measure that survives when the structure is free to contract.
# Stored energy is not: with the periodic cell removed, U_thermal fell from
# 1.19e-04 to 1.85e-06 -- 98% of it had been the hoop stress of a cell not
# allowed to shrink in x -- leaving the thermal part at 1.1% of U. A bar
# contracting freely carries no stress, but it still moves.
#
# It also puts the orientation field in a genuine three-way conflict for the
# first time: heat wants the fibres transverse (k_yy = k_perp), drift wants them
# axial (alpha_yy = alpha_par, 1/50 of transverse), and so does stiffness
# (E_yy = E1).
W_DRIFT = 0

# ── Optimisation ───────────────────────────────────────────────────────────
N_FIELDS = 3                       # rho, t1, t2  (a_4 comes from a closure)
FILTER_RADIUS = 2.0 * LX / NX
FILTER_RADIUS_ORI = 2.0 * LX / NX
# Heaviside continuation. The projection is what drives the design to 0/1 -- the
# RAMP pair alone was measured to leave it at 68% non-discrete -- and the ladder
# is what makes it survivable, since a sharp projection cannot undo a large step.
HEAVISIDE_BETAS = (2, 4, 8)
ITER_PER_STAGE = 50
# An x-uniform design is a stationary point of this cell, so a uniform start
# leaves the symmetric branch only through round-off. A perturbation makes that
# transition reproducible; > 0 to use it.
SEED, PERTURB = 0, 0.0

# Stress is reported, not constrained: sigma = rho**q C0:eps with 0 < q < PENAL
# (q > 0 relaxes the singularity phenomenon), aggregated by a volume-weighted
# p-norm and corrected towards the true peak between iterations.
STRESS_P = 12.0
STRESS_RELAX_Q = 2.0
STRESS_ALPHA = 0.5

OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs",
    f"thermal_isolation_2d_aniso_C{ENERGY_ALLOW:g}_m{W_MASS:g}d0")
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

print(f"2-D thermal isolation   {NX}x{NY} QUAD4  {n_nodes:,} nodes")
print(f"  material E1/E2 = {E1/E2:.0f}, k_par/k_perp = {K_PAR/K_PERP:.0f}; "
      f"RAMP q_E = {RAMP_Q_E:g}, q_k = {RAMP_Q:g}")
print(f"  load cases " + ", ".join(f"{n} {v} w={w:g}" for n, v, w in CASES))
print(f"  design {N_FIELDS} x {n_nodes:,} = {N_FIELDS*n_nodes:,} variables; "
      f"{n_design:,} design nodes, passive {float(passive_mask.mean())*100:.1f}%")
print(f"  minimise heat/reference + {W_MASS:g}*V,  "
      f"C/reference <= {ENERGY_ALLOW:g}")

# ── Problems and solvers ───────────────────────────────────────────────────
thermal = tx.anisotropic_diffusion_problem(
    mesh, k_par=K_PAR, k_perp=K_PERP, k_min=K_MIN, ramp_q=RAMP_Q,
    sgn_beta=SGN_BETA, ele_type="QUAD4")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(bottom_edge, "all", T_HOT),
    fe.DirichletBCSpec(top_edge, "all", T_COLD),
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

stress_at_quad = tx.anisotropic_stress_measure(
    E1=E1, E2=E2, G12=G12, nu12=NU12, q=STRESS_RELAX_Q,
    alpha_par=ALPHA_PAR, alpha_perp=ALPHA_PERP, t_ref=T_REF, sgn_beta=SGN_BETA)
mech, solvers, energies, vonmises_fns = [], [], [], []
bc_me = ts_me = None
for name, tvec, _w in CASES:
    p = tx.anisotropic_elasticity_problem(
        mesh, E1=E1, E2=E2, G12=G12, nu12=NU12, e_min=E_MIN, ramp_q=RAMP_Q_E,
        alpha_par=ALPHA_PAR, alpha_perp=ALPHA_PERP, t_ref=T_REF,
        sgn_beta=SGN_BETA, ele_type="QUAD4",
        location_fns=(top_edge,), tractions=(tvec,))
    if bc_me is None:
        bc_me = fe.DirichletBCConfig([
            fe.DirichletBCSpec(bottom_edge, "all", 0.0)]).create_bc(p)
        ts_me = fe.TracedStructure.from_problem(p)
    mech.append(p)
    solvers.append(fe.create_solver(
        p, bc_me, solver_options=direct, adjoint_solver_options=direct,
        linear=True, traced_structure=ts_me,
        traced_params=fe.TracedParams(volume_vars=(_o, _z, _z, _z))))
    energies.append(fe.create_energy_fn(p))
    vonmises_fns.append(tx.create_vonmises_fn(p, stress_at_quad))
# A mechanical problem with NO traction: the thermal strain in the energy
# density is then the only load, so its solution is the pure cooldown response.
drift_problem = tx.anisotropic_elasticity_problem(
    mesh, E1=E1, E2=E2, G12=G12, nu12=NU12, e_min=E_MIN, ramp_q=RAMP_Q_E,
    alpha_par=ALPHA_PAR, alpha_perp=ALPHA_PERP, t_ref=T_REF,
    sgn_beta=SGN_BETA, ele_type="QUAD4")
ts_dr = fe.TracedStructure.from_problem(drift_problem)
solve_drift = fe.create_solver(
    drift_problem, bc_me, solver_options=direct, adjoint_solver_options=direct,
    linear=True, traced_structure=ts_dr,
    traced_params=fe.TracedParams(volume_vars=(_o, _z, _z, _z)))
guess_u = fe.zero_like_initial_guess(mech[0], bc_me)
_top_w = jnp.asarray((pts[:, 1] >= LY - TOL).astype(float))

filter_rho = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS)
filter_ori = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS_ORI)


# ── Design fields and responses ────────────────────────────────────────────
def process(x, beta):
    """Raw design vector -> physical (rho, t1, t2, t4, t5).

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


def solve_thermal(fields):
    """Returns (heat, mechanical TracedParams). The solved T rides along the
    design fields as a fourth volume variable -- the standard feax bridge for
    staggered multiphysics -- and the chain rho -> T -> u stays differentiable."""
    tp_th = fe.TracedParams(volume_vars=fields)
    sol_T = solve_T(tp_th, guess_T, traced_structure=ts_th)
    T = fe.TracedParams.node_var_from_solution(thermal, sol_T)
    return conduction_energy(sol_T, tp_th), fe.TracedParams(
        volume_vars=fields + (T,))


def stored_energy(tp_me):
    """Weighted MEAN over load cases of U = 1/2 int eps_el : C : eps_el, each
    normalised by its own reference, so the reference design scores 1.0.

    U replaces compliance because ``int t.u`` stops meaning anything once there
    is a thermal strain: it is no longer twice the strain energy, and it cannot
    see thermal self-stress at all. U reduces to 1/2 the compliance for pure
    mechanical loading and to the thermal self-energy with the traction removed,
    and lower is better in both directions."""
    return sum(w * energies[i](
        solvers[i](tp_me, guess_u, traced_structure=ts_me).flat, tp_me)
        / energy_ref[i] for i, (_n, _v, w) in enumerate(CASES)) / W_SUM


def thermal_drift_sq(tp_me):
    """Mean SQUARED ``u_y`` of the top edge under cooldown alone -- the square of
    its RMS axial motion, which is what the objective wants.

    Squared before averaging, not after. The earlier measure was the mean of
    ``u_y``, and a mean cancels: an edge that tilts or bows scores the same zero
    as one that does not move at all. An RMS cannot cancel, so it sees the
    tilting and bowing that actually distort the inner shell's mounting face.

    ``u_x`` is deliberately EXCLUDED. It is not squeamishness about the extra
    term -- including it puts a floor under the objective that no design can get
    past. The expansion tensor's trace is invariant under orientation::

        trace(alpha) = 2 alpha_perp + (alpha_par - alpha_perp) trace(a2)
                     = alpha_par + alpha_perp        (trace(a2) = 1)

    so ``alpha_xx + alpha_yy`` is a material constant and the orientation field
    can only TRADE one against the other: driving ``alpha_yy`` to ``alpha_par``
    forces ``alpha_xx`` to ``alpha_perp``. Density is no help either, since the
    displacement of a freely contracting body does not depend on its stiffness.
    Measured, that floor is real: the same run scored RMS 0.047 of the reference
    on ``u_y`` alone and 0.466 with ``u_x`` included, and the two-component
    version left the design 65% grey with the compliance constraint slack --
    the optimiser thrashing against a bound it cannot move.

    And the lateral term is not a requirement anyway. The vertical edges are
    free, so the skirt's own transverse shrinkage is unstressed and does not
    displace the shell it carries.

    Returned squared rather than as the RMS itself because ``sqrt`` is not
    smooth at zero, and zero is exactly where a successful design is heading.
    Normalised by the reference block's value, the term is then
    ``(RMS / RMS_ref)^2`` -- 1.0 at the reference, 0 for a structure that does
    not move on cooldown."""
    u = solve_drift(tp_me, guess_u, traced_structure=ts_dr).field(0)
    return jnp.sum(u[:, 1]**2 * _top_w) / jnp.sum(_top_w)


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
# The reference block is solved the same way: its own thermal solve supplies
# the T that its own thermal strain uses, so U_ref already contains the
# contraction the reference itself suffers.
_h_ref_raw, _tp_ref = solve_thermal(ref)
heat_ref = float(_h_ref_raw)
energy_ref = [float(energies[i](
    solvers[i](_tp_ref, guess_u, traced_structure=ts_me).flat, _tp_ref))
    for i in range(len(CASES))]
stress_ref = float(stress_pnorm(_tp_ref)[1])
drift_ref = float(thermal_drift_sq(_tp_ref))      # mean squared, not RMS
_c_pn = [1.0]
print(f"  reference: heat={heat_ref:.4e}  "
      + "  ".join(f"U_{n}={c:.4e}" for (n, _v, _w), c in zip(CASES, energy_ref))
      + f"  peak von Mises={stress_ref:.4e}  drift RMS={drift_ref**0.5:.4e}")
print("-" * 78)


@jax.jit
def evaluate(x, beta):
    def _h(z):
        return solve_thermal(process(z, beta))[0] / heat_ref

    def _u(z):
        return stored_energy(solve_thermal(process(z, beta))[1])

    def _d(z):
        return thermal_drift_sq(solve_thermal(process(z, beta))[1]) / drift_ref

    h, gh = jax.value_and_grad(_h)(x)
    c, gc = jax.value_and_grad(_u)(x)
    dr, gdr = jax.value_and_grad(_d)(x)
    v, gv = jax.value_and_grad(
        lambda z: jnp.sum(process(z, beta)[0] * design_mask) / n_design)(x)
    f = process(x, beta)
    spn, smax = stress_pnorm(solve_thermal(f)[1])
    a11f = 0.5 * (f[1] + 1.0)
    solid = f[0] * design_mask
    return (h, gh, c, gc, v, gv, dr, gdr, spn, smax,
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
                           "drift", "objective", "a11_struts", "a11_bands", "mnd", "spn", "smax")}
it = [0]


def objective(xx, grad):
    h, gh, c, gc, v, gv, dr, gdr, spn, smax, a11s, a11b, mnd = _cached(xx)
    if grad.size > 0:
        grad[:] = gh + W_MASS * gv + W_DRIFT * gdr
    it[0] += 1
    j = float(h) + W_MASS * float(v) + W_DRIFT * float(dr)
    # Refresh the p-norm correction once per outer iteration, from the previous
    # point, so nothing non-stationary reaches MMA's surrogate.
    _c_pn[0] = tx.adaptive_scale(_c_pn[0], float(smax), float(spn),
                                 alpha=STRESS_ALPHA)
    for k, val in (("iteration", it[0]), ("beta", _beta[0]),
                   ("heat", float(h)), ("compliance", float(c)),
                   ("volume", float(v)), ("drift", float(dr)),
                   ("objective", j),
                   ("a11_struts", float(a11s)), ("a11_bands", float(a11b)),
                   ("mnd", float(mnd)), ("spn", float(spn) / stress_ref),
                   ("smax", float(smax) / stress_ref)):
        history[k].append(val)
    if it[0] % 10 == 0 or it[0] == 1:
        print(f"  it {it[0]:4d} b={_beta[0]:4.1f} heat={float(h):.3e} "
              f"C={float(c):6.2f} V={float(v):.4f} "
              f"a11[str={float(a11s):.2f} bnd={float(a11b):.2f}] "
              f"grey={float(mnd):4.1f}% sMAX={float(smax)/stress_ref:6.1f} "
              f"(lim {ENERGY_ALLOW:g})  J={j:.4f} "
              f"[h={float(h):.4f} m={W_MASS*float(v):.4f} "
              f"d={W_DRIFT*float(dr):.4f}]  drift={float(dr)**0.5:.3f}x")
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
    opt = nlopt.opt(nlopt.LD_MMA, n_total)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(objective)
    opt.add_inequality_constraint(compliance_constraint, 1e-8)
    opt.set_maxeval(ITER_PER_STAGE)
    print(f"  -- stage {stage+1}/{len(HEAVISIDE_BETAS)}: beta = {beta:g} --")
    x_opt = opt.optimize(x_opt)
dt = time.perf_counter() - t0

fields = [onp.asarray(f) for f in process(jnp.asarray(x_opt),
                                          jnp.float64(_beta[0]))]
rho = fields[0]
(h_f, _, c_f, _, v_f, _, dr_f, _, spn_f, smax_f,
 a11s_f, a11b_f, mnd_f) = _cached(x_opt)

print("-" * 78)
print(f"objective   : {history['objective'][0]:.4f} -> "
      f"{history['objective'][-1]:.4f}   "
      f"[heat {float(h_f):.4f}  mass {W_MASS*float(v_f):.4f}  "
      f"drift term {W_DRIFT*float(dr_f):.4f}]")
print(f"thermal drift: RMS {float(dr_f)**0.5:.4f} x the reference block's "
      f"({drift_ref**0.5:.3e}).  Reported whatever W_DRIFT is, so a W_DRIFT = 0 "
      f"control still says what the drift cost would have been.")
print(f"heat        : {history['heat'][0]:.4e} -> {float(h_f):.4e} x reference")
print(f"compliance  : {float(c_f):.3f} x reference (limit {ENERGY_ALLOW:g}, "
      f"{'ACTIVE' if float(c_f) > 0.98 * ENERGY_ALLOW else 'SLACK'})")
print(f"volume      : design {float(v_f):.4f}   whole cell {rho.mean():.4f}   "
      f"solid (rho>0.5) {(rho > 0.5).mean()*100:.1f}%")
print(f"discreteness: {float(mnd_f):.1f}% non-discrete")
print(f"stress      : p-norm {float(spn_f)/stress_ref:.2f}x, peak "
      f"{float(smax_f)/stress_ref:.2f}x of reference (reported only)")
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
    for s in range(1, len(HEAVISIDE_BETAS)):
        axes[1].axvline(s * ITER_PER_STAGE, color="0.8", lw=0.8)
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
    # the TRUE material stress (q = 0), not the relaxed measure the constraint
    # uses -- the relaxed one is a numerical device and would mislead here.
    jf = tuple(jnp.asarray(f) for f in fields)
    _, tp_me = solve_thermal(jf)
    T = onp.asarray(jnp.asarray(tp_me.volume_vars[3]))
    a2j = jnp.asarray(a2)
    k_yy = onp.asarray(jax.vmap(
        lambda a: tx.orientation_conductivity(a, K_PAR, K_PERP)[1, 1])(a2j))
    a_yy = onp.asarray(jax.vmap(
        lambda a: tx.orientation_cte(a, ALPHA_PAR, ALPHA_PERP)[1, 1])(a2j))
    true_stress = tx.anisotropic_stress_measure(
        E1=E1, E2=E2, G12=G12, nu12=NU12, q=0.0, alpha_par=ALPHA_PAR,
        alpha_perp=ALPHA_PERP, t_ref=T_REF, sgn_beta=SGN_BETA)

    points = [("density", rho), ("temperature", T.reshape(-1)),
              ("a11", a2[:, 0, 0]), ("a12", a2[:, 0, 1]),
              ("k_yy", k_yy), ("alpha_yy", a_yy),
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
    # the cooldown-only case: no traction, so this is the drift field itself
    _add(drift_problem, solve_drift, ts_dr, "cooldown")

    save_sol(mesh, os.path.join(OUTDIR, "final.vtu"),
             point_infos=points, cell_infos=cells)
except Exception as e:
    print(f"(no vtu: {e})")

print(f"wrote density.npy / orientation.npy / history.csv / design.png / "
      f"final.vtu to {os.path.relpath(OUTDIR, os.getcwd())}/")
