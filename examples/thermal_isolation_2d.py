"""2-D thermal isolation: block the heat, keep holding the load.

    bottom patch   T = T_HOT,  and mechanically CLAMPED
    top patch      T = T_COLD, and pulled straight up (TENSILE traction)
    everything else  insulated, traction-free

    minimise    heat leaking from the hot patch to the cold one
    subject to  mechanical compliance <= ALLOW x (compliance of the solid design)
                0 <= rho <= 1

The simplest member of the thermal-isolation family: one domain, two terminals,
one load case, no periodicity, no narrow band. It is also the one that shows the
family's central difficulty most clearly, so it is worth knowing what to expect
before reading the picture it produces.

What is, and is not, degenerate here
------------------------------------
One invariance is exact and worth stating first, because it decides how the
answer looks. Scale the whole design, ``rho -> s rho``. With ``k`` and ``E`` both
interpolated as ``rho^3``, conductivity and modulus scale by the same ``s^3``,
so the temperature and displacement fields are unchanged up to a constant and

    heat x compliance        is INVARIANT.

The run confirms it to four figures: the uniform ``rho = 0.5`` start reports
``heat/solid = 0.125`` and ``C/C_solid = 8.000``, product ``1.000`` -- exactly the
product of the solid block it was scaled from. Two consequences:

* there is no reward for simply using less material, which is why this problem
  needs no volume constraint and why the answer does NOT come out thin;
* ``heat x compliance`` is the honest figure of merit. Raw heat can be driven
  anywhere by moving ``COMPLIANCE_ALLOW``; the product cannot.

The 1-D version of the argument goes further, and that is the part to distrust.
Along a load path of varying section ``A(l)``, thermal resistance
``integral dl/(kA)`` and axial compliance ``integral dl/(EA)`` are the same
integral, so a NETWORK OF BARS has stiffness/conductance ``= E_SOLID/K_SOLID``
whatever its topology -- taper, meander, parallel paths, all identical -- and
topology optimization would have nothing to do at all.

That does not survive in a 2-D continuum. The elastic operator carries shear and
a Poisson coupling with no scalar-diffusion counterpart, and the optimiser finds
the gap: at ``COMPLIANCE_ALLOW = 20`` it reaches

    heat x compliance = 0.596      (1.68x better than any uniform scaling)

So the shape freedom is real, if modest. The bar argument is a bound on the
wrong problem, not a prediction for this one.

What the picture looks like, and why
-------------------------------------
Not a torn structure, and not a thin neck. Tearing sends the compliance to
infinity and the constraint forbids it, while thinning buys nothing because of
the invariance above -- so the optimiser spends its freedom on the WIDTH PROFILE
of a fat column, sized so the compliance lands exactly on the limit. Volume
barely moves off ``RHO_INIT``.

It also comes out GREY (here: mean density 0.50, but only 36% of nodes above
0.5). With ``PENAL_K == PENAL_E`` the invariance holds pointwise as well, so
intermediate density is never penalised on the objective/constraint trade and
nothing pushes the design towards 0/1.

Knobs
-----
* ``COMPLIANCE_ALLOW`` -- the stiffness budget, as a multiple of the solid
  block's compliance. It sets the scale of the answer, not its shape.
* ``PENAL_K != PENAL_E`` -- breaks the invariance, including the pointwise one.
  Lowering ``PENAL_K`` makes intermediate density conduct more than it stiffens,
  and the optimiser starts preferring crisp solid-or-void.
* A Heaviside projection on the filtered density -- the direct way to force 0/1
  without touching the physics.
* A load with bending or shear in it. Resistance to a lateral load scales with
  ``A d^2`` and conduction only with ``A``, so material moved away from the axis
  buys stiffness superlinearly. This is the big lever, and it is why real
  cryogenic supports are thin-walled and spread out rather than solid posts.

Optimiser: MMA, not OC. ``dE/drho > 0`` here -- more material always leaks more
heat -- and OC's multiplicative update only fires for the other sign; it would
drive every cell to the lower bound and disconnect the structure outright.

Outputs (in outputs/thermal_isolation_2d/): density.npy, history.csv,
density.png, final.vtu.
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
from feax.utils import save_sol          # not re-exported from feax's top level

import thermax as tx

# ── Domain ─────────────────────────────────────────────────────────────────
LX, LY = 1.0, 1.0
NX, NY = 100, 100
PATCH_FRAC = 0.2                   # patch width / LX, centred on each edge

# ── Materials ──────────────────────────────────────────────────────────────
K_SOLID, K_MIN = 1.0, 1e-4         # heat enters only through the Dirichlet
                                   # patches, so K_MIN is regularisation only
E_SOLID, E_MIN, NU = 1.0, 1e-6, 0.3
PENAL_K, PENAL_E = 3.0, 3.0        # equal on purpose -- see the docstring

T_HOT, T_COLD = 1.0, 0.0
TRACTION = 1.0                     # pulling +y on the top patch

# ── Optimisation ───────────────────────────────────────────────────────────
FILTER_RADIUS = 2.0 * LX / NX      # Helmholtz filter, ~2 elements
COMPLIANCE_ALLOW = 20.0             # C* = ALLOW x C(rho = 1)
RHO_INIT = 0.5
MAX_ITER = 150

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "outputs", "thermal_isolation_2d")
os.makedirs(OUTDIR, exist_ok=True)

TOL = 1e-6 * max(LX, LY)


def _patch(y_edge):
    return lambda p: (jnp.isclose(p[1], y_edge, atol=TOL)
                      & (jnp.abs(p[0] - 0.5 * LX) <= 0.5 * PATCH_FRAC * LX + TOL))


bottom_patch = _patch(0.0)
top_patch = _patch(LY)

# ── Mesh ───────────────────────────────────────────────────────────────────
mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                              ele_type="QUAD4")
n_nodes = mesh.points.shape[0]
print(f"2-D thermal isolation  {NX}x{NY} QUAD4  "
      f"{n_nodes:,} nodes, {mesh.cells.shape[0]:,} cells")
print(f"  patches: central {PATCH_FRAC*100:.0f}% of the bottom (T={T_HOT}, "
      f"clamped) and top (T={T_COLD}, pulled +y) edges")
print(f"  SIMP p_k={PENAL_K:g}, p_E={PENAL_E:g}"
      + ("  <- EQUAL: heat x compliance is scale-invariant, expect a grey "
         "fat column" if PENAL_K == PENAL_E
         else "  <- unequal: scale invariance broken"))

# ── Thermal: two terminals, everything else insulated ──────────────────────
thermal = tx.diffusion_problem(mesh, dim=2, ele_type="QUAD4")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(bottom_patch, "all", T_HOT),
    fe.DirichletBCSpec(top_patch, "all", T_COLD),
]).create_bc(thermal)

# ── Mechanical: clamped at the cold patch, pulled at the hot one ───────────
mech = tx.elasticity_problem(
    mesh, dim=2, nu=NU, ele_type="QUAD4",
    location_fns=(top_patch,),
    tractions=((0.0, TRACTION),),
)
bc_me = fe.DirichletBCConfig([
    fe.DirichletBCSpec(bottom_patch, "all", 0.0),
]).create_bc(mech)

# ── Solvers (both linear, both differentiable) ─────────────────────────────
ts_th = fe.TracedStructure.from_problem(thermal)
ts_me = fe.TracedStructure.from_problem(mech)
sample_th = fe.TracedParams(volume_vars=(jnp.full((n_nodes,), K_SOLID),))
sample_me = fe.TracedParams(volume_vars=(jnp.full((n_nodes,), E_SOLID),))
direct = fe.DirectSolverOptions()

solve_T = fe.create_solver(thermal, bc_th, solver_options=direct,
                           adjoint_solver_options=direct, linear=True,
                           traced_params=sample_th, traced_structure=ts_th)
solve_u = fe.create_solver(mech, bc_me, solver_options=direct,
                           adjoint_solver_options=direct, linear=True,
                           traced_params=sample_me, traced_structure=ts_me)
guess_T = fe.zero_like_initial_guess(thermal, bc_th)
guess_u = fe.zero_like_initial_guess(mech, bc_me)

conduction_energy = tx.create_conduction_energy_fn(thermal)
compliance_fn = gene.create_compliance_fn(mech)
filter_fn = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS)


# ── Responses ──────────────────────────────────────────────────────────────
def heat_leak(rho):
    """Conduction energy; proportional to the heat crossing the cell."""
    tp = fe.TracedParams(volume_vars=(
        tx.simp_conductivity(rho, K_SOLID, K_MIN, PENAL_K),))
    return conduction_energy(solve_T(tp, guess_T, traced_structure=ts_th), tp)


def compliance(rho):
    tp = fe.TracedParams(volume_vars=(
        tx.simp_modulus(rho, E_SOLID, E_MIN, PENAL_E),))
    return compliance_fn(solve_u(tp, guess_u, traced_structure=ts_me))


@jax.jit
def evaluate(x):
    """Both responses and both gradients, from one filtered design.

    The two solves are independent -- the thermal one never touches ``u`` and the
    mechanical one never touches ``T`` -- so putting them under one ``jit``
    costs nothing and spares nlopt a second forward pass per iteration.
    """
    rho = filter_fn(x)
    h, gh = jax.value_and_grad(heat_leak)(rho)
    c, gc = jax.value_and_grad(compliance)(rho)
    # chain the filter back onto the raw design variables
    _, vjp = jax.vjp(filter_fn, x)
    return h, vjp(gh)[0], c, vjp(gc)[0]


# ── Reference: the all-solid design ────────────────────────────────────────
solid = jnp.ones((n_nodes,))
heat_solid = float(heat_leak(solid))
comp_solid = float(compliance(solid))
comp_limit = COMPLIANCE_ALLOW * comp_solid
print(f"  solid reference: heat={heat_solid:.4e}  compliance={comp_solid:.4e}")
print(f"  compliance limit C* = {COMPLIANCE_ALLOW:g} x C_solid = "
      f"{comp_limit:.4e}")
print(f"  floor set by K_MIN alone (rho=0 everywhere): "
      f"{float(heat_leak(jnp.zeros((n_nodes,))))/heat_solid:.3e} x solid")
print("-" * 68)

# ── MMA ────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _cached(xx):
    key = xx.tobytes()
    if key not in _cache:
        _cache.clear()
        _cache[key] = [onp.asarray(a) for a in evaluate(jnp.asarray(xx))]
    return _cache[key]


history = {"iteration": [], "heat": [], "compliance": [], "volume": [],
           "product": []}
it = [0]


def objective(xx, grad):
    h, gh, c, _ = _cached(xx)
    if grad.size > 0:
        grad[:] = gh / heat_solid
    it[0] += 1
    vol = float(onp.mean(filter_fn(jnp.asarray(xx))))
    h_rel, c_rel = float(h) / heat_solid, float(c) / comp_solid
    history["iteration"].append(it[0])
    history["heat"].append(h_rel)
    history["compliance"].append(c_rel)
    history["volume"].append(vol)
    # Invariant under rho -> s rho, so this is the part the SHAPE earned; 1.0 is
    # any uniform scaling of the solid block, and lower is better.
    history["product"].append(h_rel * c_rel)
    if it[0] % 5 == 0 or it[0] == 1:
        print(f"  it {it[0]:4d}  heat/solid={h_rel:.4e}  "
              f"C/C_solid={c_rel:7.3f} (limit {COMPLIANCE_ALLOW:g})  "
              f"vol={vol:.3f}  heatxC={h_rel * c_rel:.3f}")
    return h_rel


def compliance_constraint(xx, grad):
    _, _, c, gc = _cached(xx)
    if grad.size > 0:
        grad[:] = gc / comp_limit
    return float(c) / comp_limit - 1.0


opt = nlopt.opt(nlopt.LD_MMA, n_nodes)
opt.set_lower_bounds(onp.zeros(n_nodes))
opt.set_upper_bounds(onp.ones(n_nodes))
opt.set_min_objective(objective)
opt.add_inequality_constraint(compliance_constraint, 1e-8)
opt.set_maxeval(MAX_ITER)

t0 = time.perf_counter()
x_opt = opt.optimize(onp.full(n_nodes, RHO_INIT))
dt = time.perf_counter() - t0

rho_final = onp.asarray(filter_fn(jnp.asarray(x_opt)))
h_fin, _, c_fin, _ = _cached(x_opt)

print("-" * 68)
print(f"heat leak   : {history['heat'][0]:.4e} -> {float(h_fin)/heat_solid:.4e} "
      f"x solid   (/{history['heat'][0]/max(float(h_fin)/heat_solid, 1e-30):.1f})")
print(f"compliance  : {float(c_fin)/comp_solid:.3f} x solid   "
      f"(limit {COMPLIANCE_ALLOW:g}, "
      f"{'ACTIVE' if float(c_fin) > 0.98 * comp_limit else 'slack'})")
print(f"volume      : {rho_final.mean():.3f}   solid fraction "
      f"(rho>0.5): {(rho_final > 0.5).mean()*100:.1f}%")
prod = (float(h_fin) / heat_solid) * (float(c_fin) / comp_solid)
print(f"heat x C    : {history['product'][0]:.3f} -> {prod:.3f}   "
      f"({1.0/max(prod, 1e-30):.2f}x better than any uniform scaling of the "
      f"solid block)")
print(f"              this is the scale-invariant figure of merit: what the "
      f"SHAPE earned, independent of COMPLIANCE_ALLOW")
print(f"evaluations : {it[0]}   wall time {dt:.1f}s")

# ── Outputs ────────────────────────────────────────────────────────────────
onp.save(os.path.join(OUTDIR, "density.npy"), rho_final.astype(onp.float32))
with open(os.path.join(OUTDIR, "history.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["iteration", "heat_over_solid", "compliance_over_solid",
                "volume", "heat_times_compliance"])
    for i in range(len(history["iteration"])):
        w.writerow([history["iteration"][i], history["heat"][i],
                    history["compliance"][i], history["volume"][i],
                    history["product"][i]])

pts = onp.asarray(mesh.points)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    tp = axes[0].tripcolor(pts[:, 0], pts[:, 1], rho_final,
                           cmap="gray_r", vmin=0.0, vmax=1.0, shading="gouraud")
    axes[0].set_title("density")
    axes[0].set_aspect("equal")
    fig.colorbar(tp, ax=axes[0])

    axes[1].semilogy(history["iteration"], history["heat"], "r-",
                     label="heat / solid (objective)")
    axes[1].semilogy(history["iteration"], history["product"], "k-",
                     label="heat x C (shape only)")
    axes[1].axhline(1.0, color="0.6", ls=":", label="uniform scaling")
    axes[1].set_xlabel("evaluation")
    axes[1].set_ylabel("relative to solid block")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("objective")

    axes[2].plot(history["iteration"], history["compliance"], "b-",
                 label="C / C_solid")
    axes[2].axhline(COMPLIANCE_ALLOW, color="r", ls="--", label="limit")
    axes[2].set_xlabel("evaluation")
    axes[2].set_ylabel("compliance / solid")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("constraint")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "density.png"), dpi=150)
    plt.close()
except ImportError:
    print("(no matplotlib: skipped density.png)")

# Final fields, for a look at where the heat actually goes.
try:
    rho_j = jnp.asarray(rho_final)
    tp_th = fe.TracedParams(volume_vars=(
        tx.simp_conductivity(rho_j, K_SOLID, K_MIN, PENAL_K),))
    tp_me = fe.TracedParams(volume_vars=(
        tx.simp_modulus(rho_j, E_SOLID, E_MIN, PENAL_E),))
    T = onp.asarray(solve_T(tp_th, guess_T, traced_structure=ts_th).field(0))
    u = onp.asarray(solve_u(tp_me, guess_u, traced_structure=ts_me).field(0))
    save_sol(mesh, os.path.join(OUTDIR, "final.vtu"),
                point_infos=[("density", rho_final),
                             ("temperature", T.reshape(-1)),
                             ("displacement", onp.column_stack(
                                 [u, onp.zeros(len(u))]))])
except Exception as e:
    print(f"(no vtu: {e})")

print(f"wrote density.npy / history.csv / density.png / final.vtu to "
      f"{os.path.relpath(OUTDIR, os.getcwd())}/")
