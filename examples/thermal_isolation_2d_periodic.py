"""2-D thermal isolation, periodic cell, tension AND shear.

One cell of a circumferential array of supports -- the tank skirt problem
unrolled, with the hoop direction laid along x:

    left/right edges   PERIODIC (the neighbouring supports)
    bottom patch       T = T_HOT,  and mechanically CLAMPED
    top patch          T = T_COLD, and loaded
    everything else    insulated, traction-free

    load cases   tension (0, +T) and shear (+T, 0), weighted 1:1
    minimise     heat leaking from the hot patch to the cold one
    subject to   0.5 C_tension/C_tension_solid + 0.5 C_shear/C_shear_solid
                   <= COMPLIANCE_ALLOW
                 0 <= rho <= 1

Why this cell and these two load cases
--------------------------------------
``thermal_isolation_2d.py`` -- insulated sides, tension only -- is degenerate by
construction, and measured out that way: ``heat x compliance`` fell only to
0.596, i.e. shape bought 1.68x and nothing else was available. Two reasons, both
addressed here.

* **Tension has no geometric lever.** Axial stiffness is ``EA`` with no
  dependence on where the material sits; thermal conductance is also ``kA``. So
  moving material around cannot change the ratio. SHEAR is different: resistance
  to a lateral load scales with the second moment ``~A d^2`` while conduction
  scales with ``A``, so material moved away from the axis buys stiffness
  superlinearly. That asymmetry is the whole design space, and a tension-only
  problem does not have it.

* **An isolated cell has nothing to brace against.** Under the periodic BC the
  cell can hold structure that reacts against its neighbours rather than against
  the load path -- the unrolled equivalent of a hoop or ring stiffener. In an
  AXISYMMETRIC thermal field the hoop direction is isothermal, so such members
  conduct no heat; whether they carry load is the question this run answers.

  The catch, worth stating before reading the result: under a load that is
  uniform across the array, periodicity makes every cell deform identically, so
  a member spanning one period has the same displacement at both ends and
  carries no force either. Free, but useless. Shear is included precisely
  because it is the load case that can break that symmetry within the cell.

Reading the result: heat x compliance
--------------------------------------
Under ``rho -> s rho`` both ``k`` and ``E`` scale by ``s^3``, so heat scales by
``s^3``, every compliance by ``s^-3``, and their product is INVARIANT. The
product is therefore the scale-free figure of merit -- what the SHAPE earned,
independent of ``COMPLIANCE_ALLOW``. It is 1.0 for the solid block and for any
uniform scaling of it. The tension-only, non-periodic baseline reached 0.596.
Anything materially below that is the new load case and the periodicity paying
off; anything near it is not.

Outputs (in outputs/thermal_isolation_2d_periodic/): density.npy, history.csv,
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
import feax.flat as flat
import feax.gene as gene
from feax.flat.pbc import PeriodicPairing
from feax.utils import save_sol          # not re-exported from feax's top level

import thermax as tx

# ── Domain ─────────────────────────────────────────────────────────────────
LX, LY = 1.0, 1.0                  # x is the periodic ("hoop") direction
NX, NY = 100, 100
PATCH_FRAC = 0.2                   # patch width / LX, centred on each edge

# ── Materials ──────────────────────────────────────────────────────────────
K_SOLID, K_MIN = 1.0, 1e-4
E_SOLID, E_MIN, NU = 1.0, 1e-6, 0.3
PENAL_K, PENAL_E = 3.0, 3.0

T_HOT, T_COLD = 1.0, 0.0
TRACTION = 1.0

# (name, traction vector on the top patch, weight). Weights are applied to each
# load case's compliance NORMALISED by its own all-solid value, so "1:1" means
# equal importance rather than whichever number happens to be bigger -- shear is
# far more compliant than tension here and would otherwise swamp it.
LOAD_CASES = (
    ("tension", (0.0, TRACTION), 0.5),
    ("shear",   (TRACTION, 0.0), 0.5),
)

# ── Optimisation ───────────────────────────────────────────────────────────
FILTER_RADIUS = 2.0 * LX / NX
COMPLIANCE_ALLOW = 20.0
RHO_INIT = 0.5
MAX_ITER = 150

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "outputs", "thermal_isolation_2d_periodic")
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

# ── Periodicity in x ───────────────────────────────────────────────────────
# The patches sit in the middle of their edges, well clear of x = 0 and x = LX,
# so no Dirichlet dof lands on a periodic pair -- feax rejects that case (a
# half-pinned equivalence class is ambiguous), and it is worth keeping in mind
# if PATCH_FRAC is ever raised towards 1.
def _x_pairings(components):
    return [PeriodicPairing(
        location_master=lambda p: jnp.isclose(p[0], 0.0, atol=TOL),
        location_slave=lambda p: jnp.isclose(p[0], LX, atol=TOL),
        mapping=lambda p: p + jnp.array([LX, 0.0]),
        vec=c) for c in components]


P_th = flat.pbc.prolongation_matrix(_x_pairings([0]), mesh, vec=1)
P_me = flat.pbc.prolongation_matrix(_x_pairings([0, 1]), mesh, vec=2)

print(f"2-D thermal isolation, PERIODIC in x   {NX}x{NY} QUAD4  "
      f"{n_nodes:,} nodes, {mesh.cells.shape[0]:,} cells")
print(f"  patches: central {PATCH_FRAC*100:.0f}% of the bottom (T={T_HOT}, "
      f"clamped) and top (T={T_COLD}, loaded) edges")
print(f"  load cases: " + ", ".join(f"{n} {v} w={w:g}" for n, v, w in LOAD_CASES))
print(f"  dof reduction: thermal {P_th.shape[0]}->{P_th.shape[1]}, "
      f"mech {P_me.shape[0]}->{P_me.shape[1]}")

# ── Thermal ────────────────────────────────────────────────────────────────
thermal = tx.diffusion_problem(mesh, dim=2, ele_type="QUAD4")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(bottom_patch, "all", T_HOT),
    fe.DirichletBCSpec(top_patch, "all", T_COLD),
]).create_bc(thermal)

ts_th = fe.TracedStructure.from_problem(thermal)
direct = fe.DirectSolverOptions()
solve_T = fe.create_solver(
    thermal, bc_th, solver_options=direct, adjoint_solver_options=direct,
    linear=True, P=P_th, traced_structure=ts_th,
    traced_params=fe.TracedParams(volume_vars=(jnp.full((n_nodes,), K_SOLID),)))
guess_T = fe.zero_like_initial_guess(thermal, bc_th)
conduction_energy = tx.create_conduction_energy_fn(thermal)

# ── Mechanical, one problem per load case ──────────────────────────────────
# Same mesh, same material, same BC -- only the surface map differs. feax caches
# the FE tables per (mesh, vec, ele_type), so the second problem is cheap.
mech, bc_me, solvers, compliances, ts_me = [], None, [], [], None
for name, tvec, _w in LOAD_CASES:
    p = tx.elasticity_problem(mesh, dim=2, nu=NU, ele_type="QUAD4",
                              location_fns=(top_patch,), tractions=(tvec,))
    if bc_me is None:
        bc_me = fe.DirichletBCConfig([
            fe.DirichletBCSpec(bottom_patch, "all", 0.0)]).create_bc(p)
        ts_me = fe.TracedStructure.from_problem(p)
    mech.append(p)
    solvers.append(fe.create_solver(
        p, bc_me, solver_options=direct, adjoint_solver_options=direct,
        linear=True, P=P_me, traced_structure=ts_me,
        traced_params=fe.TracedParams(
            volume_vars=(jnp.full((n_nodes,), E_SOLID),))))
    compliances.append(gene.create_compliance_fn(p))
guess_u = fe.zero_like_initial_guess(mech[0], bc_me)

filter_fn = gene.create_helmholtz_filter(mesh, radius=FILTER_RADIUS)


# ── Responses ──────────────────────────────────────────────────────────────
def heat_leak(rho):
    tp = fe.TracedParams(volume_vars=(
        tx.simp_conductivity(rho, K_SOLID, K_MIN, PENAL_K),))
    return conduction_energy(solve_T(tp, guess_T, traced_structure=ts_th), tp)


def compliance_of(rho, i):
    tp = fe.TracedParams(volume_vars=(
        tx.simp_modulus(rho, E_SOLID, E_MIN, PENAL_E),))
    return compliances[i](solvers[i](tp, guess_u, traced_structure=ts_me))


# ── Reference: the all-solid design ────────────────────────────────────────
solid = jnp.ones((n_nodes,))
heat_solid = float(heat_leak(solid))
comp_solid = [float(compliance_of(solid, i)) for i in range(len(LOAD_CASES))]
weights = onp.array([w for _n, _v, w in LOAD_CASES])
print(f"  solid reference: heat={heat_solid:.4e}  "
      + "  ".join(f"C_{n}={c:.4e}" for (n, _v, _w), c
                  in zip(LOAD_CASES, comp_solid)))
print(f"  compliance limit: weighted normalised sum <= {COMPLIANCE_ALLOW:g}")
print("-" * 76)


def combined_compliance(rho):
    """Weighted sum of per-case compliances, each normalised by its own solid
    value. Still scales as ``s^-3`` under ``rho -> s rho``, so the
    heat x compliance invariant is unchanged by the normalisation."""
    terms = [w * compliance_of(rho, i) / cs
             for i, ((_n, _v, w), cs) in enumerate(zip(LOAD_CASES, comp_solid))]
    return sum(terms)


@jax.jit
def evaluate(x):
    rho = filter_fn(x)
    h, gh = jax.value_and_grad(heat_leak)(rho)
    c, gc = jax.value_and_grad(combined_compliance)(rho)
    per_case = jnp.stack([compliance_of(rho, i) / comp_solid[i]
                          for i in range(len(LOAD_CASES))])
    _, vjp = jax.vjp(filter_fn, x)
    return h, vjp(gh)[0], c, vjp(gc)[0], per_case


# ── MMA ────────────────────────────────────────────────────────────────────
_cache: dict = {}


def _cached(xx):
    key = xx.tobytes()
    if key not in _cache:
        _cache.clear()
        _cache[key] = [onp.asarray(a) for a in evaluate(jnp.asarray(xx))]
    return _cache[key]


history = {"iteration": [], "heat": [], "compliance": [], "volume": [],
           "product": [], "per_case": []}
it = [0]


def objective(xx, grad):
    h, gh, c, _, pc = _cached(xx)
    if grad.size > 0:
        grad[:] = gh / heat_solid
    it[0] += 1
    vol = float(onp.mean(filter_fn(jnp.asarray(xx))))
    h_rel, c_rel = float(h) / heat_solid, float(c)
    history["iteration"].append(it[0])
    history["heat"].append(h_rel)
    history["compliance"].append(c_rel)
    history["volume"].append(vol)
    history["product"].append(h_rel * c_rel)
    history["per_case"].append([float(v) for v in pc])
    if it[0] % 5 == 0 or it[0] == 1:
        cases = "  ".join(f"{n}={float(v):6.2f}"
                          for (n, _v, _w), v in zip(LOAD_CASES, pc))
        print(f"  it {it[0]:4d}  heat={h_rel:.4e}  C={c_rel:6.2f} "
              f"(limit {COMPLIANCE_ALLOW:g})  [{cases}]  vol={vol:.3f}  "
              f"heatxC={h_rel * c_rel:.3f}")
    return h_rel


def compliance_constraint(xx, grad):
    _, _, c, gc, _ = _cached(xx)
    if grad.size > 0:
        grad[:] = gc / COMPLIANCE_ALLOW
    return float(c) / COMPLIANCE_ALLOW - 1.0


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
h_fin, _, c_fin, _, pc_fin = _cached(x_opt)
prod = (float(h_fin) / heat_solid) * float(c_fin)

print("-" * 76)
print(f"heat leak   : {history['heat'][0]:.4e} -> {float(h_fin)/heat_solid:.4e} "
      f"x solid")
print(f"compliance  : {float(c_fin):.3f} (limit {COMPLIANCE_ALLOW:g}, "
      f"{'ACTIVE' if float(c_fin) > 0.98 * COMPLIANCE_ALLOW else 'slack'})   "
      + "  ".join(f"{n}={float(v):.2f}" for (n, _v, _w), v
                  in zip(LOAD_CASES, pc_fin)))
print(f"volume      : {rho_final.mean():.3f}   solid fraction "
      f"(rho>0.5): {(rho_final > 0.5).mean()*100:.1f}%")
print(f"heat x C    : {history['product'][0]:.3f} -> {prod:.3f}   "
      f"({1.0/max(prod, 1e-30):.2f}x better than uniform scaling; the "
      f"tension-only non-periodic baseline reached 0.596)")
print(f"evaluations : {it[0]}   wall time {dt:.1f}s")

# ── Outputs ────────────────────────────────────────────────────────────────
onp.save(os.path.join(OUTDIR, "density.npy"), rho_final.astype(onp.float32))
with open(os.path.join(OUTDIR, "history.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["iteration", "heat_over_solid", "compliance_weighted", "volume",
                "heat_times_compliance"]
               + [f"C_{n}_over_solid" for n, _v, _w in LOAD_CASES])
    for i in range(len(history["iteration"])):
        w.writerow([history["iteration"][i], history["heat"][i],
                    history["compliance"][i], history["volume"][i],
                    history["product"][i]] + history["per_case"][i])

pts = onp.asarray(mesh.points)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # Draw the cell twice side by side so the periodic seam at x = 0 / x = LX is
    # visible as a joint rather than as an edge.
    for shift in (0.0, LX):
        tpc = axes[0].tripcolor(pts[:, 0] + shift, pts[:, 1], rho_final,
                                cmap="gray_r", vmin=0.0, vmax=1.0,
                                shading="gouraud")
    axes[0].axvline(LX, color="tab:red", lw=0.8, ls="--")
    axes[0].set_title("density (cell drawn twice; dashed = periodic seam)")
    axes[0].set_aspect("equal")
    fig.colorbar(tpc, ax=axes[0])

    axes[1].semilogy(history["iteration"], history["heat"], "r-",
                     label="heat / solid (objective)")
    axes[1].semilogy(history["iteration"], history["product"], "k-",
                     label="heat x C (shape only)")
    axes[1].axhline(1.0, color="0.6", ls=":", label="uniform scaling")
    axes[1].axhline(0.596, color="tab:blue", ls="-.",
                    label="tension-only baseline")
    axes[1].set_xlabel("evaluation")
    axes[1].set_ylabel("relative to solid block")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("objective")

    per = onp.array(history["per_case"])
    for j, (n, _v, _w) in enumerate(LOAD_CASES):
        axes[2].plot(history["iteration"], per[:, j], label=f"C_{n} / solid")
    axes[2].plot(history["iteration"], history["compliance"], "k-",
                 label="weighted sum")
    axes[2].axhline(COMPLIANCE_ALLOW, color="r", ls="--", label="limit")
    axes[2].set_xlabel("evaluation")
    axes[2].set_ylabel("compliance / solid")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("constraint")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "density.png"), dpi=150)
    plt.close()
except ImportError:
    print("(no matplotlib: skipped density.png)")

try:
    rho_j = jnp.asarray(rho_final)
    tp_th = fe.TracedParams(volume_vars=(
        tx.simp_conductivity(rho_j, K_SOLID, K_MIN, PENAL_K),))
    tp_me = fe.TracedParams(volume_vars=(
        tx.simp_modulus(rho_j, E_SOLID, E_MIN, PENAL_E),))
    T = onp.asarray(solve_T(tp_th, guess_T, traced_structure=ts_th).field(0))
    infos = [("density", rho_final), ("temperature", T.reshape(-1))]
    for i, (name, _v, _w) in enumerate(LOAD_CASES):
        u = onp.asarray(solvers[i](tp_me, guess_u,
                                   traced_structure=ts_me).field(0))
        infos.append((f"u_{name}", onp.column_stack([u, onp.zeros(len(u))])))
    save_sol(mesh, os.path.join(OUTDIR, "final.vtu"), point_infos=infos)
except Exception as e:
    print(f"(no vtu: {e})")

print(f"wrote density.npy / history.csv / density.png / final.vtu to "
      f"{os.path.relpath(OUTDIR, os.getcwd())}/")
