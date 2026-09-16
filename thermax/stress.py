"""p-norm aggregated von Mises stress, for stress-constrained topology optimization.

Three things have to be right for a stress constraint to work at all, and each
is a separate decision:

1. WHICH STRESS. The equilibrium uses the SIMP modulus ``rho^p E0``, but the
   thing that fails is the solid material inside the cell, not the homogenised
   cell. Using the macroscopic stress ``rho^p C0 : eps`` would be wrong AND
   useless: it is invariant under a uniform scaling of the design (the stress
   field is fixed by equilibrium with a fixed load), so it could never stop the
   structure from thinning away. The measure here is

       sigma = rho^q C0 : eps(u),    0 < q < p

   ``q > 0`` is what relaxes the SINGULARITY PHENOMENON: at ``q = 0`` a void
   cell still reports whatever stress its neighbours drag it through, so the
   constraint stays active in void and blocks the very material removal the
   optimiser needs, leaving the true optimum in a degenerate subspace no
   gradient method can enter. ``q < p`` keeps the relaxed feasible set
   connected. ``q = p - 1`` additionally has a physical reading -- the stress in
   the ligaments of a porous cell is roughly the macroscopic one divided by
   ``rho`` -- and is the default.

2. AGGREGATION. Stress is local: one constraint per quadrature point is
   hundreds of thousands of constraints, which MMA cannot take. The p-norm

       sigma_PN = ( sum_q w_q sigma_q^P / sum_q w_q )^(1/P)

   collapses them into one. It is volume-weighted so refining the mesh does not
   move it, and it always UNDERESTIMATES the true maximum -- by more the lower
   ``P`` is, and by less the more uniform the field. Raising ``P`` closes the
   gap and ruins the conditioning; ``P = 8`` is the usual compromise.

3. THE UNDERESTIMATE. Constraining ``sigma_PN`` alone therefore leaves the real
   peak stress above the limit. :func:`adaptive_scale` implements the standard
   correction: a scalar ``c`` tracking ``sigma_max / sigma_PN``, relaxed between
   iterations so it does not chatter, with ``c * sigma_PN`` constrained instead.
   Report both numbers -- if they stay far apart the aggregation is hiding a
   local hot spot.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["create_stress_tensor_fn", "create_vonmises_fn", "pnorm",
           "adaptive_scale"]


def _flat(sol):
    """Flat DOF vector of a feax ``Solution``, or a raw array passed through.

    ``getattr(sol, "flat", sol)`` is NOT a way to write this: a jax ``Array``
    *has* a ``flat`` attribute, and it raises ``NotImplementedError`` rather
    than being absent, so the fallback never fires. ``Solution`` keeps the
    vector in ``dofs``, which a raw array does not have.
    """
    return jnp.asarray(getattr(sol, "dofs", sol))


def _von_mises_2d(sigma):
    """Plane-stress von Mises of a 2x2 tensor."""
    s11, s22, s12 = sigma[0, 0], sigma[1, 1], sigma[0, 1]
    return jnp.sqrt(jnp.maximum(
        s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2, 1e-30))


def create_stress_tensor_fn(problem, stress_at_quad):
    """Quadrature-point stress TENSOR field of a solved displacement.

    Returns ``f(sol, traced_params) -> (sigma, weights)`` with ``sigma`` of
    shape ``(num_cells, num_quads, 2, 2)`` and ``weights`` the ``JxW`` factors.
    :func:`create_vonmises_fn` is this followed by the von Mises invariant; the
    tensor itself is what a VTU wants, because the components say which way a
    member is loaded and the invariant alone does not.

    ``stress_at_quad`` is ``(u_grad, *volume_vars_at_quad) -> (2, 2)``. Whether
    it returns the relaxed measure or the true stress is the caller's choice;
    for a diagnostic field, pass ``q = 0`` so it is the real material stress.
    """
    fe0 = problem.fes[0]
    cells = problem.cells_list[0]
    sg, jxw = fe0.shape_grads, fe0.JxW
    vec, num_cells, num_quads = fe0.vec, fe0.num_cells, fe0.num_quads

    def _interp(traced_params):
        out = []
        for var in traced_params.volume_vars:
            if var.ndim == 1 and var.shape[0] == num_cells:
                out.append(jnp.tile(var[:, None], (1, num_quads)))
            else:
                out.append(jnp.einsum("qn,cn->cq", fe0.shape_vals, var[cells]))
        return out

    def tensor(sol, traced_params):
        u = _flat(sol).reshape(-1, vec)
        cell_u = u[cells]
        vars_q = _interp(traced_params)

        def per_cell(cell_sol, cell_sg, *vars_c):
            u_grads = jnp.sum(cell_sol[None, :, :, None] * cell_sg[:, :, None, :],
                              axis=1)
            return jax.vmap(lambda ug, *vq: stress_at_quad(ug, *vq))(
                u_grads, *vars_c)

        return jax.vmap(per_cell)(cell_u, sg, *vars_q), jxw

    return tensor


def _von_mises_3d(sigma):
    """Full 3-D von Mises of a 3x3 tensor: ``sqrt(3/2 s:s)`` on the deviator."""
    dev = sigma - jnp.eye(3) * (jnp.trace(sigma) / 3.0)
    return jnp.sqrt(jnp.maximum(1.5 * jnp.sum(dev * dev), 1e-30))


def create_vonmises_fn(problem, stress_at_quad):
    """Quadrature-point von Mises field: the tensor field, then the invariant.

    Returns ``f(sol, traced_params) -> (sigma_vm, weights)``, both
    ``(num_cells, num_quads)``, ready for a volume-weighted aggregation.

    Plane stress or full 3-D is chosen from the shape the stress function
    actually returns, not from a flag -- the 2x2 plane-stress invariant and the
    3x3 deviatoric one are different expressions, and the 2x2 form is NOT the
    3x3 form with a row deleted.
    """
    tensor = create_stress_tensor_fn(problem, stress_at_quad)

    def vonmises(sol, traced_params):
        sig, w = tensor(sol, traced_params)
        vm = _von_mises_2d if sig.shape[-1] == 2 else _von_mises_3d
        return jax.vmap(jax.vmap(vm))(sig), w

    return vonmises


def pnorm(sigma, weights, P: float = 8.0):
    """Volume-weighted p-norm of a stress field, and its true maximum.

    Returns ``(sigma_PN, sigma_max)``. The max is for reporting only -- it is
    not differentiable in a useful way, which is the entire reason the p-norm
    exists.

    The field is normalised by its own maximum before being raised to ``P`` and
    put back afterwards. Without that, ``sigma^8`` on a field of order 1e4
    overflows float64 long before the sum is taken.
    """
    smax = jnp.max(sigma)
    scaled = sigma / jnp.maximum(smax, 1e-30)
    s = jnp.sum(weights * scaled**P) / jnp.sum(weights)
    return smax * s ** (1.0 / P), smax


def adaptive_scale(c_prev: float, sigma_max: float, sigma_pn: float,
                   alpha: float = 0.5) -> float:
    """One update of the p-norm correction factor.

    ``c`` chases ``sigma_max / sigma_PN`` under relaxation ``alpha``; the
    constraint imposed is ``c * sigma_PN <= limit``, so that what is really
    bounded approaches the true peak. ``alpha = 1`` takes the ratio outright and
    tends to oscillate, ``alpha`` near 0 barely corrects at all.

    This is a per-iteration bookkeeping step OUTSIDE the differentiated code:
    ``c`` is held constant within an evaluation, so no gradient flows through
    it.
    """
    return alpha * float(sigma_max / max(sigma_pn, 1e-30)) + (1.0 - alpha) * c_prev
