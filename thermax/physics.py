"""feax ``Problem`` builders for the two fields a thermal-isolation problem needs.

Both take their material property as a traced volume variable (``k`` and ``E``),
NOT a density -- the SIMP map lives in :mod:`thermax.materials` and is applied by
the caller. That split is what lets one problem instance serve an optimisation
loop with any interpolation, and it keeps the constitutive code free of design
variables.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

import feax as fe

__all__ = [
    "diffusion_problem",
    "elasticity_problem",
    "anisotropic_diffusion_problem",
    "anisotropic_elasticity_problem",
]


def diffusion_problem(mesh, dim: int, ele_type: str = "QUAD4") -> fe.Problem:
    """Steady scalar diffusion, ``div(k grad T) = 0``, as a ``vec=1`` problem.

    Defined through ``get_energy_density`` rather than ``get_tensor_map``:

        psi(grad T, k) = 1/2 k |grad T|^2

    feax differentiates that to get Fourier's law ``q = k grad T`` for the
    residual, and the same density is what :func:`feax.create_energy_fn` (and
    hence :func:`thermax.objectives.create_conduction_energy_fn`) integrates for
    the objective. Writing the flux directly would give the residual but leave
    ``create_energy_fn`` with nothing to integrate.

    ``k`` arrives as the single traced volume variable, nodal or per-cell.
    """

    class _Diffusion(fe.Problem):
        def get_energy_density(self):
            def psi(grad_T, k):
                return 0.5 * k * jnp.sum(grad_T**2)
            return psi

    return _Diffusion(mesh, vec=1, dim=dim, ele_type=ele_type)


def elasticity_problem(
    mesh,
    dim: int,
    nu: float = 0.3,
    ele_type: str = "QUAD4",
    location_fns: Sequence = (),
    tractions: Sequence = (),
) -> fe.Problem:
    """Linear isotropic elasticity with a design-dependent Young's modulus.

    ``E`` arrives as the single traced volume variable; ``nu`` is fixed, which is
    the usual SIMP convention (interpolating ``nu`` buys nothing and costs
    well-posedness at the void end).

    In 2-D this is PLANE STRAIN -- ``lambda = E nu / ((1+nu)(1-2nu))``, the same
    expression as 3-D. For a thin specimen you want plane stress instead;
    substitute ``lambda = E nu / (1 - nu^2)``.

    ``location_fns`` and ``tractions`` are parallel sequences: each location
    predicate gets the constant traction vector at the same index.
    """
    if len(location_fns) != len(tractions):
        raise ValueError(
            f"location_fns ({len(location_fns)}) and tractions "
            f"({len(tractions)}) must have the same length"
        )
    traction_vectors = [jnp.asarray(t, dtype=float) for t in tractions]

    class _Elasticity(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, E):
                mu = E / (2.0 * (1.0 + nu))
                lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                eps = 0.5 * (u_grad + u_grad.T)
                return lam * jnp.trace(eps) * jnp.eye(dim) + 2.0 * mu * eps
            return stress

        def get_surface_maps(self):
            return [
                (lambda u, x, *args, _t=t: _t) for t in traction_vectors
            ]

    return _Elasticity(
        mesh, vec=dim, dim=dim, ele_type=ele_type,
        location_fns=tuple(location_fns),
    )


def anisotropic_diffusion_problem(mesh, k_par: float, k_perp: float,
                                  k_min: float = 1e-4, penal: float = 3.0,
                                  ramp_q=None, sgn_beta: float = 10.0,
                                  ele_type: str = "QUAD4") -> fe.Problem:
    """Scalar diffusion through a transversely isotropic material, ``vec=1``.

    Traced volume variables, in order: ``rho, t1, t2``.

        a2 = unit_trace_a2(t1, t2)                 (trace 1, admissible)
        K  = k_perp I + (k_par - k_perp) a2        (exact 2nd-order average)
        psi(grad T) = 1/2 w(rho) grad T . K . grad T,
        w(rho) = k_min + rho**penal (1 - k_min)

    Density carries the AMOUNT of material and the orientation carries only the
    direction; the two are deliberately separated so that ``K`` never needs a
    sub-unit trace (see :func:`thermax.orientation.unit_trace_a2`).
    """
    from .materials import ramp_conductivity
    from .orientation import orientation_conductivity, unit_trace_a2

    # ``ramp_q`` selects the interpolation. For a conductance-MINIMISING
    # objective RAMP with q < 0 is the correct one and SIMP p > 1 rewards grey;
    # see :func:`thermax.materials.ramp_conductivity`.
    if ramp_q is not None:
        def weight(rho):
            return ramp_conductivity(rho, 1.0, k_min, ramp_q)
    else:
        def weight(rho):
            return k_min + rho**penal * (1.0 - k_min)

    class _AnisoDiffusion(fe.Problem):
        def get_energy_density(self):
            def psi(grad_T, rho, t1, t2):
                K = orientation_conductivity(
                    unit_trace_a2(t1, t2, sgn_beta), k_par, k_perp)
                w = weight(rho)
                return 0.5 * w * jnp.dot(grad_T.ravel(), K @ grad_T.ravel())
            return psi

    return _AnisoDiffusion(mesh, vec=1, dim=2, ele_type=ele_type)


def anisotropic_elasticity_problem(mesh, E1: float, E2: float, G12: float,
                                   nu12: float, e_min: float = 1e-6,
                                   penal: float = 3.0, ramp_q=None,
                                   alpha_par: float = 0.0,
                                   alpha_perp: float = 0.0, t_ref: float = 0.0,
                                   sgn_beta: float = 10.0,
                                   ele_type: str = "QUAD4",
                                   location_fns=(), tractions=()) -> fe.Problem:
    """Plane-stress elasticity of an orientation-averaged composite, ``vec=2``.

    Traced volume variables, in order: ``rho, t1, t2, T``.

        a2     = unit_trace_a2(t1, t2)
        a4     = hybrid_closure(a2)
        C      = orientation_averaged_stiffness(a2, a4, E1, E2, G12, nu12)
        alpha  = orientation_cte(a2, alpha_par, alpha_perp)
        eps_el = sym(grad u) - alpha (T - t_ref)
        psi    = 1/2 w(rho) eps_el : C : eps_el

    Defined through ``get_energy_density``, not ``get_tensor_map``. feax
    differentiates psi to get the residual stress ``C : eps_el``, and the SAME
    density is what :func:`feax.create_energy_fn` integrates to give the stored
    elastic energy U -- which is the structural measure this problem needs.
    Compliance is not: ``int t.u`` stops equalling twice the strain energy the
    moment a thermal strain is present, and it cannot see thermal self-stress at
    all. Scale the traction by ``lambda`` and

        int t.u = lambda^2 C_mech + lambda (int t.u_therm)  ->  0,

    while the thermal stress is independent of ``lambda`` and stays put. A
    structure that is stiff enough to crack on cooldown scores perfectly on
    compliance. U reduces to ``1/2 int t.u`` for pure mechanical loading, so
    nothing is lost, and to the thermal self-energy when the traction is
    removed. Lower is better in both directions: stiffer against the load,
    and more accommodating of the contraction.

    ``T`` is not a design variable -- it is the solved temperature, handed over
    with ``TracedParams.node_var_from_solution``, and the chain rho -> T -> u
    stays differentiable.

    ``a_4`` comes from a CLOSURE, not from design variables of its own. The
    closure-free :func:`feax.mechanics.orientation.build_a4_2d` would give it two
    extra degrees of freedom per node that change stiffness at fixed ``a_2`` --
    hence at fixed conductivity, since conductivity is a second-order average
    and depends on ``a_2`` alone. Those degrees of freedom were measured to be
    worth very little here: they go inert wherever ``a_2`` is rank-1, and the
    converged designs sat at rank-1 through the struts and legs (0.92-0.99 by
    ``|a12|/sqrt(a11 a22)``), leaving them live only in the bands. They also
    have no 3-D counterpart in feax, so dropping them keeps the formulation
    portable.

    ``hybrid_closure`` blends the linear and quadratic closures by
    ``f = 1 - 4 det(a_2)``, which is 0 at the isotropic state -- where the
    linear closure is exact -- and 1 at full alignment, where the quadratic one
    is. Plain ``quadratic_closure`` is the wrong default: at ``a_2 = I/2`` it
    pins ``a_4`` to the +-45 cross-ply, which was measured 2.36x softer in the
    load direction than the 0/90 state the same ``a_2`` also admits.
    """
    from feax.mechanics.orientation import hybrid_closure
    from feax.mechanics.orientation import orientation_averaged_stiffness

    from .materials import ramp_modulus
    from .orientation import orientation_cte, unit_trace_a2

    # ``ramp_q`` (> 0) selects RAMP over SIMP. Pair it with a RAMP conductivity:
    # mixing SIMP stiffness with RAMP conductivity makes the void state a
    # one-way trap, because only one of the two gradients vanishes at rho = 0.
    # See :func:`thermax.materials.ramp_modulus`.
    if ramp_q is not None:
        def weight(rho):
            return ramp_modulus(rho, 1.0, e_min, ramp_q)
    else:
        def weight(rho):
            return e_min + rho**penal * (1.0 - e_min)

    if len(location_fns) != len(tractions):
        raise ValueError("location_fns and tractions must have the same length")
    traction_vectors = [jnp.asarray(t, dtype=float) for t in tractions]

    class _AnisoElasticity(fe.Problem):
        def get_energy_density(self):
            def psi(u_grad, rho, t1, t2, T):
                a2 = unit_trace_a2(t1, t2, sgn_beta)
                a4 = hybrid_closure(a2)
                C = orientation_averaged_stiffness(
                    a2, a4, E1=E1, E2=E2, G12=G12, nu12=nu12)
                alpha = orientation_cte(a2, alpha_par, alpha_perp)
                eps = (0.5 * (u_grad + u_grad.T) - alpha * (T - t_ref))
                return 0.5 * weight(rho) * jnp.einsum("ij,ijkl,kl->", eps, C, eps)
            return psi

        def get_surface_maps(self):
            return [(lambda u, x, *a, _t=t: _t) for t in traction_vectors]

    return _AnisoElasticity(mesh, vec=2, dim=2, ele_type=ele_type,
                            location_fns=tuple(location_fns))


def anisotropic_stress_measure(E1: float, E2: float, G12: float, nu12: float,
                               q: float = 2.0, alpha_par: float = 0.0,
                               alpha_perp: float = 0.0, t_ref: float = 0.0,
                               sgn_beta: float = 10.0):
    """``stress_at_quad`` for :func:`thermax.stress.create_vonmises_fn`.

    Companion to :func:`anisotropic_elasticity_problem`: same orientation
    algebra, same traced variables ``(rho, t1, t2, t4, t5)``, but weighted by
    ``rho**q`` instead of the SIMP ``rho**penal``.

    That difference is the whole point. The residual needs the homogenised
    modulus ``rho**penal E0``; the failure criterion needs the stress in the
    material itself. With ``q < penal`` the void goes stress-free faster than it
    goes soft, which is what makes the relaxed feasible set connected -- see the
    module docstring of :mod:`thermax.stress`.
    """
    from feax.mechanics.orientation import hybrid_closure
    from feax.mechanics.orientation import orientation_averaged_stiffness

    from .orientation import orientation_cte, unit_trace_a2

    def stress_at_quad(u_grad, rho, t1, t2, T):
        a2 = unit_trace_a2(t1, t2, sgn_beta)
        a4 = hybrid_closure(a2)
        C = orientation_averaged_stiffness(a2, a4, E1=E1, E2=E2, G12=G12,
                                           nu12=nu12)
        alpha = orientation_cte(a2, alpha_par, alpha_perp)
        eps = 0.5 * (u_grad + u_grad.T) - alpha * (T - t_ref)
        return rho**q * jnp.einsum("ijkl,kl->ij", C, eps)

    return stress_at_quad


# ---------------------------------------------------------------------------
# 3-D counterparts. Same orientation algebra, one dimension up, and NO thermal
# strain: these are for the uncoupled formulation, where alpha = 0 makes the
# temperature absent from the mechanical energy density entirely. Adding it back
# would mean carrying the solved T as a sixth volume variable, exactly as the
# 2-D builder does.
# ---------------------------------------------------------------------------

def anisotropic_diffusion_problem_3d(mesh, k_par: float, k_perp: float,
                                     k_min: float = 1e-4, penal: float = 3.0,
                                     ramp_q=None, sgn_beta: float = 10.0,
                                     ele_type: str = "HEX8") -> fe.Problem:
    """Scalar diffusion through a transversely isotropic material in 3-D, ``vec=1``.

    Traced volume variables, in order: ``rho, t1, t2, t12, t13``.

        a2 = unit_trace_a2_3d(t1, t2, t12, t13)    (trace 1, PSD, rank-1 at the
                                                    sharp-sign limit)
        K  = k_perp I + (k_par - k_perp) a2        (exact 2nd-order average)
        psi(grad T) = 1/2 w(rho) grad T . K . grad T

    Four orientation variables rather than the six of Nomura's 3-D construction;
    see :func:`thermax.orientation.unit_trace_a2_3d` for why the sixth is not
    merely redundant but actively inadmissible.
    """
    from .materials import ramp_conductivity
    from .orientation import orientation_conductivity, unit_trace_a2_3d

    if ramp_q is not None:
        def weight(rho):
            return ramp_conductivity(rho, 1.0, k_min, ramp_q)
    else:
        def weight(rho):
            return k_min + rho**penal * (1.0 - k_min)

    class _AnisoDiffusion3D(fe.Problem):
        def get_energy_density(self):
            def psi(grad_T, rho, t1, t2, t12, t13):
                K = orientation_conductivity(
                    unit_trace_a2_3d(t1, t2, t12, t13, sgn_beta), k_par, k_perp)
                return 0.5 * weight(rho) * jnp.dot(grad_T.ravel(),
                                                   K @ grad_T.ravel())
            return psi

    return _AnisoDiffusion3D(mesh, vec=1, dim=3, ele_type=ele_type)


def anisotropic_elasticity_problem_3d(mesh, E1: float, E2: float, G12: float,
                                      nu12: float, nu23: float,
                                      e_min: float = 1e-6, penal: float = 3.0,
                                      ramp_q=None, sgn_beta: float = 10.0,
                                      ele_type: str = "HEX8",
                                      location_fns=(), tractions=()) -> fe.Problem:
    """3-D elasticity of an orientation-averaged composite, ``vec=3``.

    Traced volume variables, in order: ``rho, t1, t2, t12, t13``.

        a2  = unit_trace_a2_3d(t1, t2, t12, t13)
        a4  = hybrid_closure(a2)                   (the 27 det scale, in 3-D)
        C   = orientation_averaged_stiffness_3d(a2, a4, E1, E2, G12, nu12, nu23)
        psi = 1/2 w(rho) eps : C : eps

    ``nu23`` is new and has no 2-D counterpart: the plane-stress builder never
    needs the transverse-plane Poisson ratio, the 3-D stiffness invariants do.
    """
    from feax.mechanics.orientation import hybrid_closure
    from feax.mechanics.orientation import orientation_averaged_stiffness_3d

    from .materials import ramp_modulus
    from .orientation import unit_trace_a2_3d

    if ramp_q is not None:
        def weight(rho):
            return ramp_modulus(rho, 1.0, e_min, ramp_q)
    else:
        def weight(rho):
            return e_min + rho**penal * (1.0 - e_min)

    if len(location_fns) != len(tractions):
        raise ValueError("location_fns and tractions must have the same length")
    traction_vectors = [jnp.asarray(t, dtype=float) for t in tractions]

    class _AnisoElasticity3D(fe.Problem):
        def get_energy_density(self):
            def psi(u_grad, rho, t1, t2, t12, t13):
                a2 = unit_trace_a2_3d(t1, t2, t12, t13, sgn_beta)
                C = orientation_averaged_stiffness_3d(
                    a2, hybrid_closure(a2), E1=E1, E2=E2, G12=G12,
                    nu12=nu12, nu23=nu23)
                eps = 0.5 * (u_grad + u_grad.T)
                return 0.5 * weight(rho) * jnp.einsum("ij,ijkl,kl->", eps, C, eps)
            return psi

        def get_surface_maps(self):
            return [(lambda u, x, *a, _t=t: _t) for t in traction_vectors]

    return _AnisoElasticity3D(mesh, vec=3, dim=3, ele_type=ele_type,
                              location_fns=tuple(location_fns))


def anisotropic_stress_measure_3d(E1: float, E2: float, G12: float,
                                  nu12: float, nu23: float, q: float = 2.0,
                                  sgn_beta: float = 10.0):
    """``stress_at_quad`` for the 3-D problem; ``rho**q`` relaxation as in 2-D."""
    from feax.mechanics.orientation import hybrid_closure
    from feax.mechanics.orientation import orientation_averaged_stiffness_3d

    from .orientation import unit_trace_a2_3d

    def stress_at_quad(u_grad, rho, t1, t2, t12, t13):
        a2 = unit_trace_a2_3d(t1, t2, t12, t13, sgn_beta)
        C = orientation_averaged_stiffness_3d(
            a2, hybrid_closure(a2), E1=E1, E2=E2, G12=G12, nu12=nu12, nu23=nu23)
        eps = 0.5 * (u_grad + u_grad.T)
        return rho**q * jnp.einsum("ijkl,kl->ij", C, eps)

    return stress_at_quad
