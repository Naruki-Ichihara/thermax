"""Objectives, and what each one is actually proportional to."""

from __future__ import annotations

import jax.numpy as jnp

import feax as fe

__all__ = ["create_conduction_energy_fn"]


def create_conduction_energy_fn(problem):
    """Conduction energy ``E = 1/2 integral k |grad T|^2 dOmega`` of a solved field.

    Returns ``f(sol, traced_params) -> scalar``, where ``sol`` is whatever
    ``feax.create_solver`` returned and ``traced_params`` is the same object the
    solve was given (the energy density needs ``k`` again).

    Why this is the heat leak
    -------------------------
    For a TWO-TERMINAL problem -- temperature prescribed on exactly two patches,
    no volumetric source, everything else insulated -- the solved field satisfies

        E = 1/2 G (T_hot - T_cold)^2,     Q = G (T_hot - T_cold),

    with ``G`` the conductance between the patches. ``E`` and the transmitted
    heat ``Q`` therefore differ by the fixed factor ``(T_hot - T_cold)/2``, so
    minimising ``E`` minimises the leak exactly, with no separate flux
    post-processing and no adjoint of a surface integral.

    The proportionality is not generic. Add a volumetric source, a third
    terminal, or a convective (Robin) boundary and ``E`` stops tracking ``Q``;
    integrate the flux through the patch instead.

    Sign note: ``dE/drho > 0`` -- more material always conducts more. That rules
    out Optimality Criteria, whose multiplicative update only fires for the
    other gradient sign. Use MMA.
    """
    energy_fn = fe.create_energy_fn(problem)

    def conduction_energy(sol, traced_params):
        # A feax Solution keeps the flat vector in ``dofs``; a raw array is
        # already flat. Not ``getattr(sol, "flat", sol)`` -- a jax Array HAS a
        # ``flat`` attribute and raises from it, so that fallback never fires.
        return energy_fn(jnp.asarray(getattr(sol, "dofs", sol)), traced_params)

    return conduction_energy
