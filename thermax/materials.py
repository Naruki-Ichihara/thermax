"""Density -> conductivity interpolation.

The one piece all three prototype examples shared verbatim, and the one place a
thermal problem differs from a load-driven one in a way that bites.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["simp_conductivity", "simp_modulus", "ramp_conductivity",
           "ramp_modulus"]


def simp_conductivity(rho, k_solid: float, k_min: float, penal: float = 3.0):
    """SIMP interpolation ``k = k_min + rho**penal * (k_solid - k_min)``.

    Parameters
    ----------
    rho
        Density field, any shape, expected in ``[0, 1]``.
    k_solid, k_min
        Conductivity of the solid and of the void.
    penal
        SIMP exponent.

    Notes
    -----
    ``k_min`` is not a numerical fudge here the way ``E_min = 1e-9`` is in
    compliance minimisation. Whether it may be that small depends on where the
    heat comes from:

    * heat enters through the BOUNDARY (a heat sink, a chip footprint) -- void
      carries no source, so ``k_min`` only has to keep the operator non-singular
      and ``1e-9`` is fine;
    * heat is generated in the VOLUME (the volume-to-point / dendritic problem)
      -- void generates heat too, and with nowhere to conduct it the temperature
      there diverges. Use ``k_min = 1e-3 * k_solid``, as the dolfin-adjoint
      Poisson-topology demo does, for the same reason.

    Getting this wrong does not raise; it produces a finite but meaningless
    objective that grows as the void gets colder-resistant.
    """
    return k_min + rho**penal * (k_solid - k_min)


def simp_modulus(rho, e_solid: float, e_min: float = 1e-6, penal: float = 3.0):
    """SIMP interpolation ``E = e_min + rho**penal * (e_solid - e_min)``.

    Identical algebra to :func:`simp_conductivity`; separate because the two
    have different constraints on the floor and, in a coupled problem, may want
    different exponents.

    ``e_min`` is pure regularisation -- it exists to keep ``K`` non-singular
    where the design has vanished -- so it can be as small as the linear solver
    tolerates, ``1e-6`` to ``1e-9``.

    Coupled-problem note
    --------------------
    When a thermal objective is traded against a mechanical constraint, whether
    ``penal`` here MATCHES the thermal one decides whether the problem has any
    design freedom at all. Along a load path of varying section ``A(l)``,

        thermal resistance = integral dl/(kA),   axial compliance = integral dl/(EA)

    are the same integral, so with ``k`` and ``E`` interpolated by the same power
    of ``rho`` the stiffness-to-conductance ratio is ``e_solid/k_solid`` for ANY
    shape: thinning, tapering or meandering costs stiffness and saves heat in
    exactly equal proportion, and the optimiser has nothing to exploit under a
    purely axial load. Different exponents break the tie (and so does any load
    with bending or shear in it, whose stiffness scales with ``A d^2`` rather
    than ``A``).
    """
    return e_min + rho**penal * (e_solid - e_min)


def ramp_conductivity(rho, k_solid: float, k_min: float, q: float = -0.5):
    """RAMP interpolation ``k = k_min + (1-k_min) * rho / (1 + q(1-rho))``.

    For a problem that MINIMISES conductance, this is the interpolation to use,
    and SIMP with ``p > 1`` is actively wrong.

    Which side penalises grey depends on the direction of the objective, and it
    flips. Compare, at equal volume, spreading material over area ``A`` at
    density ``rho`` against concentrating it as solid over area ``rho A``::

        grey        conductance  ~  f(rho) * A
        solid/void  conductance  ~  1 * rho A

    so grey is WORSE -- which is what a penalisation has to achieve -- exactly
    when ``f(rho) > rho``. Standard heat topology optimization maximises
    conductance, wants ``f(rho) < rho``, and gets it from SIMP with ``p > 1``.
    Here conductance is minimised, the requirement inverts to ``f(rho) > rho``,
    and SIMP with ``p = 3`` does the opposite of what is wanted: it makes
    intermediate density the cheapest insulator available and the design fills
    with grey.

    That grey is not purely an artefact -- ``p_k = 3`` is a (crude) statement
    that porous material insulates far better than its volume fraction, which is
    true, and is why insulation is foam. Using ``q < 0`` here is a deliberate
    decision to forbid that answer and ask for a discrete structure instead.
    Real open-cell foams sit near ``f(rho) = rho``, neither.

    Why RAMP rather than ``rho**p`` with ``p < 1``
    ----------------------------------------------
    Both give ``f(rho) > rho``. But ``d(rho**p)/drho = p rho**(p-1)`` DIVERGES as
    ``rho -> 0`` for ``p < 1``, and with a normalised Heaviside projection void
    is exactly 0, so the sensitivity blows up where the design is emptiest.
    RAMP's slope is bounded everywhere::

        R'(rho) = (1 + q) / (1 + q(1-rho))**2,     R'(0) = 1/(1+q)

    which is 2 at ``q = -0.5``. Admissible range is ``-1 < q < 0``; ``q = 0`` is
    linear (no penalisation either way) and ``q -> -1`` is the singular end.
    """
    if not -1.0 < q < 0.0:
        raise ValueError(
            f"ramp_conductivity: q must lie in (-1, 0) to penalise intermediate "
            f"density in a conductance-MINIMISING problem; got q={q}. "
            f"q = 0 is linear, q > 0 rewards grey."
        )
    return k_min + (1.0 - k_min) * rho / (1.0 + q * (1.0 - rho))


def ramp_modulus(rho, e_solid: float, e_min: float = 1e-6, q: float = 8.0):
    """RAMP interpolation ``E = e_min + (e_solid-e_min) * rho / (1 + q(1-rho))``.

    Same algebra as :func:`ramp_conductivity` with the SIGN OF q REVERSED,
    because stiffness is wanted and conductance is not. Grey is penalised when
    the interpolation sits on the unfavourable side of linear, and "unfavourable"
    flips with the direction of the objective::

        stiffness   (wanted)    penalise grey  <=>  f(rho) < rho  <=>  q > 0
        conductance (unwanted)  penalise grey  <=>  f(rho) > rho  <=>  q < 0

    ``q = 8`` is close to SIMP ``p = 3`` in strength (0.10 against 0.125 at
    ``rho = 0.5``) and rather stronger below that.

    Why not just keep SIMP here
    ---------------------------
    Because ``d(rho**3)/drho = 3 rho**2`` VANISHES at ``rho = 0``, and once the
    conductivity uses RAMP the two no longer vanish together. At an empty design
    the optimiser then sees

        adding material costs  dk/drho = (1-k_min)/(1+q_k)  = 2      (RAMP: finite)
        adding material buys   dE/drho = 3 rho**2           = 0      (SIMP: none)

    -- material costs heat and buys no stiffness, so the compliance constraint
    has nothing to pull back with and the void state is a one-way trap. Measured:
    with SIMP stiffness and RAMP conductivity and no Heaviside projection, the
    design collapsed to V = 0.0000 with C = 484,536 at evaluation ~126 and never
    recovered over the remaining 74. With both vanishing together (the original
    p_k = p_E = 3) the void state was merely neutral, which is why the trap only
    appeared once the conductivity was fixed.

    RAMP was introduced for exactly this defect of SIMP. Here ``dE/drho`` at
    ``rho = 0`` is ``(e_solid-e_min)/(1+q)``, about ``E1/9`` -- against a
    conductivity cost of 2, so returning material is overwhelmingly worthwhile
    and the trap is gone.
    """
    if q <= 0.0:
        raise ValueError(
            f"ramp_modulus: q must be > 0 to penalise intermediate density in a "
            f"stiffness-WANTED problem; got q={q}. q = 0 is linear, and q < 0 "
            f"(the conductivity convention) would reward grey here."
        )
    return e_min + (e_solid - e_min) * rho / (1.0 + q * (1.0 - rho))
