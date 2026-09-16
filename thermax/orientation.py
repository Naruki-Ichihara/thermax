"""Box-bounded parameterisation of the closure-free ``a_4`` freedom.

:func:`feax.mechanics.orientation.build_a4_2d` takes ``a_2`` plus a Schur
reflection coefficient ``rho_2 = y1 + i y2`` and sweeps the ENTIRE admissible
set of fourth-order orientation tensors for that ``a_2``. Admissibility needs
``|rho_2| <= 1`` -- the closed unit disk -- and ``build_a4_2d`` does not enforce
it.

That matters here because ``a_4`` is the one design freedom a thermal-isolation
problem gets for free: the orientation-averaged CONDUCTIVITY

    k_ij = k_perp delta_ij + (k_par - k_perp) a2_ij

is a second-order average and depends on ``a_2`` ALONE, while the stiffness is a
fourth-order average and depends on ``a_2`` and ``a_4``. Moving ``rho_2`` at
fixed ``a_2`` therefore changes stiffness at exactly zero thermal cost. Losing
that freedom to an inadmissible or absorbing parameterisation loses the point of
the exercise.

:func:`box_to_disk` is the disk analogue of feax's
:func:`~feax.mechanics.orientation.box_to_triangle` and
:func:`~feax.mechanics.orientation.box_to_tetrahedron`: an isoparametric map
from the design box ``[-1, 1]^2`` that an optimiser can box-constrain directly.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from feax.mechanics.orientation import (_serendipity_8, box_to_triangle,
                                        smooth_sgn)

__all__ = ["box_to_disk", "unit_trace_a2", "unit_trace_a2_3d",
           "orientation_conductivity", "orientation_cte"]


def unit_trace_a2(t1, t2, sgn_beta: float = 10.0):
    """Admissible 2-D ``a_2`` with trace exactly 1, from two box variables.

        a11 = (t1 + 1)/2,   a22 = 1 - a11,
        a12 = sqrt(a11 a22) * smooth_sgn(t2)

    ``t1`` sweeps the principal direction and the degree of alignment (``a11=0``
    all fibres along y, ``1`` all along x, ``1/2`` spread); ``t2`` sweeps the
    off-diagonal over its full admissible range, since a trace-1 PSD 2x2 tensor
    needs exactly ``a12^2 <= a11 a22``. Rank-1 (perfect alignment) is reached at
    ``|t2| -> 1`` and at ``t1 = +-1``.

    Why not feax's ``orientation_tensor_2d``
    ----------------------------------------
    That one is built for a SUB-unit trace, ``tau = a11 + a22 <= 1``, where the
    deficit encodes "less fibre" and is fed back in as a SIMP-style alignment
    magnitude. Here the amount of material is already the density variable, so
    the orientation field only has to carry direction -- and normalising its
    output by ``tau`` is singular at the ``tau = 0`` vertex of the triangle its
    box map reaches. Parameterising trace-1 directly avoids that, and costs one
    design variable per node less (2 instead of 3) with no loss of reach.
    """
    a11 = 0.5 * (t1 + 1.0)
    a22 = 1.0 - a11
    a12 = jnp.sqrt(jnp.maximum(a11 * a22, 0.0)) * smooth_sgn(t2, sgn_beta)
    return jnp.array([[a11, a12], [a12, a22]])


def orientation_conductivity(a2, k_par: float, k_perp: float):
    """Orientation-averaged conductivity ``k_ij = k_perp d_ij + (k_par-k_perp) a2_ij``.

    The exact second-order orientation average for a transversely isotropic
    constituent, valid for ``trace(a2) = 1`` (use :func:`unit_trace_a2`).

    This is the whole reason anisotropy is interesting for thermal isolation:
    conductivity is a SECOND-order average and so depends on ``a_2`` alone,
    while stiffness is a fourth-order one depending on ``a_2`` AND ``a_4``. In a
    problem whose temperature field is one-dimensional, the direction transverse
    to the gradient is isothermal -- so putting the fast axis there costs
    nothing thermally while still delivering ``E1`` to whatever structural duty
    that direction carries.
    """
    return k_perp * jnp.eye(a2.shape[0]) + (k_par - k_perp) * a2

# Image of the eight ``[-1, 1]^2`` nodes, in the node order of
# ``_serendipity_8``: corner, midside, corner, midside, ... starting at
# ``(-1, -1)`` and going counter-clockwise.
#
#   N1 (-1,-1) corner   -> (-s, -s)      N2 ( 0,-1) midside -> ( 0, -1)
#   N3 (+1,-1) corner   -> ( s, -s)      N4 (+1, 0) midside -> ( 1,  0)
#   N5 (+1,+1) corner   -> ( s,  s)      N6 ( 0,+1) midside -> ( 0,  1)
#   N7 (-1,+1) corner   -> (-s,  s)      N8 (-1, 0) midside -> (-1,  0)
#
# The corners go to the 45-degree points of the circle and the midsides to the
# axis points, so the element is convex and well-shaped -- which is what keeps
# its Jacobian away from zero (see the note in :func:`box_to_disk`).
_S = 1.0 / jnp.sqrt(2.0)
_DISK_Y1 = jnp.array([-_S, 0.0, _S, 1.0, _S, 0.0, -_S, -1.0])
_DISK_Y2 = jnp.array([-_S, -1.0, -_S, 0.0, _S, 1.0, _S, 0.0])


def box_to_disk(x1, x2) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map the design square ``[-1, 1]^2`` into the closed unit disk.

    Smooth 8-node quadratic serendipity interpolation, the same construction
    feax uses for ``box_to_triangle``. Feed the result straight to
    ``build_a4_2d(a2, y1, y2)``.

    Parameters
    ----------
    x1, x2 : scalar or array in ``[-1, 1]``
        Box-bounded design variables. An optimiser only needs plain bounds on
        these; no nonlinear constraint is required.

    Returns
    -------
    y1, y2
        Coordinates of ``rho_2``, satisfying ``y1**2 + y2**2 <= 1``.

    Why an isoparametric map and not a closed-form one
    --------------------------------------------------
    The obvious analytic alternative is the elliptical ("squircle") map

        y1 = x1 sqrt(1 - x2^2/2),    y2 = x2 sqrt(1 - x1^2/2),

    which is exactly ONTO the disk -- strictly better coverage than this one.
    It is also unusable here: its Jacobian determinant is
    ``(4 A B - x1^2 x2^2) / (4 sqrt(A B))`` with ``A = 1 - x1^2/2``,
    ``B = 1 - x2^2/2``, which VANISHES at all four box corners. A design
    variable that reaches a corner has zero gradient there and can never leave
    -- the same absorbing-bound failure a ``tanh`` reparameterisation has, and
    the reason feax's orientation maps are isoparametric rather than
    closed-form. This map's Jacobian stays bounded away from zero everywhere on
    the box, corners included.

    The price is coverage. Each box edge maps to a quadratic through three
    circle points, so it touches ``|rho_2| = 1`` exactly at the four axis points
    and sags to about ``0.99`` in between; the image is a very slightly shrunken
    disk. That is the right way to be wrong -- staying inside keeps every
    reachable state admissible, and ``|rho_2| = 1`` is the singular limit
    ``build_a4_2d`` warns about anyway, where the inferred fibre distribution
    collapses to two Dirac peaks.

    Measured on a 401x401 sweep of the box: ``max |rho_2| = 1.000000``, no point
    outside the disk, and ``min |det J| = 0.157`` (including at the corners,
    where the squircle map gives ``2e-16``).
    """
    N = _serendipity_8(x1, x2)
    return jnp.tensordot(_DISK_Y1, N, axes=(0, 0)), jnp.tensordot(
        _DISK_Y2, N, axes=(0, 0))


def orientation_cte(a2, alpha_par: float, alpha_perp: float):
    """Orientation-averaged thermal expansion ``alpha_ij``.

    Same second-order average as :func:`orientation_conductivity`, and for the
    same reason: expansion is a second-order tensor property, so it depends on
    ``a_2`` alone::

        alpha_ij = alpha_perp d_ij + (alpha_par - alpha_perp) a2_ij

    For a carbon lamina the two constants have OPPOSITE SIGNS -- near zero or
    slightly negative along the fibre, strongly positive across it -- so
    orientation does not merely scale the contraction, it can reverse its
    direction. That gives the orientation field a third job, independent of the
    two it already has (conductivity, which is also second-order and so shares
    ``a_2`` exactly, and stiffness, which is fourth-order).
    """
    return alpha_perp * jnp.eye(a2.shape[0]) + (alpha_par - alpha_perp) * a2


def unit_trace_a2_3d(t1, t2, t12, t13, sgn_beta: float = 10.0):
    """Admissible 3-D ``a_2`` with trace exactly 1, from FOUR box variables.

        (a11, a22) = box_to_triangle(t1, t2),   a33 = 1 - a11 - a22
        s12 = smooth_sgn(t12),  s13 = smooth_sgn(t13),  s23 = s12 * s13
        a_ij = sqrt(a_ii a_jj) * s_ij                         (i != j)

    The 3-D analogue of :func:`unit_trace_a2`, and it needs the same trace-1
    treatment for the same reason: the amount of material is already the density
    variable, so the orientation field only carries direction, and the sub-unit
    trace of ``feax.mechanics.orientation.orientation_tensor_3d`` would encode
    "less fibre" a second time.

    WHY ``s23`` IS NOT ITS OWN DESIGN VARIABLE. Taking the three off-diagonal
    signs independently -- which is what the six-variable Nomura construction
    does -- almost never produces a positive semi-definite tensor. Over 20000
    uniform samples of the box, 97.7% of the results had a negative eigenvalue,
    down to -1/3. The reason is that a rank-1 state ``a_2 = v v^T`` has
    ``s12 s13 s23 = sign(v1^2 v2^2 v3^2) = +1``, so an odd number of negative
    signs is not reachable by any fibre direction at all.

    Tying ``s23 = s12 s13`` removes exactly that inconsistency, and the result
    is PSD everywhere rather than almost nowhere: writing ``a_2 = D^(1/2) M
    D^(1/2)`` with ``D = diag(a11, a22, a33)`` and ``M`` the correlation matrix,

        det M = 1 + 2 m12 m13 m23 - m12^2 - m13^2 - m23^2
              = (1 - m12^2)(1 - m13^2)  >= 0      when m23 = m12 m13

    and the leading minors are ``1`` and ``1 - m12^2``, so ``M`` is PSD for
    every value the smooth sign can take, not only at its limits. Measured over
    the same 20000 samples: minimum eigenvalue ``+3.7e-14``, none negative.

    The cost is one degree of freedom. Trace-1 symmetric PSD 3x3 tensors form a
    5-parameter set and this map covers a 4-parameter subset of it. What the
    subset contains is the part that matters here: EVERY rank-1 state (the
    two-parameter family ``v v^T``, reached as the smooth signs sharpen -- check
    the eigenvalues, they go to ``(0, 0, 1)``) and the isotropic state. What it
    gives up is part of the rank-2/3 interior, which a converged design should
    not be sitting in.
    """
    a11, a22 = box_to_triangle(t1, t2)
    a33 = 1.0 - a11 - a22
    s12 = smooth_sgn(t12, sgn_beta)
    s13 = smooth_sgn(t13, sgn_beta)
    s23 = s12 * s13
    a12 = jnp.sqrt(jnp.maximum(a11 * a22, 0.0)) * s12
    a13 = jnp.sqrt(jnp.maximum(a11 * a33, 0.0)) * s13
    a23 = jnp.sqrt(jnp.maximum(a22 * a33, 0.0)) * s23
    return jnp.array([[a11, a12, a13],
                      [a12, a22, a23],
                      [a13, a23, a33]])
