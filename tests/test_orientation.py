import jax
import jax.numpy as jnp
import pytest

from feax.mechanics.orientation import build_a4_2d, quadratic_closure
from thermax.orientation import box_to_disk

GRID = jnp.linspace(-1.0, 1.0, 121)


def _sweep():
    X1, X2 = jnp.meshgrid(GRID, GRID, indexing="ij")
    y1, y2 = box_to_disk(X1, X2)
    return jnp.sqrt(y1**2 + y2**2)


def test_image_is_inside_the_unit_disk():
    """Every reachable rho_2 must be admissible; build_a4_2d does not check."""
    assert float(_sweep().max()) <= 1.0 + 1e-12


def test_reaches_the_boundary():
    """A map that only covered the middle of the disk would throw away the
    cross-ply states that are the whole point."""
    assert float(_sweep().max()) > 0.999


def test_axis_points_map_exactly_to_the_circle():
    for x1, x2, want in [(1.0, 0.0, (1.0, 0.0)), (-1.0, 0.0, (-1.0, 0.0)),
                         (0.0, 1.0, (0.0, 1.0)), (0.0, -1.0, (0.0, -1.0))]:
        y1, y2 = box_to_disk(x1, x2)
        assert float(y1) == pytest.approx(want[0], abs=1e-12)
        assert float(y2) == pytest.approx(want[1], abs=1e-12)


def _det_jacobian(fn, a, b):
    J = jax.jacobian(lambda v: jnp.stack(fn(v[0], v[1])))(jnp.array([a, b]))
    return float(jnp.linalg.det(J))


def test_jacobian_never_degenerates():
    """No absorbing states: a design variable that reaches a bound must still
    have a gradient. This is why the map is isoparametric and not the
    closed-form squircle map, whose Jacobian vanishes at all four corners.
    """
    worst = min(abs(_det_jacobian(box_to_disk, float(a), float(b)))
                for a in GRID[::12] for b in GRID[::12])
    assert worst > 1e-2

    def squircle(a, b):
        return a * jnp.sqrt(1.0 - b**2 / 2.0), b * jnp.sqrt(1.0 - a**2 / 2.0)

    assert abs(_det_jacobian(squircle, 1.0, 1.0)) < 1e-12


def test_a4_moves_stiffness_at_fixed_a2():
    """The freedom this map exists to expose.

    At ``a_2 = diag(1/2, 1/2)`` the orientation-averaged CONDUCTIVITY is fixed
    (it depends on ``a_2`` alone), yet ``rho_2`` sweeps the fourth-order tensor
    from the +-45 cross-ply to the 0/90 one.
    """
    a2 = jnp.array([[0.5, 0.0], [0.0, 0.5]])
    a4_m = build_a4_2d(a2, *box_to_disk(-1.0, 0.0))     # rho_2 = -1
    a4_p = build_a4_2d(a2, *box_to_disk(1.0, 0.0))      # rho_2 = +1
    assert float(a4_m[0, 0, 0, 0]) == pytest.approx(0.25, abs=1e-12)
    assert float(a4_p[0, 0, 0, 0]) == pytest.approx(0.50, abs=1e-12)

    # ... and the quadratic closure the feax examples use lands on the -45 end,
    # i.e. the softest admissible state, not a neutral one.
    a4_q = quadratic_closure(a2)
    assert float(a4_q[0, 0, 0, 0]) == pytest.approx(0.25, abs=1e-12)


def test_contraction_identity_holds():
    """``a4_ijkk == a2_ij`` exactly, at sub-unit trace where closures fail."""
    a2 = jnp.array([[0.3, 0.1], [0.1, 0.2]])           # trace 0.5
    a4 = build_a4_2d(a2, *box_to_disk(0.6, -0.4))
    err = jnp.abs(jnp.trace(a4, axis1=2, axis2=3) - a2).max()
    assert float(err) < 1e-12
