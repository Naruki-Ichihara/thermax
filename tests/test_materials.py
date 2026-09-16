import jax.numpy as jnp

from thermax import __version__
from thermax.materials import simp_conductivity


def test_version():
    assert __version__


def test_simp_endpoints():
    # No x64 here on purpose: the tolerances are float32-safe so the test says
    # nothing about whichever precision the caller happens to have enabled.
    k = simp_conductivity(jnp.array([0.0, 1.0]), k_solid=50.0, k_min=1e-3, penal=3.0)
    assert abs(float(k[0]) - 1e-3) < 1e-9
    assert abs(float(k[1]) - 50.0) < 1e-4


def test_simp_monotone():
    rho = jnp.linspace(0.0, 1.0, 11)
    k = simp_conductivity(rho, k_solid=50.0, k_min=1e-3)
    assert bool(jnp.all(jnp.diff(k) > 0.0))
