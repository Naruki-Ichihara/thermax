"""thermax -- thermal and thermo-mechanical topology optimization on top of feax.

Scope
-----
Scalar diffusion (``vec=1``) and its coupling to elasticity, packaged as reusable
pieces rather than as one-off scripts:

    materials    density -> conductivity interpolation (SIMP and friends)
    physics      the FE side: gradient operator, Fourier flux, ``flux_fn`` /
                 ``state_fn`` pairs in the shape ``feax.NarrowBandCMG`` wants
    objectives   thermal compliance, heat throughput, and their sensitivities
    optim        the update rules, and which objective each one is valid for

Nothing here re-implements feax. ``feax.NarrowBandCMG`` already solves scalar
diffusion -- it is the same matrix-free MGPCG as elasticity with ``vec=1``, a
gradient ``B`` instead of a strain-displacement one, and Fourier instead of
Hooke. thermax is the layer above: the problem setups, the objectives, and the
bookkeeping that the three prototype examples had to repeat.
"""

__version__ = "0.0.1"

from .materials import (ramp_conductivity, ramp_modulus,
                        simp_conductivity, simp_modulus)
from .io import SnapshotWriter, snapshots_to_xdmf
from .objectives import create_conduction_energy_fn
from .orientation import (box_to_disk, orientation_conductivity,
                          orientation_cte, unit_trace_a2, unit_trace_a2_3d)
from .plotstyle import INK, MM, NPG, series_colors, use_nature_style
from .stress import (adaptive_scale, create_stress_tensor_fn,
                     create_vonmises_fn, pnorm)
from .physics import (
    anisotropic_diffusion_problem,
    anisotropic_elasticity_problem,
    anisotropic_stress_measure,
    anisotropic_diffusion_problem_3d,
    anisotropic_elasticity_problem_3d,
    anisotropic_stress_measure_3d,
    diffusion_problem,
    elasticity_problem,
)

__all__ = [
    "__version__",
    "simp_conductivity",
    "simp_modulus",
    "ramp_conductivity",
    "ramp_modulus",
    "diffusion_problem",
    "elasticity_problem",
    "anisotropic_diffusion_problem",
    "anisotropic_elasticity_problem",
    "anisotropic_stress_measure",
    "anisotropic_diffusion_problem_3d",
    "anisotropic_elasticity_problem_3d",
    "anisotropic_stress_measure_3d",
    "create_stress_tensor_fn",
    "create_vonmises_fn",
    "pnorm",
    "adaptive_scale",
    "create_conduction_energy_fn",
    "SnapshotWriter",
    "snapshots_to_xdmf",
    "box_to_disk",
    "unit_trace_a2",
    "unit_trace_a2_3d",
    "orientation_conductivity",
    "orientation_cte",
    "NPG",
    "INK",
    "MM",
    "series_colors",
    "use_nature_style",
]
