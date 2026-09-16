# thermax

Thermal and thermo-mechanical topology optimization on top of [feax](https://github.com/Naruki-Ichihara/feax).

Currently a scratch workspace for design work. It is laid out as a library so it
can be lifted out to its own repository unchanged: `/workspace/thermal` is the
repo root, `thermax/` is the package.

## Status

Skeleton. `thermax.materials` is the only module with content; the rest of the
layout below is the intended decomposition, not something that exists yet.

| module | contents |
|---|---|
| `materials` | density → conductivity interpolation (SIMP), and when `k_min` may be small |
| `physics` | hex8 gradient operator, Fourier flux, the `flux_fn` / `state_fn` pairs `feax.NarrowBandCMG` expects |
| `objectives` | thermal compliance, heat throughput, and their sensitivities |
| `optim` | update rules, and which objective sign each is valid for (OC vs MMA) |

## Why a layer at all

`feax.NarrowBandCMG` already solves scalar diffusion — it is the same
matrix-free MGPCG as elasticity with `vec=1`, a gradient `B` instead of a
strain-displacement one, and Fourier instead of Hooke. Nothing here
re-implements that. What the three prototype examples in
`feax/examples/advance` each had to repeat, and what belongs here, is the layer
above: problem setups, objectives, and the sensitivity bookkeeping.

The prototypes worth migrating:

- `heat_conduction_topopt_3d_periodic.py` — volume-to-point (dendritic), OC,
  narrow band **not** applicable (void generates heat)
- `heatsink_topopt_3d_periodic.py` — heat-sink spreader, OC with the sign
  flipped to maximise throughput, design-dependent RHS
- `cryo_support_topopt_3d.py` — heat leak vs shear stiffness, MMA (OC is invalid
  when the objective grows with material)

## Install / run

Not installed. `feax` is editable-installed from `/workspace`, so it imports
from anywhere; `thermax` does not, so run from this directory:

```bash
cd /workspace/thermal
python -m pytest            # tests
python examples/....py      # examples
```

or put it on the path explicitly:

```bash
PYTHONPATH=/workspace/thermal python your_script.py
```

Add `pip install -e .` only when this becomes its own repo — it writes a `.pth`
into the system `dist-packages`.

## License

GPL-3.0-or-later, inherited from feax.
