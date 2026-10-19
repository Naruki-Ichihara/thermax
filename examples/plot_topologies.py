"""Four topologies: C/C_QI = 2 and 8, anisotropic and quasi-isotropic.

Density is greyscale, void white and solid black. The fibre director is drawn
in gold, only on the anisotropic panels -- the quasi-isotropic runs hold it
fixed at a_2 = I/2, so a director there would be a picture of a constant.

House style: thermax.use_nature_style. Field panels carry no coordinate ticks;
a scale bar gives the size instead.
"""
from __future__ import annotations

import csv
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp

import feax as fe
from feax.mechanics.orientation import principal_direction
import jax.numpy as jnp
import thermax as tx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs_no_length_scale")
LX, LY, H = 150.0, 115.0, 1.0
NX, NY = int(round(LX / H)), int(round(LY / H))
CASES = [(2, "aniso", ""), (8, "aniso", ""),
         (2, "iso", "_iso"), (8, "iso", "_iso")]
NAME = {"aniso": "Anisotropic", "iso": "Quasi-isotropic"}

mesh = fe.mesh.rectangle_mesh(Nx=NX, Ny=NY, domain_x=LX, domain_y=LY,
                              ele_type="QUAD4")
pts = onp.asarray(mesh.points)
cells = onp.asarray(mesh.cells)
cen = pts[cells].mean(axis=1)
ix = onp.clip(((cen[:, 0] - H / 2) / H).round().astype(int), 0, NX - 1)
iy = onp.clip(((cen[:, 1] - H / 2) / H).round().astype(int), 0, NY - 1)


def load(C, sfx):
    d = os.path.join(OUT, f"thermal_isolation_2d_C{C:g}_m1{sfx}")
    rho = onp.load(os.path.join(d, "density.npy")).astype(float)
    a2 = onp.load(os.path.join(d, "orientation.npy")).astype(float)
    with open(os.path.join(d, "history.csv")) as fh:
        last = list(csv.DictReader(fh))[-1]
    with h5py.File(os.path.join(d, "snapshots.h5"), "r") as h:
        ref_W = float(h.attrs["heat_ref_W"])
    grid = onp.zeros((NX, NY))
    grid[ix, iy] = rho
    return grid, a2, float(last["heat"]) * ref_W, float(rho.mean())


tx.use_nature_style()
# The director is an overlay on a black density field, not a categorical
# series, so it is picked for contrast there rather than from the NPG set:
# gold reads at 15:1 against solid material.
DIRECTOR = "#FFD700"

fig, axes = plt.subplots(2, 2, figsize=(183 * tx.MM, 160 * tx.MM))
for ax, (C, kind, sfx), tag in zip(axes.ravel(), CASES, "abcd"):
    grid, a2, Q, V = load(C, sfx)
    # origin="lower" with a transposed grid puts x across and y up, as the mesh
    # has it; without the transpose the bracket would lie on its side.
    ax.imshow(grid.T, origin="lower", extent=(0, LX, 0, LY), cmap="gray_r",
              vmin=0.0, vmax=1.0, interpolation="nearest")

    if kind == "aniso":
        d = onp.asarray(principal_direction(jnp.asarray(a2)))
        # One glyph every 3 mm, and only inside material: a director drawn
        # across the void would show a fibre direction where there is no fibre.
        step = 3
        sel = ((onp.rint(pts[:, 0]) % step == 0)
               & (onp.rint(pts[:, 1]) % step == 0))
        rho_n = grid[onp.clip(onp.rint(pts[:, 0] - H / 2).astype(int), 0, NX - 1),
                    onp.clip(onp.rint(pts[:, 1] - H / 2).astype(int), 0, NY - 1)]
        sel &= rho_n > 0.5
        ax.quiver(pts[sel, 0], pts[sel, 1], d[sel, 0], d[sel, 1],
                  color=DIRECTOR, pivot="mid", headwidth=0, headlength=0,
                  headaxislength=0, width=0.0026, scale=50, alpha=0.9)

    ax.set_xlim(0, LX)
    ax.set_ylim(0, LY)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"({tag})  {NAME[kind]},  $C/C_\\mathrm{{QI}} = {C:g}$",
                 loc="left", pad=4)
    ax.annotate(f"$Q$ = {Q:.2f} W\n$V/V_\\mathrm{{QI}}$ = {V:.3f}",
                xy=(0.985, 0.975), xycoords="axes fraction", ha="right",
                va="top", fontsize=6.5,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none",
                          alpha=0.85))

# One scale bar, on the first panel. It sits over the passive band, which is
# solid black, so the bar is drawn white on a black backing and the label wears
# a white box -- legible wherever it lands.
bar, x0, y0 = 30.0, 8.0, 8.0
a0 = axes[0, 0]
a0.plot([x0, x0 + bar], [y0, y0], color=tx.INK, lw=3.4, solid_capstyle="butt",
        zorder=5)
a0.plot([x0, x0 + bar], [y0, y0], color="white", lw=1.6, solid_capstyle="butt",
        zorder=6)
a0.annotate(f"{bar:g} mm", xy=(x0 + bar / 2, y0 + 3), ha="center", va="bottom",
            fontsize=6.5, zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"topologies_C2_C8.{ext}")
    fig.savefig(p)
    print(f"wrote {p}")
plt.close(fig)
