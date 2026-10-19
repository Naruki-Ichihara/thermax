"""Marching-cubes renders of the 3-D optimisation results.

The nodal density field is contoured at rho = 0.5 with
``skimage.measure.marching_cubes`` and the triangles are rendered offscreen by
PyVista. Two panels: anisotropic and isotropic, at the same compliance limit,
same camera, same scale.

Offscreen rendering needs OSMesa (``apt-get install libosmesa6``); without it
VTK has no GL context and the process aborts.

House style: thermax.use_nature_style for the composed figure. The renders
themselves are lit 3-D, so the palette only sets the surface colours.
"""
from __future__ import annotations

import csv
import os

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
os.environ.pop("DISPLAY", None)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
import pyvista as pv
from skimage.measure import marching_cubes

import feax as fe
import thermax as tx

pv.OFF_SCREEN = True

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
RUNS = [
    ("thermal_isolation_3d_aniso_uncoupled_C8_m1_cmg", "Anisotropic",
     tx.NPG["red"]),
    ("thermal_isolation_3d_aniso_uncoupled_C8_m1_iso_cmg", "Isotropic",
     tx.NPG["navy"]),
]
LEVEL = 0.5
PIX = (1100, 1100)


def surface(name):
    """(PolyData at rho = 0.5, Q in W, V) for one run."""
    d = os.path.join(OUT, name)
    with h5py.File(os.path.join(d, "snapshots.h5"), "r") as h:
        a = h.attrs
        L = (float(a["LX"]), float(a["LY"]), float(a["LZ"]))
        hsz, ref_W = float(a["ELEM_SIZE"]), float(a["heat_ref_W"])
    with open(os.path.join(d, "history.csv")) as fh:
        last = list(csv.DictReader(fh))[-1]

    mesh = fe.mesh.box_mesh(size=L, mesh_size=hsz, element_type="HEX8")
    pts = onp.asarray(mesh.points)
    # Node coordinates are a tensor grid but not necessarily in index order, so
    # bin by coordinate rather than trusting the numbering.
    axes = [onp.unique(pts[:, k]) for k in range(3)]
    idx = [onp.searchsorted(axes[k], pts[:, k]) for k in range(3)]
    field = onp.zeros([len(ax) for ax in axes])
    field[idx[0], idx[1], idx[2]] = onp.load(
        os.path.join(d, "density_nodal.npy")).astype(float)

    spacing = tuple(float(onp.diff(ax)[0]) for ax in axes)
    # Pad with void before contouring. The passive bands run to the domain
    # boundary at rho = 1, so without a layer of zeros outside the field never
    # crosses 0.5 there and marching cubes leaves the solid OPEN -- the render
    # then shows the inner face of each band as a bare plate. One zero layer
    # closes the surface, and the shift below undoes the padding offset.
    field = onp.pad(field, 1, mode="constant", constant_values=0.0)
    verts, faces, normals, _ = marching_cubes(field, level=LEVEL,
                                              spacing=spacing)
    verts = verts - onp.asarray(spacing)
    tri = onp.hstack([onp.full((len(faces), 1), 3), faces]).astype(onp.int64)
    poly = pv.PolyData(verts, tri)
    poly.point_data["Normals"] = normals
    rho = onp.load(os.path.join(d, "density.npy")).astype(float)
    return poly, float(last["heat"]) * ref_W, float(rho.mean()), L


def render(poly, color, L):
    p = pv.Plotter(off_screen=True, window_size=PIX)
    p.set_background("white")
    p.add_mesh(poly, color=color, smooth_shading=True, specular=0.25,
               specular_power=18, ambient=0.28, diffuse=0.85)
    # One camera for both panels, so the two are read at the same scale: an
    # isometric direction, then reset_camera fits the bounds and zoom trims the
    # margin. y is the heat and load path, so it points up on screen.
    p.camera_position = [(2.2 * L[0], 1.6 * L[1], 1.9 * L[2]),
                         (L[0] / 2, L[1] / 2, L[2] / 2),
                         (0.0, 1.0, 0.0)]
    p.reset_camera()
    p.camera.zoom(0.88)
    p.enable_anti_aliasing("ssaa")
    img = p.screenshot(return_img=True)
    p.close()
    return img


tx.use_nature_style()
panels = []
for name, label, color in RUNS:
    poly, Q, V, L = surface(name)
    panels.append((label, render(poly, color, L), Q, V, poly.n_faces))
    print(f"{label:12s} {poly.n_points:,} vertices  {poly.n_faces:,} "
          f"triangles  Q={Q:.3f} W  V={V:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(183 * tx.MM, 95 * tx.MM))
for ax, (label, img, Q, V, ntri), tag in zip(axes, panels, "ab"):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"({tag})  {label}", loc="left", pad=3)
    ax.annotate(f"$Q$ = {Q:.2f} W\n$V/V_\\mathrm{{ref}}$ = {V:.3f}",
                xy=(0.02, 0.02), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=6.5)

fig.suptitle("3-D bracket, $50^3$ mm, 1.5 mm elements, "
             "$C/C_\\mathrm{ref} = 8$;  isosurface at $\\rho = 0.5$",
             fontsize=8, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"render_3d_C8.{ext}")
    fig.savefig(p)
    print(f"wrote {p}")
plt.close(fig)
