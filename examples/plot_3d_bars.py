"""The 3-D results at C/C_ref = 8, as bars.

The solid block is the leftmost bar rather than a reference line, so the scale
is what the optimisation starts from and the two designs are read against it
directly. It is the SPATIALLY RANDOM solid (a_2 = I/3), not a quasi-isotropic
laminate -- a 50 mm block is not a laminate.

House style: thermax.use_nature_style.
"""
from __future__ import annotations

import csv
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp

import thermax as tx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# (directory or None for the solid reference, label, colour)
RUNS = [
    (None, "Solid block", "0.62"),
    ("thermal_isolation_3d_aniso_uncoupled_C8_m1_iso_cmg", "Isotropic",
     tx.NPG["navy"]),
    ("thermal_isolation_3d_aniso_uncoupled_C8_m1_cmg", "Anisotropic",
     tx.NPG["red"]),
]

# Equivalent mass: structure plus the hydrogen a leak boils off over a mission.
RHO_C = 0.61 * 1.79 + 0.39 * 1.30        # g/cm3, AS4/PEEK at Vf = 0.61
H_FG = 445.7e3                            # J/kg, LH2 at 1 atm, 20.3 K
MISSION_H = 4.0
G_PER_W = 1e3 / H_FG * 3600 * MISSION_H


def load(name):
    d = os.path.join(OUT, name)
    with open(os.path.join(d, "history.csv")) as fh:
        last = list(csv.DictReader(fh))[-1]
    with h5py.File(os.path.join(d, "snapshots.h5"), "r") as h:
        a = h.attrs
        ref_W = float(a["heat_ref_W"])
        vol_mm3 = float(a["LX"]) * float(a["LY"]) * float(a["LZ"])
    rho = onp.load(os.path.join(d, "density.npy")).astype(float)
    V = float(rho.mean())
    m_solid = RHO_C * 1e-3 * vol_mm3
    Q = float(last["heat"]) * ref_W
    return dict(Q=Q, V=V, ref_W=ref_W, m_struct=m_solid * V,
                m_boil=G_PER_W * Q, m_solid=m_solid)


tx.use_nature_style()
data = []
for name, lab, col in RUNS:
    if name is None:
        continue                       # filled in below, off any finished run
    data.append((lab, col, load(name)))
# The solid block: V = 1 by definition and Q is the reference every run stores.
_any = data[0][2]
_solid = dict(Q=_any["ref_W"], V=1.0, ref_W=_any["ref_W"],
              m_struct=_any["m_solid"], m_boil=G_PER_W * _any["ref_W"],
              m_solid=_any["m_solid"])
data.insert(0, (RUNS[0][1], RUNS[0][2], _solid))
x = onp.arange(len(data))

PANELS = [("Q", "Heat leak, $Q$ (W)", "ref_W", "{:.2f}", "W"),
          ("V", "Volume fraction, $V/V_\\mathrm{ref}$", 1.0, "{:.3f}", "")]

fig, axes = plt.subplots(1, 2, figsize=(183 * tx.MM, 68 * tx.MM))
for ax, (field, ylabel, _ref, fmt, unit) in zip(axes, PANELS):
    for xi, (lab, col, r) in zip(x, data):
        ax.bar(xi, r[field], width=0.62, color=col, linewidth=0.0, zorder=3)
        ax.annotate(fmt.format(r[field]), xy=(xi, r[field]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5)
    ax.set_ylim(0, max(r[field] for _l, _c, r in data) * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _c, _r in data])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.65, len(data) - 0.35)

axes[0].set_title("(a)", loc="left", pad=4)
axes[1].set_title("(b)", loc="left", pad=4)
fig.suptitle("3-D bracket, $50^3$ mm, 1.5 mm elements, $C/C_\\mathrm{ref} = 8$",
             fontsize=8, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"bars_3d_C8.{ext}")
    fig.savefig(p)
    print(f"wrote {p}")
plt.close(fig)

with open(os.path.join(OUT, "bars_3d_C8.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "Q_W", "V_fraction", "m_structure_g",
                f"m_boiloff_{MISSION_H:g}h_g", "m_equivalent_g"])
    for lab, _c, r in data:
        w.writerow([lab, r["Q"], r["V"], r["m_struct"], r["m_boil"],
                    r["m_struct"] + r["m_boil"]])
print(f"wrote {os.path.join(OUT, 'bars_3d_C8.csv')}")
for lab, _c, r in data:
    print(f"   {lab:14s} Q={r['Q']:7.3f} W  V={r['V']:.4f}  "
          f"m_eq={r['m_struct'] + r['m_boil']:6.1f} g")
