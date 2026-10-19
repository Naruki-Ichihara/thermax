"""Sweep curves against the compliance limit, from outputs_no_length_scale/.

Two figures, one measure each: heat leak Q in watts, and volume fraction. Both
are normalised by the quasi-isotropic solid plate, so the reference sits at the
same place in each.

House style: thermax.use_nature_style (Arial-family, black text, inward ticks,
NPG palette). Writes .png, .pdf and the table view beside the data.
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "outputs_no_length_scale")
C_VALUES = (1, 2, 4, 8, 10)
SERIES = (("aniso", "", "Anisotropic", "o", True),
          ("iso", "_iso", "Isotropic", "s", False))


def read(tag_suffix, C):
    """One run's results, or None if it did not finish."""
    d = os.path.join(OUT, f"thermal_isolation_2d_C{C:g}_m1{tag_suffix}")
    hist = os.path.join(d, "history.csv")
    if not os.path.isfile(hist):
        return None
    with open(hist) as fh:
        last = list(csv.DictReader(fh))[-1]
    # heat is logged as a ratio; heat_ref_W converts it to watts and is stored
    # with the run, so the figure cannot drift from the run that produced it.
    with h5py.File(os.path.join(d, "snapshots.h5"), "r") as h:
        ref_W = float(h.attrs["heat_ref_W"])
    # The whole-cell mean, not history's "volume", which covers the design
    # region only. Against the solid QI plate the whole cell is the fair
    # denominator, and it puts the reference at 1.0 as the heat panel does.
    rho = onp.load(os.path.join(d, "density.npy")).astype(float)
    return dict(Q=float(last["heat"]) * ref_W,
                V=float(rho.mean()),
                C_achieved=float(last["compliance"]),
                ref_W=ref_W)


# (key, y label, reference value, where the reference label sits)
FIGURES = [
    ("Q", "Heat leak, $Q$ (W)", "ref_W", "Q_vs_compliance"),
    ("V", "Volume fraction, $V/V_\\mathrm{QI}$", 1.0, "V_vs_compliance"),
]

tx.use_nature_style()
red, navy = tx.series_colors(2)
COLOR = {"aniso": red, "iso": navy}

data = {}
for key, sfx, label, marker, filled in SERIES:
    for C in C_VALUES:
        got = read(sfx, C)
        if got is not None:
            data[(key, C)] = got

for field, ylabel, ref, stem in FIGURES:
    fig, ax = plt.subplots(figsize=(89 * tx.MM, 72 * tx.MM))
    ref_value = None
    for key, sfx, label, marker, filled in SERIES:
        pts = [(C, data[(key, C)]) for C in C_VALUES if (key, C) in data]
        if not pts:
            continue
        ref_value = (pts[0][1][ref] if isinstance(ref, str) else ref)
        ax.plot([c for c, _ in pts], [r[field] for _, r in pts],
                color=COLOR[key], marker=marker, label=label,
                markerfacecolor=COLOR[key] if filled else "white",
                markeredgecolor=COLOR[key], markeredgewidth=0.9,
                clip_on=False, zorder=3)

    # The quasi-isotropic solid plate: the most a design of this material can
    # leak, and the most it can weigh.
    if ref_value is not None:
        ax.axhline(ref_value, color=tx.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
        # Centred under the line, where neither curve nor legend runs. The gap
        # is set in points, not data units, so it survives a change of y range.
        ax.annotate("QI plate", xy=(5.5, ref_value), xytext=(0, -5),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6.5)

    ax.set_xlabel("Compliance limit, $C/C_\\mathrm{QI}$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, None)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.96))

    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)

# ── The trade-off plane ────────────────────────────────────────────────────
# Heat and mass are both costs, so neither axis alone ranks a design. Plotted
# against each other, with the compliance limit as the curve parameter, "better"
# is simply down and to the left, and no weighting has to be chosen to say so.
#
# The faint straight lines are constant EQUIVALENT MASS: structural mass plus
# the hydrogen a leak boils off over a mission. That converts heat into mass at
# a rate physics fixes rather than taste -- see the slope below.
RHO_C = 0.61 * 1.79 + 0.39 * 1.30        # g/cm3, AS4/PEEK at Vf = 0.61
H_FG = 445.7e3                            # J/kg, LH2 at 1 atm, 20.3 K
VOL_MM3 = 150.0 * 115.0 * 4.0
MISSION_H = 4.0
M_QI_G = RHO_C * 1e-3 * VOL_MM3           # mass of the solid QI plate, g
G_PER_W = 1e3 / H_FG * 3600 * MISSION_H   # boil-off, g per W over the mission

fig, ax = plt.subplots(figsize=(89 * tx.MM, 72 * tx.MM))
for level in (20, 40, 60, 80, 100):       # grams of equivalent mass
    v = onp.linspace(0, 1.05, 2)
    ax.plot(v, (level - M_QI_G * v) / G_PER_W, color="0.75", lw=0.5,
            zorder=0, clip_on=True)
    ax.annotate(f"{level} g", xy=(0.02, (level - M_QI_G * 0.02) / G_PER_W),
                xytext=(1, 1), textcoords="offset points", fontsize=5.5,
                color="0.55", ha="left", va="bottom")

for key, sfx, label, marker, filled in SERIES:
    pts = [(C, data[(key, C)]) for C in C_VALUES if (key, C) in data]
    ax.plot([r["V"] for _, r in pts], [r["Q"] for _, r in pts],
            color=COLOR[key], marker=marker, label=label,
            markerfacecolor=COLOR[key] if filled else "white",
            markeredgecolor=COLOR[key], markeredgewidth=0.9, zorder=3)
    for C, r in pts:
        # Labels outside the pair of loci: anisotropic to the left, isotropic
        # to the right, so the cluster at low V stays readable.
        dx = -5 if key == "aniso" else 5
        ax.annotate(f"{C:g}", xy=(r["V"], r["Q"]), xytext=(dx, 2),
                    textcoords="offset points", fontsize=5.5,
                    ha="right" if key == "aniso" else "left",
                    color=COLOR[key])

ax.set_xlabel("Volume fraction, $V/V_\\mathrm{QI}$")
ax.set_ylabel("Heat leak, $Q$ (W)")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 3.9)
ax.annotate("better", xy=(0.10, 0.40), xytext=(0.27, 1.05),
            fontsize=6.5, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=tx.INK))
# The lower right of this plane is empty -- no design is both heavy and cool --
# so the legend and the note live there, clear of the loci and their labels.
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13))
ax.annotate(f"point labels: $C/C_\\mathrm{{QI}}$\n"
            f"grey lines: equivalent mass, {MISSION_H:g} h mission",
            xy=(0.98, 0.03), xycoords="axes fraction", fontsize=5.5,
            ha="right", va="bottom", color="0.35")
for ext in ("png", "pdf"):
    q = os.path.join(OUT, f"tradeoff_Q_vs_V.{ext}")
    fig.savefig(q)
    print(f"wrote {q}")
plt.close(fig)

with open(os.path.join(OUT, "sweep_curves.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["C_limit", "orientation", "Q_W", "V_fraction",
                "C_over_CQI_achieved", "m_structure_g",
                "m_boiloff_4h_g", "m_equivalent_4h_g"])
    for (key, C), r in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        name = next(s[2] for s in SERIES if s[0] == key)
        ms, mb = M_QI_G * r["V"], G_PER_W * r["Q"]
        w.writerow([C, name, r["Q"], r["V"], r["C_achieved"],
                    ms, mb, ms + mb])
print(f"wrote {os.path.join(OUT, 'sweep_curves.csv')}")
