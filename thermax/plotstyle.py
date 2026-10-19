"""One figure style for every plot this package produces.

Arial-family sans, black text, ticks turned inward on all four sides, and the
Nature Publishing Group categorical palette. Call :func:`use_nature_style` once
before creating a figure; take colours from :data:`NPG` by slot.

The palette is the NPG "nrc" set. Slots are assigned in fixed order and never
cycled -- a ninth series folds into "Other" or becomes a small multiple rather
than acquiring a generated hue. :func:`series_colors` returns the pair or triple
that has been checked for colour-vision separation on a white surface; anything
wider than that should be re-checked before use.

Measured on the white chart surface, OKLab dE x 100 (Machado et al. 2009 CVD
matrices at full severity), for the default two-series pair red + navy:

    normal 31.3   protanopia 17.7   deuteranopia 28.8   tritanopia 36.9
    contrast against white: 3.87 and 7.46

which clears the usual gates (normal >= 15, CVD >= 8, contrast >= 3). Colour is
still never the only cue: series also differ in marker shape.
"""
from __future__ import annotations

from typing import List

__all__ = ["NPG", "INK", "use_nature_style", "series_colors", "MM"]

# Nature Publishing Group categorical palette, fixed order.
NPG = {
    "red": "#E64B35",
    "cyan": "#4DBBD5",
    "teal": "#00A087",
    "navy": "#3C5488",
    "salmon": "#F39B7F",
    "slate": "#8491B4",
    "mint": "#91D1C2",
    "crimson": "#DC0000",
    "brown": "#7E6148",
    "taupe": "#B09C85",
}

# All text and rules are black; the palette is reserved for the data marks.
INK = "#000000"

# Journal column widths, in inches, for figsize.
MM = 1.0 / 25.4
SINGLE_COLUMN = 89 * MM
DOUBLE_COLUMN = 183 * MM

# Checked pairs and triples, best separation first. Beyond three, re-check.
_SAFE = {
    2: ["red", "navy"],
    3: ["red", "navy", "cyan"],
}


def series_colors(n: int) -> List[str]:
    """Hex colours for ``n`` series, in fixed order.

    Up to three come from a pre-checked set. Beyond that the NPG order is
    returned as-is and the caller owns the separation check.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n in _SAFE:
        return [NPG[k] for k in _SAFE[n]]
    return [NPG[k] for k in list(NPG)[:n]]


def use_nature_style(base_size: float = 8.0) -> None:
    """Apply the house style to matplotlib's global rcParams.

    ``Arial`` heads the font list and metric-compatible substitutes follow, so
    the figure is laid out identically wherever any of them is installed.
    ``Liberation Sans`` and ``Nimbus Sans`` are metric-compatible with Arial;
    ``DejaVu Sans``, matplotlib's own default, is only a last resort and is
    wider, so text may reflow against a real Arial.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "Nimbus Sans", "Arimo", "FreeSans", "DejaVu Sans"],
        "font.size": base_size,
        "axes.labelsize": base_size,
        "axes.titlesize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,

        # Black text and rules throughout.
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,

        # Ticks inward, on all four sides, minor ticks on.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=list(NPG.values())),

        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,

        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.labelspacing": 0.35,
        "legend.borderpad": 0.2,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed as TrueType, not Type 3
        "ps.fonttype": 42,
    })
