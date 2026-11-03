"""Shared print style. No data transformations or statistical settings belong here."""
from pathlib import Path
import matplotlib.pyplot as plt

SINGLE_COL = (3.45, 2.75)
DOUBLE_COL = (6.1, 2.9)
SUPPLEMENT_WIDE = (4.4, 2.8)
METHOD_COLORS = {
    "independent": "#666666", "global": "#D55E00",
    "similarity": "#0072B2", "rgpe": "#009E73",
}
METHOD_STYLES = {
    "independent": ("s", "--"), "global": ("o", "-"),
    "similarity": ("D", "-."), "rgpe": ("^", ":"),
}


def apply_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.titlesize": 9,
        "axes.linewidth": 0.6, "axes.edgecolor": "0.3",
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "0.85", "grid.alpha": 0.6, "grid.linewidth": 0.45,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "figure.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def outside_legend(ax, *, ncol=1):
    """Reserve a separate band above the axes, never cover observations."""
    return ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02),
                     ncol=ncol, borderaxespad=0, handlelength=2.5,
                     handletextpad=0.6, labelspacing=0.3, frameon=False)


def save_figure(fig, directory: Path, stem: str):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = directory / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04,
                    transparent=False, dpi=600 if ext == "png" else None)
        print(f"wrote {path}")
