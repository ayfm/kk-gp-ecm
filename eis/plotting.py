"""Matplotlib settings of the figures: serif text and small fonts for a figure 3.5 in wide."""
import matplotlib.pyplot as plt

COL = 3.5    # figure width in inches


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        # serif text in a Times-like font and Computer Modern math through mathtext
        # (no LaTeX installation needed)
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman No9 L", "Nimbus Roman", "Times New Roman",
            "Liberation Serif", "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "lines.markersize": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # embed TrueType fonts in the PDF
        "ps.fonttype": 42,
    })
