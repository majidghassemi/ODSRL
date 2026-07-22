"""
Shared figure style for the AAAI two-column submission.

Every plotting script imports this and calls nothing else style-related: no per-figure
rcParam overrides, so all panels are typographically identical. Figures are authored at
FINAL PHYSICAL SIZE and must not be scaled in LaTeX (use \\includegraphics with no
width= argument, or width equal to the native size).

Font: Computer Modern, matching the paper's article-class default (the .tex loads no
font package). Sizes are final: 8pt axis labels, 7pt ticks, 7pt legend, no in-plot titles.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- slot sizes (inches), authored at final print size -----------------------
FULL_COL = (3.05, 2.10)     # A: full-column figure
HALF_COL = (1.60, 1.45)     # B, C, G, H: half-column subfigure panels
WIDE_PANEL = (2.20, 1.80)   # D, E, F: wide-figure panels

# --- semantic palette (Okabe-Ito derived, colorblind-safe) -------------------
# ESA (per-region) is ALWAYS this color, always solid, always drawn last (on top).
ESA = "#0072B2"
# Global-trust ESA: same color, dashed -> reads as same method, different variant.
ESA_GLOBAL = ESA
# "Captured by the majority" (naive mean / sycophantic majority): recurring warm role.
CAPTURED = "#D55E00"
# Muted baselines.
GRAY_DARK = "#5A5A5A"
GRAY_MID = "#8C8C8C"
GRAY_LIGHT = "#B4B4B4"
TEAL = "#009E73"        # desaturated Okabe-Ito, for a named baseline (e.g. KL-DRO)
REF_LINE = "#9A9A9A"    # vertical/horizontal reference lines

LW_ESA = 1.6
LW_BASE = 1.1
BAND_ALPHA = 0.18


def ramp(n):
    """n colors for an ORDERED family of the same method (a hyperparameter sweep).

    The semantic palette above is categorical -- it names roles, not magnitudes -- so a
    sweep over one knob (bias ratio, eta, reference noise) needs a ramp instead. This
    one runs ESA blue -> neutral -> CAPTURED orange: the two ends are the palette's
    existing "ESA works" and "captured by the majority" roles, so a sweep that ends in
    capture reads with the rest of the paper, and blue-orange is the colorblind-safe
    axis. Lightness varies along the ramp too, so it survives grayscale printing.
    """
    stops = [ESA, "#6F6F6F", CAPTURED]

    def rgb(h):
        return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))

    pts = [rgb(s) for s in stops]
    if n == 1:
        return [stops[0]]
    out = []
    for i in range(n):
        u = i / (n - 1) * (len(pts) - 1)
        j = min(int(u), len(pts) - 2)
        f = u - j
        c = tuple(pts[j][ch] + f * (pts[j + 1][ch] - pts[j][ch]) for ch in range(3))
        out.append("#%02X%02X%02X" % tuple(round(v * 255) for v in c))
    return out

# Dash patterns: mean and median must be distinguishable at 7pt.
DASH_MEAN = (0, (4, 1.6))
DASH_MEDIAN = (0, (1.4, 1.4))
DASH_GLOBAL = (0, (5, 1.5))
DASH_OTHER = (0, (3, 1, 1, 1))


def apply():
    """Install the shared rcParams. Call once at import time in each script."""
    plt.rcParams.update({
        # Computer Modern to match the article-class body font. cmr10 is the genuine
        # CM Roman face bundled with matplotlib; CMU Serif is not installed here.
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        # cmr10 has no U+2212; without this, negative ticks render as missing glyphs.
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,

        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        "axes.linewidth": 0.6,
        "axes.spines.top": False,      # despine top and right everywhere
        "axes.spines.right": False,
        "axes.labelpad": 2.0,

        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,

        "legend.frameon": False,       # frame off, placed inside the axes
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.28,
        "legend.borderaxespad": 0.2,
        "legend.borderpad": 0.0,

        "lines.solid_capstyle": "round",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,            # embed as Type-42 so text stays selectable
        "ps.fonttype": 42,
    })


def grid(ax):
    """Subtle y-grid only. No x-grid."""
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)


def band(ax, x, mean, lo, hi, color):
    """Uncertainty band: no outline, alpha 0.18, same color as its line."""
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, linewidth=0)


def boot_ci(M, sub=1, n_boot=2000, seed=0):
    """Across-seed mean of a [seeds, T] matrix with a 95% bootstrap CI OF THE MEAN.

    Lives here because the three ablation panels must not diverge in how their band is
    computed. Bootstrap rather than +/-1 SD for the same reason Fig. B uses one: where a
    setting is bimodal across seeds (ESA either escapes or stays trapped), the seed SD
    describes the spread of two outcomes, not the uncertainty in the curve being drawn,
    and an SD band would floor at a negative regret.
    """
    import numpy as np

    C = np.asarray(M)[:, ::sub]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, C.shape[0], size=(n_boot, C.shape[0]))
    lo, hi = np.percentile(C[idx].mean(axis=1), [2.5, 97.5], axis=0)
    return np.arange(C.shape[1]) * sub, C.mean(0), lo, hi


# --- shared geometry for the recovery panels D, E, F ------------------------
# These three panels MUST share y-limits and tick positions exactly, so the numbers
# live here rather than in the individual scripts.
RECOVERY_YLIM = (0.0, 1.0)
RECOVERY_YTICKS = [0.0, 0.25, 0.50, 0.75, 1.00]
RECOVERY_YTICKLABELS = ["0", ".25", ".50", ".75", "1"]
BAR_WIDTH = 0.55

# --- shared geometry for the bandit regret panels I, J ----------------------
# Both panels are the same environment at a different sycophant ratio, so the linear-
# regret baselines must land at the same height in each or the comparison is lost.
REGRET_YLIM = (0, 2650)
REGRET_YTICKS = [0, 1000, 2000]
REGRET_YTICKLABELS = ["0", "1k", "2k"]

# --- shared geometry for the ablation panels K, L ---------------------------
# K and L are both ESA in the regime where it works, so they share a scale. M (reference
# noise) is the panel where ESA starts to fail and needs five times the range; it sets
# its own limits and says so in its docstring rather than flattening K and L to fit.
ABLATION_YLIM = (0, 230)
ABLATION_YTICKS = [0, 100, 200]


def recovery_panel(ax, labels, values, cis, colors, hatches=None, ylabel="Recovery"):
    """Draw one of the D/E/F recovery panels.

    values : point estimates; cis : list of (lo, hi) 95% bootstrap intervals.
    Error bars are asymmetric, drawn from the stored CI bounds -- never mean +/- k*se.
    """
    x = range(len(values))
    hatches = hatches or [None] * len(values)
    lo = [v - c[0] for v, c in zip(values, cis)]
    hi = [c[1] - v for v, c in zip(values, cis)]

    for i, (v, col, h) in enumerate(zip(values, colors, hatches)):
        if h:   # variant of the same method: same color, hollow + hatched
            ax.bar(i, v, width=BAR_WIDTH, facecolor="none", edgecolor=col,
                   linewidth=0.9, hatch=h, zorder=2)
        else:
            ax.bar(i, v, width=BAR_WIDTH, color=col, linewidth=0, zorder=2)

    ax.errorbar(list(x), values, yerr=[lo, hi], fmt="none", ecolor="#333333",
                elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=3)

    # A bar of exactly zero draws nothing, which reads as "series missing" rather than
    # "value is zero". Label the true zeros so the collapse is unambiguous.
    for i, v in enumerate(values):
        if v == 0:
            ax.text(i, 0.022, "0", ha="center", va="bottom", fontsize=7,
                    color="#333333", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, len(values) - 0.35)
    ax.set_ylim(*RECOVERY_YLIM)
    ax.set_yticks(RECOVERY_YTICKS)
    ax.set_yticklabels(RECOVERY_YTICKLABELS)
    ax.set_ylabel(ylabel)
    grid(ax)


PAD_INCHES = 0.02


def save(fig, path, size, iters=6, tol=1e-3):
    """Vector PDF at EXACTLY `size` inches, saved with a tight bounding box.

    bbox_inches='tight' crops the page to the drawn ink, so the saved page is normally
    smaller (or, with a long y-label, wider) than the figsize -- which would break both
    the 'design at final physical size, no downstream scaling' rule and the panel-height
    coordination between paired figures.

    Fix: grow/shrink the canvas until the tight-cropped page equals the target. Each
    iteration measures the tight bbox and corrects the figure size by the shortfall;
    font sizes are absolute, so only the axes area absorbs the change and this converges
    in two or three passes. The result is a genuinely tight page of exactly `size`.

    Either element of `size` may be None, meaning "match the other and let this one fall
    where the content puts it". That is not a loophole for missing a slot -- it is for
    axes with a locked aspect ratio, where fixing the width fixes the height too and
    demanding both would just pad the page with whitespace and break the tight crop.
    """
    import io
    import os
    import re

    target_w, target_h = size

    def rendered_size():
        """Actual page size of the tight-cropped PDF, in inches.

        savefig's tight crop does not exactly equal fig.get_tightbbox(), so measure the
        real output rather than predicting it.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=PAD_INCHES)
        m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]",
                      buf.getvalue())
        x0, y0, x1, y1 = (float(v) for v in m.groups())
        return (x1 - x0) / 72.0, (y1 - y0) / 72.0

    for _ in range(iters):
        got_w, got_h = rendered_size()
        dw = 0.0 if target_w is None else target_w - got_w
        dh = 0.0 if target_h is None else target_h - got_h
        if abs(dw) < tol and abs(dh) < tol:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + dw, h + dh)

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=PAD_INCHES)
    got_w, got_h = rendered_size()
    print(f"saved {path}  ({got_w:.3f} x {got_h:.3f} in)")
