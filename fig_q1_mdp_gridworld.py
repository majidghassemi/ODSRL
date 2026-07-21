"""
Fig. B -- q1_mdp_gridworld (half column, 1.60 x 1.45 in).

Reads gridworld_candy_curves.npz (produced by gen_gridworld.py). Plots nothing that is
not in that file.

Story: the two ESA variants are the foreground. Per-region trust localizes the audit
and escapes the proxy goal; global trust averages the rare divergence away and stays
trapped with the majority. Mean and Median are background gray -- both pinned at 1.00.

NOTE ON THE GLOBAL-ESA BAND: global-trust ESA is bimodal across seeds (trapped in 76 of
100, escapes in the rest), so its seed-spread SD is ~0.43 by construction. The band here
is therefore a 95% bootstrap CI of the mean, not an SD band -- see smoothed().
"""
import numpy as np

import style
from style import (ESA, CAPTURED, GRAY_MID, LW_ESA, LW_BASE, HALF_COL,
                   DASH_MEAN, DASH_MEDIAN, DASH_GLOBAL)
import matplotlib.pyplot as plt

style.apply()

WIN = 500       # smoothing window (episodes); raw traces are 0/1 per episode
SUB = 100       # subsample after smoothing, for a compact vector file

d = np.load("gridworld_candy_curves.npz")


def smoothed(M, win=WIN, n_boot=2000, seed=0):
    """Per-seed moving average, then the across-seed mean with a 95% bootstrap CI.

    The band is a 95% bootstrap CI OF THE MEAN (resampling the 100 seeds), not +/-1 SD
    of the seed spread. Global-trust ESA is bimodal across seeds (trapped in 76 of 100),
    so its SD is ~0.43 and an SD band would flood the panel while describing seed spread
    rather than uncertainty in the plotted curve.
    """
    k = np.ones(win) / win
    C = np.stack([np.convolve(r, k, mode="valid") for r in M])[:, ::SUB]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, C.shape[0], size=(n_boot, C.shape[0]))
    boots = C[idx].mean(axis=1)                       # [n_boot, T]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    x = np.arange(C.shape[1]) * SUB
    return x, C.mean(0), lo, hi


fig, ax = plt.subplots(figsize=HALF_COL)

# Baselines first (background), ESA variants last so they sit on top.
# Mean and Median are both pinned at exactly 1.00 and would hide one another, so the
# gray Median is drawn solid and wider underneath and shows through the Mean's dashes.
series = [
    ("median",     "Median",     GRAY_MID, "-",         1.5,     2),
    ("mean",       "Mean",       CAPTURED, DASH_MEAN,   LW_BASE, 3),
    ("global_esa", "ESA global", ESA,      DASH_GLOBAL, LW_BASE, 4),
    ("region_esa", "ESA region", ESA,      "-",         LW_ESA,  5),
]

for key, label, color, ls, lw, z in series:
    x, m, lo, hi = smoothed(d[key])
    style.band(ax, x, m, lo, hi, color)     # band = 95% bootstrap CI of the mean
    ax.plot(x, m, color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

ax.set_xlabel(r"Episodes ($\times 10^3$)")
ax.set_ylabel("P(visit proxy goal)")
ax.set_xlim(0, d["mean"].shape[1])
ax.set_ylim(-0.05, 1.08)
ax.set_xticks([0, 25000, 50000])
ax.set_xticklabels(["0", "25", "50"])
ax.set_yticks([0, 0.5, 1.0])
style.grid(ax)

# Least data-dense region: the strip between the escaped ESA curve (0.0) and the
# global-ESA CI (~0.43), on the right where all curves have flattened.
ax.legend(loc="center left", bbox_to_anchor=(0.28, 0.26))

style.save(fig, "new results/q1_mdp_gridworld.pdf", HALF_COL)
