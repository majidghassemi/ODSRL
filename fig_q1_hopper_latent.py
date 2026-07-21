"""
Fig. C -- q1_hopper_latent (half column, 1.60 x 1.45 in).

Reads hopper_latent_curves.npz (produced by gen_hopper_curves.py). Plots nothing that
is not in that file.

Story: under an 80% velocity-penalizing majority, both majority-following learners
plateau just under 1.0 latent return, while per-region ESA audits the source and keeps
climbing past 2.3. The y-axis spans both regimes so the separation is the whole panel.

Last-1k latent return, 5 seeds: standard PPO 0.968 +/- 0.005, GAIL-style 0.965 +/- 0.009,
per-region ESA 2.365 +/- 0.137.
"""
import numpy as np

import style
from style import ESA, CAPTURED, GRAY_MID, LW_ESA, LW_BASE, HALF_COL, DASH_MEAN, DASH_OTHER
import matplotlib.pyplot as plt

style.apply()

WIN = 2000      # rolling window (env steps) -- per-step latent reward is very noisy
SUB = 200       # subsample after smoothing, for a compact vector file

d = np.load("hopper_latent_curves.npz")


def smoothed(M, win=WIN):
    """Per-seed rolling mean, then mean +/- 1 SD across the 5 seeds."""
    k = np.ones(win) / win
    C = np.stack([np.convolve(r, k, mode="valid") for r in M])[:, ::SUB]
    x = np.arange(C.shape[1]) * SUB
    return x, C.mean(0), C.std(0)


fig, ax = plt.subplots(figsize=HALF_COL)

# Baselines first, ESA last so it sits on top.
series = [
    ("gail",     "GAIL-style",   GRAY_MID, DASH_OTHER, LW_BASE, 2),
    ("standard", "Standard PPO", CAPTURED, DASH_MEAN,  LW_BASE, 3),
    ("esa",      "ESA region",   ESA,      "-",        LW_ESA,  4),
]

for key, label, color, ls, lw, z in series:
    x, m, sd = smoothed(d[key])
    style.band(ax, x, m, m - sd, m + sd, color)     # band = +/-1 SD over 5 seeds
    ax.plot(x, m, color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

ax.set_xlabel(r"Training steps ($\times 10^3$)")
ax.set_ylabel("Latent return")
ax.set_xlim(0, d["esa"].shape[1])
ax.set_ylim(0.5, 2.7)
ax.set_xticks([0, 25000, 50000])
ax.set_xticklabels(["0", "25", "50"])
ax.set_yticks([1.0, 1.5, 2.0, 2.5])
style.grid(ax)

# Least data-dense corner: upper left (ESA only reaches the top band late).
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04))

style.save(fig, "new results/q1_hopper_latent.pdf", HALF_COL)
