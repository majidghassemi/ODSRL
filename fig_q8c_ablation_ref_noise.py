"""
Fig. M -- q8c_ablation_ref_noise (half column, 1.60 x 1.45 in). One of the K/L/M trio.

Reads bandit_ablation_curves.npz (produced by gen_bandit_ablations.py). Plots nothing
that is not in that file.

Story: the reference is the one thing ESA cannot do without, and this panel measures how
good it has to be. A clean or lightly noisy reference (sigma_ref = 0, 0.1) is
indistinguishable. Past that, ESA does NOT degrade smoothly -- it fails on a growing
FRACTION of seeds: at sigma_ref=0.5 two of fifteen seeds track the majority to linear
regret, at 1.0 three of fifteen do, while the remaining seeds stay near 120. The mean
curve therefore sits between two outcomes that never occur, which is exactly what the
wide bootstrap band is reporting, and the diverging-seed counts are annotated on the
curves so the bimodality is not read as a smooth cost.

This is the panel that motivates using bootstrap CIs of the mean rather than +/-1 SD
across all three ablations: the seed SD here is 900 on a mean of 602, so an SD band
would floor below zero regret and swamp the panel.

SEPARATE Y-SCALE: K and L share style.ABLATION_YLIM; this panel needs five times the
range and sets its own, because compressing K and L to fit 1100 would flatten the
comparisons they exist to make.
"""
import json

import numpy as np

import style
from style import HALF_COL, LW_BASE
import matplotlib.pyplot as plt

style.apply()

SUB = 20            # subsample for a compact vector file
DIVERGED = 1000     # final regret above this = tracked the majority (bounded runs ~120,
                    # linear runs ~2400), so the threshold sits in an empty gap

d = np.load("bandit_ablation_curves.npz")
values = json.load(open("bandit_ablation_summary.json"))["values"]["c"]
colors = style.ramp(len(values))

fig, ax = plt.subplots(figsize=HALF_COL)

for i, (v, c) in enumerate(zip(values, colors)):
    R = d[f"c_{i}"]
    x, m, lo, hi = style.boot_ci(R, sub=SUB)
    style.band(ax, x, m, lo, hi, c)
    ax.plot(x, m, color=c, linewidth=LW_BASE, label=rf"${v:g}$", zorder=2 + i)

    # Annotate only the settings that actually break, with how many seeds broke.
    n_bad = int((R[:, -1] > DIVERGED).sum())
    if n_bad:
        ax.annotate(f"{n_bad}/{R.shape[0]}", xy=(x[-1], m[-1]), xytext=(-1, 2),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color=c, zorder=5)

ax.set_xlabel(r"Steps ($\times 10^3$)")
ax.set_ylabel("Cumulative latent regret")
ax.set_xlim(0, d["c_0"].shape[1])
ax.set_xticks([0, 2500, 5000])
ax.set_xticklabels(["0", "2.5", "5"])
ax.set_ylim(0, 1150)
ax.set_yticks([0, 500, 1000])
style.grid(ax)

# Least data-dense corner: upper left (even the failing means stay under 700).
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.06), ncol=2, columnspacing=0.7,
          handlelength=1.1, labelspacing=0.18, title=r"ref. $\sigma$", title_fontsize=7)

style.save(fig, "new results/q8c_ablation_ref_noise.pdf", HALF_COL)
