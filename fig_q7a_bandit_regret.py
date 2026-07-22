"""
Fig. I -- q7a_bandit_regret (half column, 1.60 x 1.45 in). Pairs with panel J.

Reads bandit_regret_curves.npz (produced by gen_bandit_regret.py). Plots nothing that
is not in that file.

Story: 80% of the evaluators are biased and the divergence arm IS covered. That is the
regime the coverage-boundary theorem admits: an honest minority plus coverage, so
per-region active-audit ESA concentrates trust on the honest sources and its cumulative
latent regret flattens. Every majority-trusting aggregator tracks the corrupted majority
and pays LINEAR regret.

Mean and Dawid-Skene end at 2433 and 2472 out of ~2472 -- 1.6% apart, which is under a
point on a 1.45 in panel. They are drawn in the same "captured by the majority" color
with one legend entry rather than pretending two distinguishable lines exist.

Shares y-limits and ticks with panel J via style.REGRET_YLIM.
"""
import numpy as np

import style
from style import CAPTURED, ESA, HALF_COL, LW_BASE, LW_ESA
import matplotlib.pyplot as plt

style.apply()

SUB = 20        # subsample for a compact vector file; curves are smooth at this rate

d = np.load("bandit_regret_curves.npz")
x = np.arange(d["a_esa"].shape[1])[::SUB]

fig, ax = plt.subplots(figsize=HALF_COL)

# Bands are +/-1 SD across the 25 seeds.
series = [
    # (key, label, color, linestyle, linewidth, zorder)
    ("a_dawid", "Mean, Dawid-Skene", CAPTURED, "-", LW_BASE, 2),
    ("a_mean",  None,                CAPTURED, "-", LW_BASE, 2),
    ("a_esa",   "ESA active-audit",  ESA,      "-", LW_ESA,  4),
]

for key, label, color, ls, lw, z in series:
    m, sd = d[key].mean(0)[::SUB], d[key].std(0)[::SUB]
    style.band(ax, x, m, m - sd, m + sd, color)
    ax.plot(x, m, color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

ax.set_xlabel(r"Steps ($\times 10^3$)")
ax.set_ylabel("Cumulative latent regret")
ax.set_xlim(0, d["a_esa"].shape[1])
ax.set_xticks([0, 2500, 5000])
ax.set_xticklabels(["0", "2.5", "5"])
ax.set_ylim(*style.REGRET_YLIM)
ax.set_yticks(style.REGRET_YTICKS)
ax.set_yticklabels(style.REGRET_YTICKLABELS)
style.grid(ax)

# Least data-dense corner: upper left (both regimes rise from the origin).
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04))

style.save(fig, "new results/q7a_bandit_regret.pdf", HALF_COL)
