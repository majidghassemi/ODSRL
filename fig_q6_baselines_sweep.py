"""
Fig. N -- q6_baselines_sweep (half column, 1.60 x 1.45 in). Simulated twin of panel H.

Reads baselines_sweep_results.json (produced by gen_baselines_sweep.py). Plots nothing
that is not in that file.

Story: the same sweep as panel H -- recovery as the biased fraction of the panel grows --
but on the bandit, where the y-axis can be the quantity the theory actually bounds:
cumulative latent regret at T=5000. Every majority-trusting aggregator goes linear at or
before the theoretical tipping point ratio* = arm_gap / bias_mag = 0.17, the ratio at
which the mean bias overtakes the latent gap. ESA stays at ~250 until the bias is
100% homogeneous, where no honest minority exists and (with no fail-safe, as configured
here) it collapses with the rest -- the same boundary panel J measures in time.

The six collapsing baselines get one color and one legend entry, as in panel H: their
individual collapse ratios differ (Mean at 0.2, Wass-DRO by 0.3, the rest by 0.4) and
that staircase is visible without six legend rows to read it. KL-DRO is drawn distinctly
for the same reason as in panel H -- its partial robustness is discussed in the text.
"""
import json

import numpy as np

import style
from style import CAPTURED, DASH_OTHER, ESA, HALF_COL, LW_BASE, LW_ESA, REF_LINE, TEAL
import matplotlib.pyplot as plt

style.apply()

d = json.load(open("baselines_sweep_results.json"))
ratios = np.array(d["ratios"], dtype=float)
GRAY_BASELINES = ["Mean", "Median", "Dawid-Skene", "GAIL", "Wass-DRO", "RRM"]

fig, ax = plt.subplots(figsize=HALF_COL)

# Theoretical tipping point. Minimally labeled -- the caption carries it.
ax.axvline(d["tip"], color=REF_LINE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)

for i, name in enumerate(GRAY_BASELINES):
    ax.plot(ratios, d["mean"][name], color=CAPTURED, linewidth=LW_BASE,
            label="Majority-trusting" if i == 0 else None, zorder=2)

ax.plot(ratios, d["mean"]["KL-DRO"], color=TEAL, linewidth=LW_BASE,
        linestyle=DASH_OTHER, label="KL-DRO", zorder=3)
ax.plot(ratios, d["mean"]["ESA"], color=ESA, linewidth=LW_ESA, label="ESA", zorder=4)

ax.set_xlabel("Sycophant fraction")
ax.set_ylabel("Latent regret at $T$")
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(["0", ".50", "1"])
ax.set_ylim(*style.REGRET_YLIM)
ax.set_yticks(style.REGRET_YTICKS)
ax.set_yticklabels(style.REGRET_YTICKLABELS)
style.grid(ax)

# Least data-dense region: mid-height on the right. Past 0.55 every baseline has already
# collapsed to the top of the panel and ESA is still flat along the bottom, so the middle
# of that half is empty -- the risers all happen to the left of it.
ax.legend(loc="center right", bbox_to_anchor=(1.04, 0.44), handlelength=1.2,
          labelspacing=0.2)

style.save(fig, "new results/q6_baselines_sweep.pdf", HALF_COL)
