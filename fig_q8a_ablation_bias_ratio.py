"""
Fig. K -- q8a_ablation_bias_ratio (half column, 1.60 x 1.45 in). One of the K/L/M trio.

Reads bandit_ablation_curves.npz (produced by gen_bandit_ablations.py). Plots nothing
that is not in that file.

Story: ESA's regret is flat in the biased FRACTION from 50% to 95% -- the honest
minority does not have to be large, it only has to exist. At 100% it does not exist,
and what keeps regret bounded there is the absolute-distrust fail-safe (tau=0.5, on for
every curve in this panel), not trust concentration. Panel J isolates that mechanism by
switching the fail-safe off at the same 100%: regret goes linear.

The family is an ordered sweep of one knob, so it uses style.ramp rather than the
categorical palette: ESA blue at the benign end, "captured" orange at 100% biased.

Bands are 95% bootstrap CIs of the mean over the 15 seeds (style.boot_ci) -- the same
estimator in all three ablation panels, for the reason given there.

Shares y-limits and ticks with panel L via style.ABLATION_YLIM.
"""
import json

import numpy as np

import style
from style import HALF_COL, LW_BASE
import matplotlib.pyplot as plt

style.apply()

SUB = 20        # subsample for a compact vector file; curves are smooth at this rate

d = np.load("bandit_ablation_curves.npz")
values = json.load(open("bandit_ablation_summary.json"))["values"]["a"]
colors = style.ramp(len(values))

fig, ax = plt.subplots(figsize=HALF_COL)

for i, (v, c) in enumerate(zip(values, colors)):
    x, m, lo, hi = style.boot_ci(d[f"a_{i}"], sub=SUB)
    style.band(ax, x, m, lo, hi, c)
    ax.plot(x, m, color=c, linewidth=LW_BASE, label=f"{v:.0%}", zorder=2 + i)

ax.set_xlabel(r"Steps ($\times 10^3$)")
ax.set_ylabel("Cumulative latent regret")
ax.set_xlim(0, d["a_0"].shape[1])
ax.set_xticks([0, 2500, 5000])
ax.set_xticklabels(["0", "2.5", "5"])
ax.set_ylim(*style.ABLATION_YLIM)
ax.set_yticks(style.ABLATION_YTICKS)
style.grid(ax)

# Least data-dense corner: upper left. Every curve saturates well below 200, so the top
# third of the panel is empty; two columns keep the five entries off the curves.
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.06), ncol=2, columnspacing=0.7,
          handlelength=1.1, labelspacing=0.18)

style.save(fig, "new results/q8a_ablation_bias_ratio.pdf", HALF_COL)
