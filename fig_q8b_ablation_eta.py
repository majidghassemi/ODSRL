"""
Fig. L -- q8b_ablation_eta (half column, 1.60 x 1.45 in). One of the K/L/M trio.

Reads bandit_ablation_curves.npz (produced by gen_bandit_ablations.py). Plots nothing
that is not in that file.

Story: the trust learning rate is not a tuned knob. Across a 20x range of eta the final
regret moves from 193 to 111 -- monotone, and every setting stays in the bounded regime,
so nothing here is a hyperparameter-search artifact. Larger eta simply concentrates
trust on the honest minority sooner, which shows up as an earlier knee, not a different
outcome. Fixed at 80% biased with no fail-safe, so this is trust concentration alone.

Ordered sweep of one knob, so style.ramp again: ESA blue at the slowest setting through
to orange at the fastest. Unlike panels K and M the ramp encodes speed, not severity --
the caption says so.

Bands are 95% bootstrap CIs of the mean over the 15 seeds (style.boot_ci).

Shares y-limits and ticks with panel K via style.ABLATION_YLIM.
"""
import json

import numpy as np

import style
from style import HALF_COL, LW_BASE
import matplotlib.pyplot as plt

style.apply()

SUB = 20        # subsample for a compact vector file; curves are smooth at this rate

d = np.load("bandit_ablation_curves.npz")
values = json.load(open("bandit_ablation_summary.json"))["values"]["b"]
colors = style.ramp(len(values))

fig, ax = plt.subplots(figsize=HALF_COL)

for i, (v, c) in enumerate(zip(values, colors)):
    x, m, lo, hi = style.boot_ci(d[f"b_{i}"], sub=SUB)
    style.band(ax, x, m, lo, hi, c)
    ax.plot(x, m, color=c, linewidth=LW_BASE, label=rf"$\eta={v:g}$", zorder=2 + i)

ax.set_xlabel(r"Steps ($\times 10^3$)")
ax.set_ylabel("Cumulative latent regret")
ax.set_xlim(0, d["b_0"].shape[1])
ax.set_xticks([0, 2500, 5000])
ax.set_xticklabels(["0", "2.5", "5"])
ax.set_ylim(*style.ABLATION_YLIM)
ax.set_yticks(style.ABLATION_YTICKS)
style.grid(ax)

# Least data-dense corner: upper left (all three curves saturate below 210).
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.06), handlelength=1.1,
          labelspacing=0.18)

style.save(fig, "new results/q8b_ablation_eta.pdf", HALF_COL)
