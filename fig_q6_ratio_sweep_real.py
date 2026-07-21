"""
Fig. H -- q6_ratio_sweep_real (half column, 1.60 x 1.45 in). Pairs with panel G.

Reads q6_ratio_sweep_results.json. Plots nothing that is not in that file.

Story: recovery as the sycophant fraction of the grader panel is swept. Every
majority-trusting aggregator survives while sycophants are a minority and collapses past
the 0.5 tipping point; ESA, auditing the source on the verifiable region, is flat at 0.80
all the way to a 90% sycophantic panel. KL-DRO is drawn distinctly because its partial
0.38 plateau is discussed in the text -- it exploits the same cardinal channel ESA uses,
without a reference.

The frac=1.0 entry is NaN for every method (an all-sycophant panel has no honest grader
to recover, so recovery is undefined) and is dropped rather than plotted as zero.

NOTE: Dawid-Skene coincides with ESA at 0.80 until frac=0.8 before collapsing to 0.13.
ESA is drawn last and on top, so the gray Dawid-Skene curve is visible only where it
diverges -- which is the only place it carries information.
"""
import json

import numpy as np

import style
from style import (ESA, TEAL, GRAY_MID, REF_LINE, LW_ESA, LW_BASE, HALF_COL,
                   DASH_OTHER)
import matplotlib.pyplot as plt

style.apply()

d = json.load(open("q6_ratio_sweep_results.json"))
fracs = np.array(d["fracs"], dtype=float)
curves = {k: np.array(v, dtype=float) for k, v in d["curves"].items()}

# Drop horizons where every method is NaN (frac = 1.0: no honest grader exists).
keep = ~np.all(np.isnan(np.stack(list(curves.values()))), axis=0)
fracs = fracs[keep]
curves = {k: v[keep] for k, v in curves.items()}

GRAY_BASELINES = ["Mean", "Median", "Dawid-Skene", "Wass-DRO", "RRM"]

fig, ax = plt.subplots(figsize=HALF_COL)

# Tipping point: half the panel is sycophantic. Minimally labeled -- the caption carries it.
ax.axvline(0.5, color=REF_LINE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)

# Collapsing baselines: one legend entry for all five, colors do the rest.
for i, name in enumerate(GRAY_BASELINES):
    ax.plot(fracs, curves[name], color=GRAY_MID, linewidth=LW_BASE,
            label="Other baselines" if i == 0 else None, zorder=2)

ax.plot(fracs, curves["KL-DRO"], color=TEAL, linewidth=LW_BASE,
        linestyle=DASH_OTHER, label="KL-DRO", zorder=3)
ax.plot(fracs, curves["ESA"], color=ESA, linewidth=LW_ESA, label="ESA", zorder=4)

ax.set_xlabel("Sycophant fraction")
ax.set_ylabel("Recovery")
ax.set_xlim(0, 0.9)
ax.set_ylim(*style.RECOVERY_YLIM)
ax.set_xticks([0, 0.25, 0.5, 0.75])
ax.set_xticklabels(["0", ".25", ".50", ".75"])
ax.set_yticks(style.RECOVERY_YTICKS)
ax.set_yticklabels(style.RECOVERY_YTICKLABELS)
style.grid(ax)

# Least data-dense corner: lower left (all methods start high and fall only past 0.5).
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.03))

style.save(fig, "new results/q6_ratio_sweep_real.pdf", HALF_COL)
