"""
Fig. A -- q2_coverage_boundary (full column, 3.05 x 2.10 in).

Reads coverage_boundary_curves.npz (produced by gen_coverage_boundary.py). Plots
nothing that is not in that file.

Story: an on/off transition at the coverage edge. With the divergence arm covered,
ESA's cumulative latent regret saturates (it identifies the biased sources and stops
paying). With it uncovered, ESA is provably no better than naive mean aggregation --
and the two curves coincide numerically, so the mean is drawn thicker underneath and
uncovered-ESA dashed on top to keep both legible.
"""
import numpy as np

import style
from style import ESA, CAPTURED, LW_ESA, LW_BASE, FULL_COL
import matplotlib.pyplot as plt

style.apply()

SUB = 10        # subsample for a compact vector file; curves are smooth at this rate

d = np.load("coverage_boundary_curves.npz")
x = np.arange(d["esa_covered"].shape[1])[::SUB]

fig, ax = plt.subplots(figsize=FULL_COL)

# Bands are +/- 1 SD across the 20 seeds.
series = [
    # (key, label, color, linestyle, linewidth, zorder)
    ("mean",          "Naive mean",          CAPTURED, "-",            1.5,    2),
    ("esa_uncovered", "ESA, uncovered",      ESA,      (0, (4, 1.6)),  LW_BASE, 3),
    ("esa_covered",   "ESA, covered",        ESA,      "-",            LW_ESA,  4),
]

for key, label, color, ls, lw, z in series:
    m = d[key].mean(0)[::SUB]
    sd = d[key].std(0)[::SUB]
    style.band(ax, x, m, m - sd, m + sd, color)
    ax.plot(x, m, color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

ax.set_xlabel(r"Steps ($\times 10^3$)")
ax.set_ylabel("Cumulative latent regret")
ax.set_xlim(0, d["esa_covered"].shape[1])
ax.set_ylim(0, None)
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
style.grid(ax)

# Least data-dense corner: upper left (all curves rise from the origin).
ax.legend(loc="upper left", handlelength=1.6)

style.save(fig, "new results/q2_coverage_boundary.pdf", FULL_COL)
