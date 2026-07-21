"""
Fig. G -- q3_audit_cost (half column, 1.60 x 1.45 in). Promoted from the appendix.

Reads audit_cost_results.json (produced by gen_audit_cost.py). Plots nothing that is
not in that file.

Story: the fixed-rate auditor pays a reference query for a constant FRACTION of steps,
so its cost is linear in the horizon and reaches ~8000 queries by T=80k. Active auditing
stops querying once a region resolves, so its budget is horizon-INDEPENDENT: flat at 32
across three decades of T, while still recovering the optimum in 100% of seeds.

Both axes are log: the two curves differ by ~2.4 decades at the right edge and a linear
y-axis would flatten the active line onto the origin. Pairs with panel H at equal height.
"""
import json

import numpy as np

import style
from style import ESA, GRAY_DARK, LW_ESA, LW_BASE, HALF_COL, DASH_MEAN
import matplotlib.pyplot as plt

style.apply()

d = json.load(open("audit_cost_results.json"))
T = np.array(d["horizons"], dtype=float)

fig, ax = plt.subplots(figsize=HALF_COL)

# Bands are +/-1 SD over the 10 seeds.
for key, label, color, ls, lw, z in [
        ("fixed", r"Fixed-rate", GRAY_DARK, DASH_MEAN, LW_BASE, 2),
        ("active", "Active", ESA, "-", LW_ESA, 3)]:
    m = np.array(d[key]["mean"])
    sd = np.array(d[key]["std"])
    style.band(ax, T, m, np.maximum(m - sd, 1e-9), m + sd, color)
    ax.plot(T, m, color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Horizon $T$ (steps)")
ax.set_ylabel("Reference queries")
ax.set_xlim(80, 1.1e5)
ax.set_ylim(5, 2e4)
ax.set_xticks([1e2, 1e3, 1e4, 1e5])
ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])
ax.set_yticks([1e1, 1e2, 1e3, 1e4])
ax.set_yticklabels([r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"])
style.grid(ax)

# Annotate the flat line with the asymptote the text quotes.
asym = d["active_asymptote"]
ax.annotate(f"{asym:.0f}", xy=(T[-1], asym), xytext=(-1, 4),
            textcoords="offset points", ha="right", va="bottom",
            fontsize=7, color=ESA)

ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04))

style.save(fig, "new results/q3_audit_cost.pdf", HALF_COL)
