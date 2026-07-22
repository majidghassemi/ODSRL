"""
Fig. J -- q7b_bandit_regret_failsafe (half column, 1.60 x 1.45 in). Pairs with panel I.

Reads bandit_regret_curves.npz (produced by gen_bandit_regret.py). Plots nothing that
is not in that file.

Story: same environment as panel I but the bias is 100% HOMOGENEOUS -- there is no
honest minority left to concentrate trust on, so pure trust auditing provably cannot
recover (internal-only impossibility) and ESA lands exactly on the majority-trusting
baselines. What saves it is the ABSOLUTE-DISTRUST fail-safe: tau detects that every
source in the region is bad and falls back to the reference, which is only available
because the arm is covered. Panels I and J are the two halves of the same claim --
coverage is necessary, an honest minority is not sufficient on its own.

THREE COINCIDENT CURVES: Mean, Dawid-Skene and fail-safe-less ESA all end at 2472 +/- 1,
i.e. identical to within the seed noise. Drawing three dash patterns on top of one
another would invent a distinction the data does not have, so the two baselines share
the "captured" color and one legend entry, and fail-safe-less ESA is drawn over them
dashed in the ESA color -- same-method-different-variant, as everywhere else in the set.
Its being invisible except where it separates IS the result.

Shares y-limits and ticks with panel I via style.REGRET_YLIM.
"""
import numpy as np

import style
from style import CAPTURED, DASH_GLOBAL, ESA, HALF_COL, LW_BASE, LW_ESA
import matplotlib.pyplot as plt

style.apply()

SUB = 20        # subsample for a compact vector file; curves are smooth at this rate

d = np.load("bandit_regret_curves.npz")
x = np.arange(d["b_esa"].shape[1])[::SUB]

# The merged legend entry claims the two baselines are indistinguishable here; check it.
_f = {k: d[f"b_{k}"].mean(0)[-1] for k in ("mean", "dawid", "esa")}
assert abs(_f["mean"] - _f["dawid"]) < 0.01 * _f["mean"], _f
assert abs(_f["esa"] - _f["dawid"]) < 0.01 * _f["dawid"], _f

fig, ax = plt.subplots(figsize=HALF_COL)

# Bands are +/-1 SD across the 25 seeds.
series = [
    # (key, label, color, linestyle, linewidth, zorder)
    ("b_dawid", "Mean, Dawid-Skene", CAPTURED, "-",         LW_BASE, 2),
    ("b_mean",  None,                CAPTURED, "-",         LW_BASE, 2),
    ("b_esa",   "ESA, no fail-safe", ESA,      DASH_GLOBAL, LW_BASE, 3),
    ("b_esa_fs", "ESA + fail-safe",  ESA,      "-",         LW_ESA,  4),
]

for key, label, color, ls, lw, z in series:
    m, sd = d[key].mean(0)[::SUB], d[key].std(0)[::SUB]
    style.band(ax, x, m, m - sd, m + sd, color)
    ax.plot(x, m, color=color, linestyle=ls, linewidth=lw, label=label, zorder=z)

ax.set_xlabel(r"Steps ($\times 10^3$)")
ax.set_ylabel("Cumulative latent regret")
ax.set_xlim(0, d["b_esa"].shape[1])
ax.set_xticks([0, 2500, 5000])
ax.set_xticklabels(["0", "2.5", "5"])
ax.set_ylim(*style.REGRET_YLIM)
ax.set_yticks(style.REGRET_YTICKS)
ax.set_yticklabels(style.REGRET_YTICKLABELS)
style.grid(ax)

# Least data-dense corner: upper left. The coincident failing curves run diagonally to
# the top right and the fail-safe curve hugs the bottom, so the wedge above the diagonal
# is empty -- and this matches panel I, which places its legend identically.
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.04))

style.save(fig, "new results/q7b_bandit_regret_failsafe.pdf", HALF_COL)
