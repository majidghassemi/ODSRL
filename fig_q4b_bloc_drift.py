"""
Fig. P -- q4b_bloc_drift (half column, 1.60 x 1.45 in). Pairs with panel O.

Reads bloc_partition_results.json (produced by gen_bloc_partition.py). Plots nothing
that is not in that file.

Story: collusion that forms MID-RUN is not covered by a partition computed once. The
drift alarm watches the leading eigenvalue of the residual report covariance -- residual
because the per-step consensus is subtracted first, which strips the common truth signal
and leaves only coordination. Before the bloc forms the statistic sits at ~1 (no shared
structure beyond noise); twelve of twenty evaluators start sharing a component at step
800 and it jumps past the 1.15 threshold within 25 steps, at which point every region is
re-flagged for audit.

THE STATISTIC, NOT THE ALARM: the underlying experiment records a 0/1 alarm, but a
binary spike train shows only that something fired, not that the margin is a factor of
ten. The alarm is the thresholded version of what is drawn here, and the generator
asserts at every checkpoint that the two agree.

WHY IT FALLS BACK TO 1: this is a CHANGE detector, not a state detector. It compares the
most recent 200 steps against the 200 before them, so once both windows sit inside the
colluding phase (past step ~1100) there is no jump left to see and the ratio returns to
baseline. The bloc has not gone away -- it has become the new normal, which is why the
alarm's job is to re-flag regions for audit rather than to score them continuously.
"""
import json

import numpy as np

import style
from style import CAPTURED, ESA, HALF_COL, LW_ESA, REF_LINE
import matplotlib.pyplot as plt

style.apply()

d = json.load(open("bloc_partition_results.json"))["drift"]
steps = np.array(d["steps"])
ratio = np.array(d["ratio"])

fig, ax = plt.subplots(figsize=HALF_COL)

# When the bloc forms, and the threshold it has to clear. Both are references.
ax.axvline(d["bloc_forms"], color=CAPTURED, linewidth=0.8, linestyle=(0, (2, 2)),
           zorder=1)
ax.axhline(d["thresh"], color=REF_LINE, linewidth=0.6, linestyle=(0, (1.4, 1.4)),
           zorder=1)

ax.plot(steps, ratio, color=ESA, linewidth=LW_ESA, zorder=3)

ax.set_xlabel("Step")
ax.set_ylabel("Eigenvalue ratio")
ax.set_xlim(steps[0], steps[-1])
ax.set_xticks([500, 1000, 1500])
ax.set_ylim(0, 12)
ax.set_yticks([1, 5, 10])
style.grid(ax)

# Two words of in-panel labeling, because a bare vertical line is unreadable without
# them and the alternative is a legend that costs more space than the labels do.
ax.text(d["bloc_forms"] - 25, 11.4, "bloc forms", ha="right", va="top", fontsize=7,
        color=CAPTURED)
ax.text(steps[-1], d["thresh"] + 0.35, "alarm", ha="right", va="bottom", fontsize=7,
        color="#6F6F6F")

style.save(fig, "new results/q4b_bloc_drift.pdf", HALF_COL)
