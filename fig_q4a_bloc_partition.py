"""
Fig. O -- q4a_bloc_partition (half column, 1.60 x 1.45 in). Pairs with panel P.

Reads bloc_partition_results.json (produced by gen_bloc_partition.py). Plots nothing
that is not in that file.

Story: covariance clustering recovers the evaluator PARTITION with no oracle and no
reference, and keeps doing so when the colluders are the majority -- 0.90 at 60%
colluding, 0.70 at 80%, all well above the 0.50 chance line for a two-way split. What it
cannot do is say WHICH bloc is honest: both clusters are internally coherent, and a
coherent colluding bloc looks exactly like a coherent honest one from the inside. That
is the internal-only impossibility in miniature, and it is why the audit is not optional
-- correlation routing narrows the question, the reference answers it.

NO ERROR BARS: the across-seed SD is exactly 0.000 at all four ratios (10 seeds), so
error bars would be zero-height ticks implying a precision claim the panel does not need
to make. The zero SD is stated in the caption instead.

The bars are ESA blue because this is a component of ESA, not a competing method.
"""
import json

import numpy as np

import style
from style import ESA, HALF_COL, REF_LINE
import matplotlib.pyplot as plt

style.apply()

d = json.load(open("bloc_partition_results.json"))
ratios = d["ratios"]
acc = d["partition"]["mean"]
assert max(d["partition"]["std"]) == 0.0, "SD is nonzero -- the panel must show it"

fig, ax = plt.subplots(figsize=HALF_COL)

# Chance for a two-way split. Drawn under the bars: it is a reference, not a series.
ax.axhline(0.5, color=REF_LINE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)

x = range(len(ratios))
ax.bar(list(x), acc, width=style.BAR_WIDTH, color=ESA, linewidth=0, zorder=2)

ax.set_xticks(list(x))
ax.set_xticklabels([f"{r:.0%}" for r in ratios])
ax.set_xlim(-0.65, len(ratios) - 0.35)
ax.set_xlabel("Colluding fraction")
ax.set_ylabel("Partition accuracy")
ax.set_ylim(0.0, 1.0)
ax.set_yticks(style.RECOVERY_YTICKS)
ax.set_yticklabels(style.RECOVERY_YTICKLABELS)
style.grid(ax)

style.save(fig, "new results/q4a_bloc_partition.pdf", HALF_COL)
