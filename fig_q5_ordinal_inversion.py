"""
Fig. F -- q5_ordinal_inversion (wide panel, 2.20 x 1.80 in).

Reads bootstrap_ci.json -> "ordinal". Plots nothing that is not in that file.

Story: ESA's detection signal lives in the CARDINAL channel. Eliciting absolute scores
exposes the sycophant's miscalibration and ESA recovers 80%. Eliciting the SAME panel by
pairwise PREFERENCE -- the modality RLHF/DPO use -- hides it: the sycophant is the better
ranker, trust inverts onto it, and recovery falls to 0.

Both bars are the same method under different elicitation, so both use the ESA color;
the ordinal variant is hollow and hatched (the bar-chart analogue of the dashed
same-method-different-variant line used in the curve panels).

Shares y-limits and ticks with panels D and E via style.recovery_panel.
"""
import json

import style
from style import ESA, WIDE_PANEL
import matplotlib.pyplot as plt

style.apply()

ordinal = json.load(open("bootstrap_ci.json"))["ordinal"]

values = [ordinal["cardinal"]["recovery"], ordinal["ordinal"]["recovery"]]
cis = [tuple(ordinal["cardinal"]["ci95"]), tuple(ordinal["ordinal"]["ci95"])]

fig, ax = plt.subplots(figsize=WIDE_PANEL)
# Error bars are the stored 95% bootstrap CIs (5000 resamples over 100 items).
style.recovery_panel(ax, ["Cardinal", "Ordinal"], values, cis, [ESA, ESA],
                     hatches=[None, "////"], ylabel="Recovery on TruthfulQA")
style.save(fig, "new results/q5_ordinal_inversion.pdf", WIDE_PANEL)
