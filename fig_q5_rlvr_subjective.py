"""
Fig. D -- q5_rlvr_subjective (wide panel, 2.20 x 1.80 in).

Reads bootstrap_ci.json. Plots nothing that is not in that file.

Story: on subjective TruthfulQA items graded by a sycophantic majority, ESA with frozen
V-trust recovers the correct answer 80% of the time; naive mean aggregation recovers 1%.

Bars are point estimates; error bars are the stored 95% bootstrap CIs over the 100 items
(5000 resamples), exactly as computed by exp_bootstrap_ci.py. Shares y-limits and ticks
with panels E and F via style.recovery_panel.
"""
import json

import style
from style import ESA, CAPTURED, WIDE_PANEL
import matplotlib.pyplot as plt

style.apply()

ci = json.load(open("bootstrap_ci.json"))["q5_q6"]
ESA_KEY, MEAN_KEY = "ESA (frozen V-trust)", "Mean"

values = [ci[ESA_KEY]["recovery"], ci[MEAN_KEY]["recovery"]]
cis = [tuple(ci[ESA_KEY]["ci95"]), tuple(ci[MEAN_KEY]["ci95"])]

fig, ax = plt.subplots(figsize=WIDE_PANEL)
style.recovery_panel(ax, ["ESA", "Naive mean"], values, cis, [ESA, CAPTURED],
                     ylabel="Recovery on TruthfulQA")
style.save(fig, "new results/q5_rlvr_subjective.pdf", WIDE_PANEL)
