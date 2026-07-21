"""
Fig. E -- q5_rlvr_subjective_harsh (wide panel, 2.20 x 1.80 in).

Reads rlvr_subjective_results_harsh.json (point estimates) and bootstrap_ci.json (the
naive-mean interval). Identical design to panel D so the collapse reads by comparison.

Story: when the sycophant grades HARSHLY on the verifiable region, it looks well
calibrated there, ESA's frozen V-trust is misplaced, and recovery collapses to 0 --
below even the naive mean. The coverage condition is what D has and E lacks.

CI PROVENANCE
  naive mean : stored 95% bootstrap CI from bootstrap_ci.json. The mean baseline does
               not depend on ESA's trust, and the harsh run reports the same 0.01
               recovery, so the stored interval applies unchanged.
  ESA (harsh): recovery is exactly 0.00 over the n=100 items. A 0/1 per-item vector with
               mean exactly 0 is identically zero, so every bootstrap resample has mean
               0 and the 95% CI is necessarily [0, 0] -- the same degenerate interval
               exp_bootstrap_ci.py stores for the ordinal condition. Derived, not assumed.

The trust-mass concentration (resid_biased_mass = 1.0 here vs ~3e-6 in D) does not fit
legibly at 2.2 in alongside the bars; it is left to the caption.
"""
import json

import style
from style import ESA, CAPTURED, WIDE_PANEL
import matplotlib.pyplot as plt

style.apply()

harsh = json.load(open("rlvr_subjective_results_harsh.json"))
mean_ci = json.load(open("bootstrap_ci.json"))["q5_q6"]["Mean"]

# The harsh sweep is flat in the coupling parameter; every entry carries the same
# recovery, so take the final (fully-coupled) point.
last = harsh["sweep"][-1]
esa_rec, mean_rec = last["rec_esa"], last["rec_mean"]
assert esa_rec == 0.0, f"expected exact 0 recovery for the [0,0] CI argument, got {esa_rec}"
assert mean_rec == mean_ci["recovery"], "harsh mean differs from the stored CI's point estimate"

values = [esa_rec, mean_rec]
cis = [(0.0, 0.0), tuple(mean_ci["ci95"])]

fig, ax = plt.subplots(figsize=WIDE_PANEL)
style.recovery_panel(ax, ["ESA", "Naive mean"], values, cis, [ESA, CAPTURED],
                     ylabel="Recovery on TruthfulQA")
style.save(fig, "new results/q5_rlvr_subjective_harsh.pdf", WIDE_PANEL)
