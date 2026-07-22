"""
Fig. Q -- q5_rlvr_subjective_natural (wide panel, 2.20 x 1.80 in).

Reads rlvr_subjective_results_natural.json (point estimates) and bootstrap_ci.json (the
intervals). Identical design to panels D and E so the three conditions read by direct
comparison.

Story: the capability control. In panel D the sycophantic graders are Qwen2.5-7B and the
honest ones are 32B, so a sceptic can say ESA is only detecting the weaker model. Here
BOTH sides are Qwen2.5-32B and the sycophancy is elicited naturally rather than by a
coupling instrument -- same architecture, same scale, same decoding. ESA still recovers
80% and the naive mean still gets 1%. What ESA audits is calibration on the verifiable
region, not model capability.

The coupling sweep is flat in this condition: the measured drift sits at 0.98 for every
coupling value, so there is no x-axis to plot against and the panel is the same two-bar
comparison as D. Panel R is the condition where the instrument does move drift.

CI PROVENANCE
  Both bars carry the stored 95% bootstrap intervals from bootstrap_ci.json rather than
  new ones, and this is exact rather than an approximation: exp_bootstrap_ci.ci()
  bootstraps the mean of a per-item 0/1 recovery vector over the same n=100 items, and
  the bootstrap distribution of such a vector's mean depends only on how many 1s it
  contains. This condition reports 0.80 and 0.01 on the same item set, i.e. the same
  counts, so it has the same interval by construction. (The stored ordinal-vs-cardinal
  entry, computed from a different panel, lands on [0.72, 0.87] for 0.80 -- the same
  interval, which is what that argument predicts.)
"""
import json

import style
from style import CAPTURED, ESA, WIDE_PANEL
import matplotlib.pyplot as plt

style.apply()

nat = json.load(open("rlvr_subjective_results_natural.json"))
ci = json.load(open("bootstrap_ci.json"))["q5_q6"]

# Flat sweep: assert it before taking the last point, or the "one condition" claim above
# is unchecked.
recs = {(s["rec_esa"], s["rec_mean"]) for s in nat["sweep"]}
assert len(recs) == 1, f"sweep is not flat in this condition: {recs}"
esa_rec, mean_rec = nat["sweep"][-1]["rec_esa"], nat["sweep"][-1]["rec_mean"]

# The CI reuse above is only valid if the point estimates match the stored ones.
assert esa_rec == ci["ESA (frozen V-trust)"]["recovery"], esa_rec
assert mean_rec == ci["Mean"]["recovery"], mean_rec

values = [esa_rec, mean_rec]
cis = [tuple(ci["ESA (frozen V-trust)"]["ci95"]), tuple(ci["Mean"]["ci95"])]

fig, ax = plt.subplots(figsize=WIDE_PANEL)
style.recovery_panel(ax, ["ESA", "Naive mean"], values, cis, [ESA, CAPTURED],
                     ylabel="Recovery on TruthfulQA")
style.save(fig, "new results/q5_rlvr_subjective_natural.pdf", WIDE_PANEL)
