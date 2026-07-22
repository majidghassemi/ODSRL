"""
Fig. R -- q5_rlvr_subjective_32bvs32b (wide panel, 2.20 x 1.80 in).

Reads rlvr_subjective_results_32bvs32b.json. Plots nothing that is not in that file.

Story: the drift-invariance check, and the panel that makes panel E's failure mean
something. Both grader sides are Qwen2.5-32B, and the coupling instrument sweeps the
measured cross-region drift Delta from 0.84 down to 0.02 -- a factor of fifty. Recovery
does not move: ESA holds 0.80 at every Delta and the naive mean holds 0.01. So Delta is
NOT the variable that decides whether ESA works, which is why the harsh condition (panel
E, Delta = 0.96, recovery 0) cannot be explained by drift either -- what separates the
regimes there is cardinal calibration loss on the verifiable region, not how far the
graders drift between regions.

A flat line is the result here, so the panel is drawn to make flatness legible: the same
recovery y-axis as panels D/E/F/Q, and Delta on x running high-to-low in the direction
the instrument sweeps it.

Bands, not error bars: recovery is a per-item 0/1 mean over the same n=100 items at every
Delta, and the sweep reports one point estimate per Delta with no stored interval, so
nothing is drawn that the file does not contain. The bootstrap interval at 0.80 is
[0.72, 0.87] (bootstrap_ci.json) and applies at every point identically -- it is quoted
in the caption rather than repeated nine times across a flat line.
"""
import json

import numpy as np

import style
from style import CAPTURED, ESA, LW_BASE, LW_ESA, WIDE_PANEL, DASH_MEAN
import matplotlib.pyplot as plt

style.apply()

d = json.load(open("rlvr_subjective_results_32bvs32b.json"))
sweep = sorted(d["sweep"], key=lambda s: s["drift"])
drift = np.array([s["drift"] for s in sweep])
esa = np.array([s["rec_esa"] for s in sweep])
mean = np.array([s["rec_mean"] for s in sweep])

fig, ax = plt.subplots(figsize=WIDE_PANEL)

ax.plot(drift, mean, color=CAPTURED, linestyle=DASH_MEAN, linewidth=LW_BASE,
        marker="s", markersize=2.6, label="Naive mean", zorder=2)
ax.plot(drift, esa, color=ESA, linewidth=LW_ESA, marker="o", markersize=2.6,
        label="ESA (frozen V-trust)", zorder=3)

ax.set_xlabel(r"Measured cross-region drift $\Delta$")
ax.set_ylabel("Recovery on TruthfulQA")
ax.set_xlim(0, 0.9)
ax.set_xticks([0, 0.25, 0.50, 0.75])
ax.set_xticklabels(["0", ".25", ".50", ".75"])
ax.set_ylim(*style.RECOVERY_YLIM)
ax.set_yticks(style.RECOVERY_YTICKS)
ax.set_yticklabels(style.RECOVERY_YTICKLABELS)
style.grid(ax)

# Least data-dense region: the middle band between the two flat lines.
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45))

style.save(fig, "new results/q5_rlvr_subjective_32bvs32b.pdf", WIDE_PANEL)
