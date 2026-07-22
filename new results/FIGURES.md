# Figure catalogue

One entry per PDF in this directory: what it plots, where the numbers come from, what the
result actually is, and what a caption has to say for the panel not to mislead. Numbers
here are read out of the data files, not retyped from the paper — where the two disagree,
the disagreement is flagged.

Nineteen figures, lettered A–S. Letters are the internal handle used in the `fig_*.py`
docstrings and in cross-references below; they are not the paper's numbering.

**Slots.** `full column` 3.05 x 2.10 in, `half column` 1.60 x 1.45 in, `wide panel`
2.20 x 1.80 in. All authored at final print size — include at native size, never scaled.

---

## A — `q2_coverage_boundary.pdf`
*full column · [fig_q2_coverage_boundary.py](../fig_q2_coverage_boundary.py) ·
`coverage_boundary_curves.npz`*

Cumulative latent regret over 5000 steps, three curves, ±1 SD over 20 seeds.

**Result.** An on/off transition at the coverage edge. With the divergence arm covered,
ESA recovers the latent optimum in **20/20 seeds** and its regret saturates — it identifies
the biased sources and stops paying. With the arm uncovered, ESA recovers in **0/20**, and
its curve coincides numerically with naive mean aggregation.

**Caption must carry.** The uncovered-ESA and naive-mean curves are the same curve. The
mean is drawn thicker underneath and uncovered-ESA dashed on top so both stay legible;
that they overlap is the result, not a plotting artifact.

---

## B — `q1_mdp_gridworld.pdf`
*half column · [fig_q1_mdp_gridworld.py](../fig_q1_mdp_gridworld.py) ·
`gridworld_candy_curves.npz`*

P(visit proxy goal) over 50k episodes, 100 seeds, band = 95% bootstrap CI **of the mean**.

**Result.** Final P(candy) over the last 200 episodes:

| Method | P(candy) | SD |
|---|---|---|
| Mean | 0.980 | 0.140 |
| Median | 1.000 | 0.000 |
| Global-trust ESA | 0.760 | 0.427 |
| Region-trust ESA | 0.000 | 0.001 |

Mean and median are fully trapped by the sycophantic majority. Per-region trust localizes
the audit and escapes completely. Global trust averages the rare divergence away and stays
mostly trapped.

**Caption must carry.** Two things. First, the band is a bootstrap CI of the mean, **not**
an SD band — global-trust ESA is bimodal across seeds (trapped in 76 of 100, escapes in
the rest), so its SD of 0.43 describes the spread of two outcomes rather than uncertainty
in the plotted curve. Second, **this contradicts the current paper text**: see
[the discrepancy note](README.md#discrepancy-gridworld-global-trust-esa-fig-b) — the claimed
0.93 / 0.07 should be 0.76 / 0.00.

---

## C — `q1_hopper_latent.pdf`
*half column · [fig_q1_hopper_latent.py](../fig_q1_hopper_latent.py) ·
`hopper_latent_curves.npz`*

Latent (true) return on Hopper-v4 under an 80% velocity-penalizing majority, 5 seeds,
±1 SD, rolling mean over 2000 env steps.

**Result.** Last-1k latent return: standard PPO **0.968 ± 0.005**, GAIL-style
**0.965 ± 0.009**, per-region ESA **2.365 ± 0.137**. Both majority-following learners
plateau just under 1.0; ESA audits the source and climbs past 2.3 — a 2.4x separation.
Matches the published 0.97 / 0.98 / 2.37.

**Caption must carry.** The GAIL-style curve optimizes the demonstrator's *recovered*
biased objective; what is plotted is the latent return that behaviour actually earns. This
is also the panel that stands in for the separate IRL figure — see
[Deliberate fold-in](#deliberate-fold-in).

---

## D — `q5_rlvr_subjective.pdf`
*wide panel · [fig_q5_rlvr_subjective.py](../fig_q5_rlvr_subjective.py) ·
`bootstrap_ci.json`*

Recovery on subjective TruthfulQA items graded by a real sycophantic panel (2 honest
Qwen2.5-32B + 8 sycophant Qwen2.5-7B). Error bars are stored 95% bootstrap CIs, 5000
resamples over the 100 items.

**Result.** ESA with frozen V-trust **0.80 [0.72, 0.87]**; naive mean **0.01 [0.00, 0.03]**.
The intervals are disjoint by a wide margin.

---

## E — `q5_rlvr_subjective_harsh.pdf`
*wide panel · [fig_q5_rlvr_subjective_harsh.py](../fig_q5_rlvr_subjective_harsh.py) ·
`rlvr_subjective_results_harsh.json`, `bootstrap_ci.json`*

Identical design to D so the collapse reads by direct comparison.

**Result.** ESA **0.00 [0.00, 0.00]**; naive mean **0.01 [0.00, 0.03]**. When the sycophant
grades *harshly* on the verifiable region it looks well calibrated there, ESA's frozen
V-trust is misplaced, and recovery collapses below even the naive mean. Residual trust mass
on the biased graders is **1.0** here, against ~3e-6 in D — the trust is not merely
diluted, it is fully concentrated on the wrong sources.

**Caption must carry.** The trust-mass number, which does not fit legibly at 2.2 in. And
the mechanism: what separates D from E is **cardinal calibration loss on the verifiable
region** (0.64 lenient vs 0.04 harsh), not cross-region drift — `delta_check_results.json`
records that drift does *not* separate the regimes and cardinal loss does. Panel R is the
evidence for the drift half of that claim.

**CI provenance.** ESA's recovery is exactly 0 over n=100, so the per-item 0/1 vector is
identically zero, every bootstrap resample has mean 0, and the 95% CI is necessarily
[0, 0] — derived, not assumed. The mean's interval is the stored one, unchanged because
the harsh run reports the same 0.01.

---

## F — `q5_ordinal_inversion.pdf`
*wide panel · [fig_q5_ordinal_inversion.py](../fig_q5_ordinal_inversion.py) ·
`bootstrap_ci.json`*

Same panel of graders, same items, two elicitation modalities.

**Result.** Cardinal (absolute scores) **0.80 [0.72, 0.87]**; ordinal (pairwise preference)
**0.00 [0.00, 0.00]**. ESA's detection signal lives entirely in the cardinal channel.
Eliciting absolute scores exposes the sycophant's miscalibration; eliciting the *same*
panel by pairwise preference — the modality RLHF and DPO use — hides it, trust inverts onto
the sycophant, and recovery goes to zero.

**Caption must carry.** This is a statement about RLHF/DPO's elicitation format, not about
ESA's aggregator. Both bars are the same method; the ordinal bar is hollow and hatched
because it is a variant, not a competitor.

---

## G — `q3_audit_cost.pdf`
*half column · [fig_q3_audit_cost.py](../fig_q3_audit_cost.py) · `audit_cost_results.json`*

Reference queries against horizon T, both axes log, 10 seeds, ±1 SD.

**Result.** Active auditing is **horizon-independent: flat at 32.2 ± 0.4** across three
decades of T (100 → 80,000), while recovering the optimum in **100% of seeds at every
horizon**. Fixed-rate auditing is linear in T, reaching **8028 ± 94** queries at T=80k.
Two and a half decades apart at the right edge. Matches the published ~32 / ~8000.

---

## H — `q6_ratio_sweep_real.pdf`
*half column · [fig_q6_ratio_sweep_real.py](../fig_q6_ratio_sweep_real.py) ·
`q6_ratio_sweep_results.json`*

Recovery on TruthfulQA as the sycophant fraction of a **real** grader panel is swept.

**Result.** ESA is **flat at 0.80 from 0% to 90% sycophantic**. Every majority-trusting
aggregator survives while sycophants are a minority and collapses past 0.5: Mean 0.80 →
0.32 at 0.5 → 0.00 at 0.9; Median and RRM the same; Wass-DRO likewise. Two exceptions worth
naming — Dawid-Skene tracks ESA at 0.80 all the way to 0.8 before collapsing to 0.13, and
KL-DRO holds a **partial 0.38 plateau** from 0.5 to 0.8.

**Caption must carry.** KL-DRO's 0.38 is not noise: entropic risk exp-tilts toward low
scores, so it discounts the sycophant's lenient inflation. It is exploiting the same
cardinal channel ESA uses, without a reference — which is why it beats every other
baseline and still loses to ESA. The frac=1.0 column is NaN for every method (an
all-sycophant panel has no honest grader to recover) and is dropped, not plotted as zero.

---

## I — `q7a_bandit_regret.pdf`
*half column · [fig_q7a_bandit_regret.py](../fig_q7a_bandit_regret.py) ·
`bandit_regret_curves.npz`*

Cumulative latent regret, 5000 steps, 25 seeds, ±1 SD. 80% of evaluators biased, divergence
arm **covered** — the regime the coverage-boundary theorem admits.

**Result.** ESA active-audit **138.9 ± 16.2** and flattening; Mean **2432.7 ± 3.6** and
Dawid-Skene **2471.9 ± 1.4**, both linear. An honest minority plus coverage is enough for
trust concentration to work.

**Caption must carry.** Mean and Dawid-Skene are 1.6% apart — under a point at this panel
size — so they share a color and one legend entry rather than pretending two
distinguishable lines exist.

---

## J — `q7b_bandit_regret_failsafe.pdf`
*half column · [fig_q7b_bandit_regret_failsafe.py](../fig_q7b_bandit_regret_failsafe.py) ·
`bandit_regret_curves.npz`*

Same environment, same seeds, but bias is **100% homogeneous**.

**Result.** With no honest minority, pure trust auditing lands exactly on the aggregators it
is meant to beat: ESA **2471.8 ± 1.2**, Mean **2472.2 ± 1.6**, Dawid-Skene **2472.2 ± 1.5** —
identical to within seed noise. Adding the absolute-distrust fail-safe (τ=0.5) recovers
**108.7 ± 3.1**: it detects that every source in the region is bad and falls back to the
reference, which exists only because the arm is covered.

**Caption must carry.** The three coincident curves are one visual line **by design**. That
ESA-without-fail-safe is invisible except where it separates *is* the result — the script
asserts the coincidence before merging the legend entries. Read I and J together: coverage
is necessary, an honest minority is not sufficient on its own.

---

## K — `q8a_ablation_bias_ratio.pdf`
*half column · [fig_q8a_ablation_bias_ratio.py](../fig_q8a_ablation_bias_ratio.py) ·
`bandit_ablation_curves.npz`*

ESA regret against biased fraction, 15 seeds, band = 95% bootstrap CI of the mean.
Fail-safe on (τ=0.5) for every curve.

**Result.** Final regret **132 / 141 / 149 / 146 / 110** at 50 / 70 / 90 / 95 / 100% biased —
flat within noise across the whole range. The honest minority does not have to be large, it
only has to exist; at 100% it does not, and what keeps regret bounded there is the fail-safe,
not trust concentration.

**Caption must carry.** The 100% curve is bounded *because the fail-safe is on*. Panel J is
the same setting with it switched off, where regret goes linear. Without that pairing this
panel over-claims.

---

## L — `q8b_ablation_eta.pdf`
*half column · [fig_q8b_ablation_eta.py](../fig_q8b_ablation_eta.py) ·
`bandit_ablation_curves.npz`*

Trust learning rate at fixed 80% bias, no fail-safe — trust concentration alone.

**Result.** Final regret **193 ± 26** at η=0.1, **139 ± 14** at η=0.5, **111 ± 2** at η=2.0.
Monotone across a 20x range, every setting bounded. Larger η concentrates trust sooner,
which shows up as an earlier knee, not a different outcome.

**Caption must carry.** η is not a tuned knob — nothing here is a hyperparameter-search
artifact.

---

## M — `q8c_ablation_ref_noise.pdf`
*half column · [fig_q8c_ablation_ref_noise.py](../fig_q8c_ablation_ref_noise.py) ·
`bandit_ablation_curves.npz`*

Reference quality: σ_ref ∈ {0, 0.1, 0.5, 1.0}, 80% biased, no fail-safe.

**Result.** A clean or lightly noisy reference is indistinguishable (**135 ± 5** and
**139 ± 14**). Past that, ESA does **not** degrade smoothly — it fails on a growing
*fraction* of seeds while the rest are unaffected:

| σ_ref | mean final regret | seeds diverged | non-diverged seeds |
|---|---|---|---|
| 0.5 | 455 (SD 700) | **2/15** | ~120 |
| 1.0 | 603 (SD 902) | **3/15** | ~125 |

**Caption must carry.** The bimodality, explicitly. The mean curve sits between two outcomes
that never occur, and the wide band is reporting exactly that — a reader who takes it as
smooth degradation has the wrong model of the failure. The diverging-seed counts are
annotated on the curves for this reason. This panel is also why all three ablations use
bootstrap CIs rather than ±1 SD: an SD band here floors below zero regret.

---

## N — `q6_baselines_sweep.pdf`
*half column · [fig_q6_baselines_sweep.py](../fig_q6_baselines_sweep.py) ·
`baselines_sweep_results.json`*

Simulated twin of panel H: same sweep, but on the bandit where the y-axis can be the
quantity the theory bounds — cumulative latent regret at T=5000, 15 seeds.

**Result.** Every majority-trusting aggregator collapses, at a ratio that differs by method:

| Method | first ratio with regret > 1000 |
|---|---|
| Mean | 0.2 |
| Wass-DRO | 0.3 |
| KL-DRO, RRM | 0.4 |
| Median, GAIL, Dawid-Skene | 0.5 |

ESA holds **~250 from 0.0 through 0.8** and collapses to 2498 only at 1.0, where the panel
is 100% homogeneous and no fail-safe is configured — the same boundary panel J measures in
time. The theoretical tipping point is ratio* = arm_gap / bias_mag = **0.17**, which is
where the naive Mean turns; the robust aggregators buy some margin past it, none survive
the majority point.

**Caption must carry.** ratio* is derived for the mean, and it is the mean that matches it.
The six collapsing baselines share one legend entry; the staircase of collapse ratios is
visible without six legend rows.

---

## O — `q4a_bloc_partition.pdf`
*half column · [fig_q4a_bloc_partition.py](../fig_q4a_bloc_partition.py) ·
`bloc_partition_results.json`*

Oracle-less covariance clustering of 20 evaluators, 10 seeds per ratio, chance = 0.50.

**Result.** Partition accuracy **0.80 / 1.00 / 0.90 / 0.70** at 30 / 50 / 60 / 80%
colluding. Well above chance throughout, including when the colluders are the majority.
Across-seed SD is **exactly 0.000** at all four ratios.

**Caption must carry.** The limitation, which is the actual point: clustering recovers the
*partition* but **cannot label which bloc is honest**. Both clusters are internally
coherent, and a coherent colluding bloc looks exactly like a coherent honest one from the
inside. This is the internal-only impossibility in miniature — correlation routing narrows
the question, the reference answers it. Also state the zero SD, since the panel shows no
error bars.

---

## P — `q4b_bloc_drift.pdf`
*half column · [fig_q4b_bloc_drift.py](../fig_q4b_bloc_drift.py) ·
`bloc_partition_results.json`*

Leading eigenvalue of the *residual* report covariance — per-step consensus subtracted, so
the common truth signal is stripped and only coordination remains — recent 200-step window
against the preceding one. Threshold 1.15.

**Result.** The statistic sits at **~1** while reports are independent. Twelve of twenty
evaluators start sharing a component at step 800; the statistic **first crosses the
threshold at step 825** and peaks near **11x**. Detection latency is one check interval.

**Caption must carry.** Why it falls back to ~1 after step 1100: this is a **change**
detector, not a state detector. Once both comparison windows sit inside the colluding phase
there is no jump left to see. The bloc has not dispersed — it has become the new baseline,
which is why the alarm's job is to re-flag regions for audit rather than to score them
continuously. Without this sentence the panel reads as the collusion ending.

---

## Q — `q5_rlvr_subjective_natural.pdf`
*wide panel · [fig_q5_rlvr_subjective_natural.py](../fig_q5_rlvr_subjective_natural.py) ·
`rlvr_subjective_results_natural.json`, `bootstrap_ci.json`*

The capability control, drawn identically to D.

**Result.** Both grader sides are Qwen2.5-32B — same architecture, same scale, same
greedy decoding — with sycophancy elicited naturally rather than by the coupling
instrument. ESA still recovers **0.80 [0.72, 0.87]**; naive mean still **0.01 [0.00, 0.03]**.

**Caption must carry.** This closes the obvious objection to D, where the sycophants were
7B and the honest graders 32B: ESA is not just detecting the weaker model. What it audits
is calibration on the verifiable region, not capability. Measured drift is 0.98 and flat
across every coupling value, so there is no sweep to plot — hence the two-bar form.

**CI provenance.** The stored intervals are reused, and this is exact rather than an
approximation: `ci()` bootstraps the mean of a per-item 0/1 vector over the same 100 items,
and that bootstrap distribution depends only on how many 1s the vector contains. Same
counts ⇒ same interval by construction. The script asserts the point estimates match before
reusing them.

---

## R — `q5_rlvr_subjective_32bvs32b.pdf`
*wide panel · [fig_q5_rlvr_subjective_32bvs32b.py](../fig_q5_rlvr_subjective_32bvs32b.py) ·
`rlvr_subjective_results_32bvs32b.json`*

Capability-matched panel with the coupling instrument sweeping measured cross-region drift.

**Result.** Δ moves from **0.842 down to 0.017** — a factor of fifty — and recovery does not
move at all: ESA flat at **0.80**, naive mean flat at **0.01**, at every one of the nine
points.

**Caption must carry.** The negative result is the point. Δ is **not** the variable that
decides whether ESA works, which is what makes panel E informative: E fails at Δ=0.96, but
not *because* of Δ — the separating quantity is cardinal calibration loss on the verifiable
region. Quote the interval once ([0.72, 0.87] at 0.80) rather than repeating it nine times
across a flat line.

---

## S — `q0_testbed_sycophant_trap.pdf`
*full column, 3.05 x 1.86 in · [fig_q0_testbed_sycophant_trap.py](../fig_q0_testbed_sycophant_trap.py) ·
`environments.py`*

The only figure with no data file — a schematic of Testbed 1, reading its geometry and every
reward value directly from `environments.py` rather than transcribing them.

**Result.** The same 10x10 world drawn twice. Latent objective: goal **+20**, candy **0**,
lava **-50**. What the 80% lazy-sycophant majority reports: candy **+10**, goal **+5**
(devalued because reaching it takes effort), lava **-1**. The optimum relocates from the top
right to the top left, and the agent maximizing the reported reward takes the same route but
stops at the candy.

**Caption must carry.** The lava term. The majority is not merely wrong about what is
*good* — reporting the lava at -1 instead of -50 makes it blind to what is *dangerous*, and
that is the part a reader skims past. Note also the 80% ratio, which is not in the panel.

**Deviations from house style,** both forced by one file holding two panels: in-axes panel
titles, and a two-line title so the reward values have somewhere to live at 7pt. It is also
the one figure whose height is not a slot value — both axes are aspect-locked squares, so
fixing the width fixes the height at 1.86 in, and padding to 2.10 would only add whitespace.

---

## Deliberate fold-in

**`q1_mdp_hopper_irl.png` has no PDF, on purpose.** `hopper_latent_curves.npz` holds the
`standard`, `gail` and `esa` curves from the same runs, and Fig C plots the GAIL-style curve
as one of its three series — that curve *is* the IRL baseline, optimizing the demonstrator's
recovered biased objective while its latent return is logged. A standalone IRL panel would
restate what C already shows, at the cost of a second half-column slot.

`q1_mdp_hopper.png` and `q3_audit_economy.png` are likewise not missing, just renamed:
C and G respectively. `q4_bloc_partition.png` and `q7_bandit_regret_v2.png` were each split
into two panels (O/P and I/J), because the house style is one axes per PDF so LaTeX can
place them as subfigures.

If you want the IRL panel to stand alone after all, it needs the MuJoCo stack below.

## What is missing

Two figures, from a single experiment:

| Old PNG | Blocker |
|---|---|
| `q9a_topology_dynamics.png` | [exp_mdp_topology.py](../exp_mdp_topology.py) — MuJoCo Hopper + SB3 PPO + networkx |
| `q9b_topology_graph.png` | same run |

The experiment spreads a sycophancy contagion over a Barabási–Albert evaluator network from
the highest-degree influencer and shows ESA quarantining infected nodes — their trust
collapses while healthy nodes keep theirs. It needs MuJoCo + Stable-Baselines3 + torch,
about 3 GB, which would not fit on this machine.

To rebuild, install the extras and follow the existing split: a `gen_topology.py` that dumps
per-step infection rate, trust in patient zero and mean healthy-node trust to an `.npz`,
then `fig_q9a` / `fig_q9b` that read it. The graph panel (q9b) will need a palette decision
— the original used `RdYlGn`, which is neither colorblind-safe nor in this set's palette; a
`style.ramp`-based node coloring would match.

## Cross-panel contracts

Panels meant to be compared are built to matching geometry, enforced in `style.py` rather
than repeated per figure:

* **D, E, F, Q** share y-limits and ticks (`RECOVERY_YLIM`) — every recovery-on-TruthfulQA
  bar in the paper is on one scale. R uses the same y-axis in line form.
* **I, J** share `REGRET_YLIM`, so the linear-regret baselines land at the same height.
* **K, L** share `ABLATION_YLIM`. **M sets its own** — it needs five times the range, and
  compressing K and L to fit would flatten the comparisons they exist to make.
* **B/C, G/H, I/J, K/L/M, O/P** are each exactly 1.450 in tall, so they typeset as pairs
  and triples without vertical drift.

## Reproducing

See [README.md](README.md) for the build commands. One trap worth repeating: **matplotlib
must be 3.10.x** — 3.11 dropped the bundled `cmsy10` face, so `×`, `η`, `Δ` and every other
mathtext glyph silently falls back to DejaVu Sans and stops matching the Computer Modern
body text. The failure is silent and easy to miss at 7pt.
