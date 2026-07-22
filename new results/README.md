# New results — redesigned AAAI figures

Vector PDFs for the two-column AAAI submission. Every number, curve and band in these
figures comes from a data file in the repo; nothing is redrawn or approximated from the
old PNGs.

## Build

```
python gen_coverage_boundary.py          # -> coverage_boundary_curves.npz      (Fig A)
python gen_gridworld.py --seeds 100      # -> gridworld_candy_curves.npz        (Fig B)
python gen_hopper_curves.py              # -> hopper_latent_curves.npz          (Fig C)
python gen_audit_cost.py                 # -> audit_cost_results.json           (Fig G)
python gen_bandit_regret.py --seeds 25   # -> bandit_regret_curves.npz          (Figs I, J)
python gen_bandit_ablations.py           # -> bandit_ablation_curves.npz        (Figs K, L, M)
python gen_baselines_sweep.py            # -> baselines_sweep_results.json      (Fig N)
python gen_bloc_partition.py             # -> bloc_partition_results.json       (Figs O, P)

for f in fig_*.py; do python "$f"; done  # -> "new results"/*.pdf
```

The `gen_*.py` scripts only run experiments and dump data; the `fig_*.py` scripts only
read data and draw. Re-rendering a figure never re-runs an experiment. All styling lives
in `style.py`; no figure overrides it.

The four bandit/bloc generators are pure numpy and take about four minutes together. The
three that need a data file they cannot rebuild here are noted under *Not regenerated*.

**matplotlib must be 3.10.x.** 3.11 dropped the bundled `cmsy10` face, so `\times`,
`\eta`, `\Delta` and every other mathtext glyph silently falls back to DejaVu Sans and
stops matching the Computer Modern body text. `pip install 'matplotlib==3.10.9'`.

## Figures

| PDF | Slot | Size (in) | Data source |
|---|---|---|---|
| A `q2_coverage_boundary.pdf` | full column | 3.05 x 2.10 | `coverage_boundary_curves.npz` |
| B `q1_mdp_gridworld.pdf` | half column | 1.60 x 1.45 | `gridworld_candy_curves.npz` |
| C `q1_hopper_latent.pdf` | half column | 1.60 x 1.45 | `hopper_latent_curves.npz` |
| D `q5_rlvr_subjective.pdf` | wide panel | 2.20 x 1.80 | `bootstrap_ci.json` |
| E `q5_rlvr_subjective_harsh.pdf` | wide panel | 2.20 x 1.80 | `rlvr_subjective_results_harsh.json`, `bootstrap_ci.json` |
| F `q5_ordinal_inversion.pdf` | wide panel | 2.20 x 1.80 | `bootstrap_ci.json` |
| G `q3_audit_cost.pdf` | half column | 1.60 x 1.45 | `audit_cost_results.json` |
| H `q6_ratio_sweep_real.pdf` | half column | 1.60 x 1.45 | `q6_ratio_sweep_results.json` |
| I `q7a_bandit_regret.pdf` | half column | 1.60 x 1.45 | `bandit_regret_curves.npz` |
| J `q7b_bandit_regret_failsafe.pdf` | half column | 1.60 x 1.45 | `bandit_regret_curves.npz` |
| K `q8a_ablation_bias_ratio.pdf` | half column | 1.60 x 1.45 | `bandit_ablation_curves.npz` |
| L `q8b_ablation_eta.pdf` | half column | 1.60 x 1.45 | `bandit_ablation_curves.npz` |
| M `q8c_ablation_ref_noise.pdf` | half column | 1.60 x 1.45 | `bandit_ablation_curves.npz` |
| N `q6_baselines_sweep.pdf` | half column | 1.60 x 1.45 | `baselines_sweep_results.json` |
| O `q4a_bloc_partition.pdf` | half column | 1.60 x 1.45 | `bloc_partition_results.json` |
| P `q4b_bloc_drift.pdf` | half column | 1.60 x 1.45 | `bloc_partition_results.json` |
| Q `q5_rlvr_subjective_natural.pdf` | wide panel | 2.20 x 1.80 | `rlvr_subjective_results_natural.json`, `bootstrap_ci.json` |
| R `q5_rlvr_subjective_32bvs32b.pdf` | wide panel | 2.20 x 1.80 | `rlvr_subjective_results_32bvs32b.json` |
| S `q0_testbed_sycophant_trap.pdf` | full column | 3.05 x 1.86 | `environments.py` (schematic — no experiment) |

Pages are cropped tight (`pad_inches=0.02`) yet land on the slot size to within 0.005 in,
because `style.save` measures the actual cropped page and iterates the canvas size.
**Include them at native size — do not scale.** Fig S is the one exception to the height
column: its two panels are aspect-locked squares, so fixing the width fixes the height at
1.86 in, and it is saved with `size=(FULL_COL[0], None)` rather than padded out to 2.10.

Panels that must be read against each other are built to matching geometry: D/E/F/Q share
y-limits and ticks; I/J share them via `style.REGRET_YLIM`; K/L share them via
`style.ABLATION_YLIM`; B/C, G/H, I/J, K/L/M and O/P are each 1.450 in tall.

## Bands — what each one is

* Fig A: +/-1 SD over 20 seeds.
* Fig B: 95% bootstrap CI **of the mean** over 100 seeds (see discrepancy note below).
* Fig C: +/-1 SD over 5 seeds.
* Figs D/E/F/Q: stored 95% bootstrap CIs, 5000 resamples over the 100 items.
* Fig G: +/-1 SD over 10 seeds.
* Figs I/J: +/-1 SD over 25 seeds.
* Figs K/L/M: 95% bootstrap CI **of the mean** over 15 seeds (`style.boot_ci`). Not SD,
  for the reason Fig B is not: panel M is bimodal across seeds, and one estimator across
  the trio keeps the three comparable.
* Figs N/O/P/R: no band. N and R plot a single point estimate per x; O's across-seed SD
  is exactly 0.000 at all four ratios; P is a single deterministic run.

Captions must match these. Fig B is a bootstrap CI, **not** an SD band.

## Things the figures say that the caption has to carry

* **Fig J** — Mean, Dawid-Skene and fail-safe-less ESA all end at 2472 +/- 1. They are one
  visual line by design, not an omission: at 100% homogeneous bias, trust auditing lands
  exactly on the aggregators it is supposed to beat. The script asserts the coincidence
  before merging the legend entries.
* **Fig M** — ESA does not degrade smoothly as the reference gets noisy, it fails on a
  growing FRACTION of seeds: 2/15 at sigma_ref=0.5, 3/15 at 1.0, with the rest near 120.
  The counts are annotated on the curves; the wide band is that bimodality, not noise.
* **Fig O** — clustering recovers the partition but **cannot label which bloc is honest**.
  Both clusters are internally coherent. This is the internal-only impossibility, and it
  is the reason the audit is not optional.
* **Fig P** — the drift statistic returns to ~1 after step 1100 because it compares
  consecutive windows: once both sit inside the colluding phase there is no jump left to
  detect. The bloc has not dispersed, it has become the new baseline.
* **Fig R** — recovery is invariant to measured drift Delta across a factor of fifty, so
  Delta is not what decides whether ESA works. That is what makes Fig E's failure
  informative: it is cardinal calibration loss, not drift.
* **Fig Q** — both grader sides are Qwen2.5-32B here, so the 80% recovery in Fig D is not
  ESA detecting the weaker 7B model.
* **Fig S** — the sycophantic majority also reports the lava at -1 instead of -50. The
  proxy is not only wrong about what is good, it is blind to what is dangerous.

## Not regenerated

Three of the old PNGs in `paper/figures/` have no PDF here, because their experiments need
MuJoCo + Stable-Baselines3 + torch (~3 GB) and this machine could not host them:

| Old PNG | Blocker |
|---|---|
| `q9a_topology_dynamics.png` | `exp_mdp_topology.py` — MuJoCo Hopper + SB3 PPO + networkx |
| `q9b_topology_graph.png` | same run as q9a |
| `q1_mdp_hopper_irl.png` | `exp_mdp_hopper_irl.py` — MuJoCo Hopper + SB3 PPO |

`q1_mdp_hopper.png` and `q1_mdp_hopper_irl.png` are both **already covered by Fig C**:
`hopper_latent_curves.npz` holds the `standard`, `gail` and `esa` curves from the same
runs, and the GAIL-style curve is that IRL baseline. Only the q9 topology pair is
genuinely missing. To rebuild it, install the extras and follow the existing split — add
a `gen_topology.py` that dumps the per-step infection rate and per-node trust to an
`.npz`, then a `fig_q9*.py` that reads it.

## Reproduction status

Regenerated numbers that match the published claims:

* Hopper (Fig C): standard PPO 0.968 +/- 0.005, GAIL-style 0.965 +/- 0.009,
  per-region ESA 2.365 +/- 0.137 — matches `CHANGELOG.md:79` (0.97 / 0.98 / 2.37).
* Audit economy (Fig G): active flat at 32.2 across three decades of T; fixed-rate
  linear to 8028 at T=80k — matches `CHANGELOG.md:35` (~32 / ~8000).
* Coverage boundary (Fig A): ESA recovers in 100% of seeds when the divergence arm is
  covered, 0% when it is not; the uncovered arm coincides with the naive mean.
* Bandit regret (Figs I, J): 80% biased and covered — ESA 138.9 +/- 16.2 against Mean
  2432.7 and Dawid-Skene 2471.9. 100% homogeneous — pure ESA 2471.8 (fails, as the
  internal-only impossibility predicts), ESA + absolute-distrust fail-safe 108.7.
* Baseline sweep (Fig N): every majority-trusting aggregator goes linear at or before the
  majority point — Mean at 0.2 (the theoretical tipping point ratio* = arm_gap / bias_mag
  = 0.17 is derived for the mean, and it is the mean that matches it), Wass-DRO by 0.3,
  KL-DRO and RRM by 0.4, Median/GAIL/Dawid-Skene by 0.5. ESA holds ~250 through 0.8 and
  collapses only at 1.0, where no honest minority exists and no fail-safe is configured.

### Discrepancy: gridworld global-trust ESA (Fig B)

`CHANGELOG.md:78` and `paper/aaai27_esa_skeleton.tex:296` claim
**global-trust ESA 0.93, region-trust ESA 0.07**. The current code does not reproduce
0.93 at any horizon:

| Method | Claimed | 100 seeds, 50k episodes |
|---|---|---|
| Mean | 1.00 | 0.980 +/- 0.140 |
| Median | 1.00 | 1.000 +/- 0.000 |
| Global-trust ESA | **0.93** | **0.760 +/- 0.427** |
| Region-trust ESA | **0.07** | **0.000 +/- 0.001** |

It is not an episode-count artifact: global-ESA sits at 0.665 (15 seeds) both at episode
3000 and at 50k. The cause is that global-trust ESA is **bimodal across seeds** — it is
trapped in 76 of 100 seeds and escapes in the rest — so its across-seed SD is ~0.43 and
the mean is a poor summary. The seed count was raised from 15 to 100 to make the trapped
fraction stable; the 95% CI on 0.760 is roughly [0.68, 0.84], which excludes 0.93.

The qualitative claim is unaffected: mean and median are fully trapped, global-trust ESA
is mostly trapped, and per-region ESA escapes. **The 0.93 / 0.07 figures in the paper
text and CHANGELOG still need updating to 0.76 / 0.00,** and the bimodality deserves a
caption sentence.

## Caveat: folder name

This directory name contains a space. `\includegraphics{new results/q2_...}` needs the
path braced or the directory renamed (e.g. `new_results`) before it will compile.
