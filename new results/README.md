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

for f in fig_*.py; do python "$f"; done  # -> "new results"/*.pdf
```

The `gen_*.py` scripts only run experiments and dump data; the `fig_*.py` scripts only
read data and draw. Re-rendering a figure never re-runs an experiment. All styling lives
in `style.py`; no figure overrides it.

## Figures

| PDF | Slot | Size (in) | Data source |
|---|---|---|---|
| `q2_coverage_boundary.pdf` | full column | 3.05 x 2.10 | `coverage_boundary_curves.npz` |
| `q1_mdp_gridworld.pdf` | half column | 1.60 x 1.45 | `gridworld_candy_curves.npz` |
| `q1_hopper_latent.pdf` | half column | 1.60 x 1.45 | `hopper_latent_curves.npz` |
| `q3_audit_cost.pdf` | half column | 1.60 x 1.45 | `audit_cost_results.json` |
| `q6_ratio_sweep_real.pdf` | half column | 1.60 x 1.45 | `q6_ratio_sweep_results.json` |
| `q5_rlvr_subjective.pdf` | wide panel | 2.20 x 1.80 | `bootstrap_ci.json` |
| `q5_rlvr_subjective_harsh.pdf` | wide panel | 2.20 x 1.80 | `rlvr_subjective_results_harsh.json`, `bootstrap_ci.json` |
| `q5_ordinal_inversion.pdf` | wide panel | 2.20 x 1.80 | `bootstrap_ci.json` |

Pages are cropped tight (`pad_inches=0.02`) yet land on the slot size to within 0.002 in,
because `style.save` measures the actual cropped page and iterates the canvas size.
**Include them at native size — do not scale.** D/E/F share y-limits and ticks exactly;
G and H are both 1.450 in tall; B and C are both 1.450 in tall.

## Bands — what each one is

* Fig A: +/-1 SD over 20 seeds.
* Fig B: 95% bootstrap CI **of the mean** over 100 seeds (see discrepancy note below).
* Fig C: +/-1 SD over 5 seeds.
* Figs D/E/F: stored 95% bootstrap CIs, 5000 resamples over the 100 items.
* Fig G: +/-1 SD over 10 seeds.

Captions must match these. Fig B is a bootstrap CI, **not** an SD band.

## Reproduction status

Regenerated numbers that match the published claims:

* Hopper (Fig C): standard PPO 0.968 +/- 0.005, GAIL-style 0.965 +/- 0.009,
  per-region ESA 2.365 +/- 0.137 — matches `CHANGELOG.md:79` (0.97 / 0.98 / 2.37).
* Audit economy (Fig G): active flat at 32.2 across three decades of T; fixed-rate
  linear to 8028 at T=80k — matches `CHANGELOG.md:35` (~32 / ~8000).
* Coverage boundary (Fig A): ESA recovers in 100% of seeds when the divergence arm is
  covered, 0% when it is not; the uncovered arm coincides with the naive mean.

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
