# CHANGELOG — v2 (coverage-boundary rewrite)

This revision brings the code in line with the retargeted paper (AAAI-27 AI
Alignment track). The headline claim changed and the honest limitations are now
demonstrated, not hidden.

## Corrected claims (read this first)

| Old claim (v1) | Corrected claim (v2) |
|---|---|
| "ESA guarantees convergence even when 80% of evaluators are biased." | Recovery holds **iff the reference covers the region where bias could flip the decision** (Thm. 2). ESA recovers under coverage and **provably fails when the divergence arm is uncovered**. |
| Global scalar trust per evaluator. | **Per-region (contextual) trust**; global trust silences localized experts. |
| Fixed 10% spot-check forever. | **Active auditing** with a horizon-independent budget (~32 audits vs thousands for fixed-rate). |
| "Sparse safety axioms" as the reference. | **Verifiable subtasks (RLVR)** supply the reference for free where a ground truth exists; transfer to non-verifiable tasks is a *measured* assumption. |
| 100% bias handled by trust. | Pure trust-auditing **fails at 100% homogeneous bias** (no honest minority). The **absolute-distrust fail-safe** covers that corner by abstaining / using the reference. |

## New / upgraded modules (`src/`)

| File | Purpose | Paper section |
|---|---|---|
| `social_bandit.py` | Bandit env with a **coverage set**, full `[M,K]` bias matrix, V/N split for RLVR, reference querying. | Setup; Def. reference-with-coverage |
| `esa.py` | ESA agent: per-region trust, active auditing (successive elimination + agreement-based resolution), absolute-distrust fail-safe, correlation routing (`detect_blocs`, `drift_alarm`). | §Algorithm; B5/B6 |
| `baselines.py` | Mean, Median, Dawid-Skene, GAIL proxy, **+ WDPO/KLDPO/RRM analogues** (documented as scalar-reward analogues, not reimplementations). | §Experiments Q6 |
| `runners.py` | `run_esa`, `run_baseline` (UCB loop) → regret curve, final arm, audit count. | — |
| `esa_mdp.py` | MDP-side contextual-trust mechanism for the Hopper/PPO harness. **Not runnable here** (needs MuJoCo/SB3). | §Experiments Q1 (MDP) |

## New experiments (`src/exp_*.py`, all tested, figures → `paper/figures/`)

| Script | Result reproduced here |
|---|---|
| `exp_coverage_boundary.py` (Q2) | ESA recovers 100% when divergence covered, **0% when uncovered**; Mean 0%. **The load-bearing failure experiment.** |
| `exp_audit_economy.py` (Q3) | Active auditing flat at ~32 queries across T=1k–80k; fixed-rate grows linearly to ~8000. |
| `exp_bloc_partition.py` (Q4) | Oracle-less covariance clustering recovers the partition (0.7–1.0 acc.) under a colluding majority; **cannot label** which bloc is honest. Drift alarm fires at step ~825 after a bloc forms at 800. |
| `exp_rlvr_transfer.py` (Q5) | Train trust on V, deploy on N: N-recovery rises 10% → 100% as cross-region coupling rises. |
| `exp_baselines_sweep.py` (Q6) | Every majority-trusting baseline collapses at its tipping point; ESA holds flat 0–80%, collapses only at 100% (fail-safe territory). |

Run all: `for e in coverage_boundary audit_economy bloc_partition rlvr_transfer baselines_sweep; do python3 src/exp_$e.py; done`

## Honest notes for the baselines

`RobustDRO_KL` (KLDPO-analogue), `RobustDRO_Wass` (WDPO-analogue), and `RRM`
(RRM-analogue) are **scalar-reward aggregation analogues** of the cited LLM
preference-optimization methods, not verbatim reimplementations. They encode the
same robustness assumption (sparse corruption / mostly-honest crowd) so they fail
under a systematic majority by design. State this in the paper; do not claim they
are the original methods.

## Still needs real compute (not run in this environment)

- MDP experiments (Hopper strategic adversary, IRL/GAIL) via `advanced_experiments.py`
  with `esa_mdp.ContextualTrust` wired in — needs MuJoCo + Stable-Baselines3.
- A **real RLVR verifier** (math/code checker) for Q5 instead of the synthetic V/N
  split, and a **measured** `Delta_drift` on held-out labeled prompts.

## Dependencies

Unchanged: `numpy`, `matplotlib`, `tqdm`. The bandit code and clustering use only
numpy (`np.linalg.eigh`, `np.corrcoef`); no scipy/sklearn required.
