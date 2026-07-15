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
| `esa_mdp.py` | MDP-side contextual-trust mechanism (per-region trust, active auditing, fail-safe). Now **wired and run** in both MDP testbeds below. | §Experiments Q1 (MDP) |
| `environments.py`, `agents.py` | Gridworld sycophant-trap env + Q-learning baselines (Mean/Median/global-trust), ported from the original repo. | §Experiments Q1 (gridworld) |
| `exp_mdp_gridworld.py` | Wires `esa_mdp.ContextualTrust` (region = grid cell) into the sycophant trap; compares vs Mean/Median/global-trust. | §Experiments Q1 (gridworld) |
| `exp_mdp_hopper.py` | `StrategicHopperWrapper` with the global `TrustMechanism` replaced by `esa_mdp.ContextualTrust` (the two-line swap from the usage block); PPO on Hopper-v4. | §Experiments Q1 (Hopper) |

## New experiments (`src/exp_*.py`, all tested, figures → `paper/figures/`)

| Script | Result reproduced here |
|---|---|
| `exp_coverage_boundary.py` (Q2) | ESA recovers 100% when divergence covered, **0% when uncovered**; Mean 0%. **The load-bearing failure experiment.** |
| `exp_audit_economy.py` (Q3) | Active auditing flat at ~32 queries across T=1k–80k; fixed-rate grows linearly to ~8000. |
| `exp_bloc_partition.py` (Q4) | Oracle-less covariance clustering recovers the partition (0.7–1.0 acc.) under a colluding majority; **cannot label** which bloc is honest. Drift alarm fires at step ~825 after a bloc forms at 800. |
| `exp_rlvr_transfer.py` (Q5) | **RETIRED (synthetic)** — faked verifier + knob drift; figure removed from `paper/figures/`. Superseded by the real `exp_rlvr_real.py` and `exp_rlvr_subjective.py`. Script kept for reference only. |
| `exp_baselines_sweep.py` (Q6) | Every majority-trusting baseline collapses at its tipping point; ESA holds flat 0–80%, collapses only at 100% (fail-safe territory). |
| `exp_bandit_regret.py` (Q7) | v2 rewrite of the v1 regret comparison. **80% biased + covered**: ESA active-audit final regret **139 ± 16** vs Mean **2433 ± 4**, Dawid-Skene **2472 ± 1** (both linear). **100% homogeneous + covered**: pure ESA **fails** (2472, no honest minority) — the v1 over-claim — and only the **absolute-distrust fail-safe recovers (109 ± 3)** via the reference. Figure: `q7_bandit_regret_v2.png`. |

Run all: `for e in coverage_boundary audit_economy bloc_partition rlvr_transfer baselines_sweep; do python3 src/exp_$e.py; done`

## Full v1 → v2 figure reproduction

Every figure from the v1 repo now has a v2 equivalent, all rebuilt on the v2 stack
(per-region `ContextualTrust` for MDP, coverage-aware `esa.ESA` for bandits) and
written to `paper/figures/`:

| v1 figure | v2 figure (script) | result |
|---|---|---|
| testbed illustration | `q0_testbed1_sycophant_trap.png` (`exp_testbed_figure.py`) | v2 grid layout (Candy (0,9), Lava row y=5) |
| `exp1_sycophant_trap_with_median` | `q1_mdp_gridworld.png` (`exp_mdp_gridworld.py`) | Mean/Median/global-ESA all trapped (≥0.93), region-ESA escapes (0.07) |
| `sycophantic_hopper_performance/_trust` | `q1_mdp_hopper.png` (`exp_mdp_hopper.py`) | ESA 2.37 ± 0.27 vs PPO 0.97 ± 0.00, 5 seeds; trust-in-liars → 0.13 |
| `advanced_experiments_irl` | `q1_mdp_hopper_irl.png` (`exp_mdp_hopper_irl.py`) | IRL/GAIL 0.98, Standard 0.98, **ESA 1.92** (clean latent eval) |
| `exp2_bandit_regret_comparison_*` | `q7_bandit_regret_v2.png` (`exp_bandit_regret.py`) | see Q7 above (v2 coverage/fail-safe correction) |
| `ablation_bias_ratio/eta/internal_noise` | `q8a/q8b/q8c_*.png` (`exp_bandit_ablations.py`) | ESA flat 50–95%, fail-safe holds 100%; robust to η; graceful under ref noise |
| `social_topology_dynamics/_graph` | `q9a/q9b_topology_*.png` (`exp_mdp_topology.py`) | 95% infection, Patient-Zero trust → 0.005 (quarantined), healthy 0.055 |

Cleanup: all figures consolidated under `paper/figures/`; stale root duplicates and
scratch logs removed.

## Honest notes for the baselines

`RobustDRO_KL` (KLDPO-analogue), `RobustDRO_Wass` (WDPO-analogue), and `RRM`
(RRM-analogue) are **scalar-reward aggregation analogues** of the cited LLM
preference-optimization methods, not verbatim reimplementations. They encode the
same robustness assumption (sparse corruption / mostly-honest crowd) so they fail
under a systematic majority by design. State this in the paper; do not claim they
are the original methods.

## MDP experiments — now run (were pending real compute)

Both Q1 MDP testbeds are wired to `esa_mdp.ContextualTrust` and executed on real
compute (MuJoCo 3.10 + Stable-Baselines3 2.9, RTX 4090 box).

| Script | Result |
|---|---|
| `exp_mdp_gridworld.py` | Sycophant trap, 15 seeds × 3000 eps, 80% lazy sycophants. **P(visit Candy)**: Mean 1.00, Median 1.00, **global-trust ESA 0.93 (still trapped)**, **region-trust ESA 0.07 (escapes)**. This is *why* the old global-trust result did not match the paper: the sycophants are honest in most cells, so the rare divergence at the Candy cell is diluted under a single global weight; per-region trust localizes the audit and recovers. Figure: `q1_mdp_gridworld.png`. |
| `exp_mdp_hopper.py` | Hopper-v4, 50k steps, **5 seeds (mean ± std)**, strategic 80% lazy majority (`y = latent − bias·v_x`). **Latent (true) reward, last 1k**: Standard PPO **0.974 ± 0.004** vs **ESA region-trust 2.369 ± 0.268** (non-overlapping bands). Trust-in-liars driven 0.78 → **0.125 ± 0.031**. Figure: `q1_mdp_hopper.png`. |

## Real RLVR verifier for Q5 — built, armed for real compute

`exp_rlvr_real.py` replaces the synthetic Q5 (`z = true_r + noise`, hand-built bias
matrix, knob `Delta_drift`) with a real run: a local **Qwen2.5-3B-Instruct** grader
panel (2 honest + 8 verbose-biased rubric personas), a real **GSM8K exact-match
verifier** as the V reference, and a **measured** `Delta_drift` (per-grader
reliability_V − reliability_N). The `couple` knob is faithful to the synthetic
`n_vis`: it controls how much of the bias is exposed on V, and drift is measured
(not set) to fall as coupling rises. All LLM scores are cached to
`rlvr_real_cache.json`, so the run is checkpointed and resumes after interruption;
`--dry-run` validates the whole pipeline on CPU (done — mechanics confirmed).

Outputs: `q5_rlvr_real.png`, `rlvr_real_results.json`. Honest caveat: the panel is
one model under M rubrics (a reward-model-panel proxy), not M independently trained
reward models — but the scores, biases, and drift are real, not assumed.

**Key measured finding** (`exp_rlvr_real.py`): on verifiable GSM8K, a strong instruct
grader **refuses to be sycophantic** — no biased majority forms (7B "biased" graders
still prefer the correct answer ~88%). So the biased-majority regime is domain-
dependent; sycophancy needs a non-verifiable domain. This motivated option-2 below.

### Q5 option-2 — the load-bearing real result (`exp_rlvr_subjective.py`)

V = GSM8K (exact-match verifier); N = **TruthfulQA** (popular-misconception questions,
where sycophancy genuinely takes). **Heterogeneous panel**: 2 honest graders =
Qwen2.5-32B (4-bit, competent on adversarial N), 8 sycophants = Qwen2.5-7B. N gold =
Qwen2.5-32B as a blind neutral judge. Result (V=120, N=100):

| coupling | measured Δ_drift | N-recovery ESA | N-recovery mean |
|---|---|---|---|
| 0.00 | +1.00 | 0% | 1% |
| 0.25 | +0.80 | 0% | 1% |
| 0.50 | +0.61 | **80%** | 1% |
| 1.00 | +0.22 | **80%** | 1% |

Reads exactly as the theory: the sycophantic majority captures the naive mean on N
(1%); ESA transfers trust learned on verifiable V and recovers the honest signal
(80%) **once the bias is exposed on V** (coupling ≥ 0.5); below that the sycophancy is
hidden on V so trust can't be learned and transfer fails (ESA = mean ≈ 0%) — the
coverage boundary / Theorem-2 regime, with real graders and a strong-model gold. The
80% ceiling = the honest graders' own competence on N (`relN = 0.80`). Figure:
`q5_rlvr_subjective.png`. **Caveats to state:** (1) honest graders and the gold judge
are the same 32B, so honest N-reliability is somewhat favorably estimated; a distinct
strong model would be cleaner. (2) The synthetic `exp_rlvr_transfer.py` figure is
retired.

## Dependencies

Unchanged: `numpy`, `matplotlib`, `tqdm`. The bandit code and clustering use only
numpy (`np.linalg.eigh`, `np.corrcoef`); no scipy/sklearn required.
