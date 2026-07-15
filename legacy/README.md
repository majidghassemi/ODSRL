# legacy/ — superseded v1 artifacts

These implement the **v1** claims (global scalar trust, unconditional
"recovers at 80% bias"). They are **kept, not deleted** (the AIES submissions
reference them) but are superseded by the v2 coverage-boundary code in the repo root.

| v1 artifact | superseded by (v2) |
|---|---|
| `exp_rlvr_transfer.py` (synthetic Q5: faked verifier `z=true_r+noise`, knob `Delta_drift`) | `exp_rlvr_real.py`, `exp_rlvr_subjective.py` (real LLM graders, exact-match verifier, measured drift) |

The following v1 run-scripts live in the **upstream repo** (`dogma4-feedback-collapse/src/`),
not in this working tree, and are likewise superseded:

| upstream v1 script | superseded by (v2) |
|---|---|
| `run_bandit.py` (global-trust bandit regret) | `exp_bandit_regret.py` (per-region + fail-safe) |
| `run_sycophant.py` (global-trust gridworld) | `exp_mdp_gridworld.py` (`esa_mdp.ContextualTrust`, per-region) |
| `run_ablations.py` | `exp_bandit_ablations.py` |
| `sycophantic_hopper.py`, `advanced_experiments.py` (global `TrustMechanism`) | `exp_mdp_hopper.py`, `exp_mdp_hopper_irl.py` (per-region `ContextualTrust`) |
| `social_topology_experiment.py` (global trust) | `exp_mdp_topology.py` (`ContextualTrust`) |

Note: `agents.py` in the repo root still contains the v1 global-trust
`InternalFeedbackAgent`/`DawidSkeneAgent`; those classes are superseded by the
per-region `ESAAgent` in `exp_mdp_gridworld.py`, but `agents.py` is retained because
its `BaseAgent`/`Dogma4Agent`/`MedianAgent` are used by the v2 gridworld experiment.
