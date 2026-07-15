#!/usr/bin/env bash
# One-command reproduction. GPU-free figures regenerate from seeds/cache in minutes.
# The MuJoCo/LLM figures require GPU + the model panel and are listed, not run.
set -e
cd "$(dirname "$0")"

echo "== [1/3] synthetic bandit + gridworld figures (numpy, seeded) =="
for e in coverage_boundary audit_economy bloc_partition baselines_sweep; do python3 "exp_$e.py"; done
python3 exp_bandit_regret.py
python3 exp_bandit_ablations.py
python3 exp_testbed_figure.py
python3 exp_mdp_gridworld.py            # 15 seeds x 3000 eps, a few minutes, GPU-free

echo "== [2/3] real-grader Q5/Q6 figures + JSONs (GPU-free; need the score caches) =="
if [ -f rlvr_subjective_cache.json ]; then
  python3 exp_rlvr_subjective.py --n-v 120 --n-n 100 --inject-belief-on-V
  python3 exp_q6_real_panel.py
  python3 exp_q6_ratio_sweep.py
  python3 exp_rlvr_ordinal_check.py
  python3 exp_delta_check.py
  python3 exp_bootstrap_ci.py
  python3 exp_bloc_real.py
else
  echo "  SKIP: score caches absent. Regenerate with a GPU run of exp_rlvr_subjective.py"
  echo "        (32B honest + 7B syco panel) and exp_rlvr_subjective.py --harsh-syco --inject... variants."
fi

echo "== [3/3] GPU + MuJoCo required (run manually on a GPU box) =="
echo "  python3 exp_mdp_hopper.py --seeds 5"
echo "  python3 exp_mdp_hopper_irl.py"
echo "  python3 exp_mdp_topology.py"
echo "  full real-grader panel from scratch: exp_rlvr_subjective.py (downloads Qwen2.5-32B + 7B)"

echo "== smoke test =="
python3 quick_check.py
