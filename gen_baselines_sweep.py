"""
Data generator for Fig. N (q6_baselines_sweep). NO PLOTTING HERE.

Reuses the exact experiment from exp_baselines_sweep.py (same env, same seeds, same
horizon): cumulative latent regret at T for every aggregator as the sycophant ratio is
swept. The theoretical tipping point is the ratio at which the mean bias overtakes the
latent gap, ratio* = arm_gap / bias_mag.

This is the SIMULATED counterpart of Fig. H (q6_ratio_sweep_real, real graders): there
the y-axis is recovery on TruthfulQA, here it is cumulative latent regret on the bandit.

Writes baselines_sweep_results.json: per-method mean and SD of the final regret at each
ratio, over the seeds. Fig. N reads that file and nothing else.

Run: python3 gen_baselines_sweep.py --seeds 15
"""
import argparse
import json

import numpy as np

from baselines import (DawidSkene, GAILProxy, Mean, Median, RRM, RobustDRO_KL,
                       RobustDRO_Wass)
from runners import run_baseline, run_esa
from social_bandit import SocialBanditEnv

STEPS = 5000
K, M, BIAS_ARM, BIAS_MAG, ARM_GAP = 10, 10, 1, 3.0, 0.5
RATIOS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
TIP = ARM_GAP / BIAS_MAG                # theoretical tipping point
METHODS = ["Mean", "Median", "Dawid-Skene", "GAIL", "KL-DRO", "Wass-DRO", "RRM", "ESA"]

AGGREGATORS = {"Mean": Mean, "Median": Median, "Dawid-Skene": lambda: DawidSkene(M),
               "GAIL": GAILProxy, "KL-DRO": RobustDRO_KL,
               "Wass-DRO": RobustDRO_Wass, "RRM": RRM}


def final_regret(ratio, which, seed):
    env = SocialBanditEnv(k=K, M=M, optimal_arm=0, bias_arm=BIAS_ARM,
                          bias_mag=BIAS_MAG, arm_gap=ARM_GAP,
                          sycophant_ratio=ratio, coverage="full", seed=seed)
    if which == "ESA":
        return run_esa(env, STEPS, audit_mode="active", seed=seed)["regret"][-1]
    return run_baseline(env, AGGREGATORS[which](), STEPS)["regret"][-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    args = ap.parse_args()

    print(f"Fig N data: {len(RATIOS)} ratios x {len(METHODS)} methods x {args.seeds} "
          f"seeds (tipping point at ratio={TIP:.2f})", flush=True)

    out = {"ratios": RATIOS, "steps": STEPS, "seeds": args.seeds, "tip": TIP,
           "arm_gap": ARM_GAP, "bias_mag": BIAS_MAG,
           "mean": {m: [] for m in METHODS}, "std": {m: [] for m in METHODS}}

    for r in RATIOS:
        for m in METHODS:
            vals = [final_regret(r, m, s) for s in range(args.seeds)]
            out["mean"][m].append(float(np.mean(vals)))
            out["std"][m].append(float(np.std(vals)))
        print(f"  ratio={r:.1f}: " +
              "  ".join(f"{m}={out['mean'][m][-1]:.0f}" for m in METHODS), flush=True)

    json.dump(out, open("baselines_sweep_results.json", "w"), indent=2)
    print("saved baselines_sweep_results.json")
