"""
Data generator for Figs. I/J (q7a, q7b bandit regret). NO PLOTTING HERE.

Reuses the exact experiment from exp_bandit_regret.py (same env, same seeds, same
horizon) and dumps the per-seed cumulative latent regret traces so the figures can be
re-rendered without re-running the bandit.

Two conditions:
  A (q7a) 80% biased, divergence arm covered  -- honest minority + coverage, so
          per-region active-audit ESA concentrates trust on the honest sources.
  B (q7b) 100% homogeneous bias, covered      -- no honest minority, so pure trust
          auditing cannot recover; the absolute-distrust fail-safe (tau) detects that
          every source in the region is bad and falls back to the reference.

Writes bandit_regret_curves.npz: a [seeds, STEPS] float32 matrix per curve, keyed
<panel>_<method>, plus the scalar settings. Figs. I/J read that file and nothing else.

Run: python3 gen_bandit_regret.py --seeds 25
"""
import argparse
import json

import numpy as np

from baselines import DawidSkene, Mean
from runners import run_baseline, run_esa
from social_bandit import SocialBanditEnv

K, M = 5, 10
OPT, BIAS_ARM, GAP, BIAS_MAG = 0, 1, 0.5, 1.0
STEPS = 5000

# (curve key, kind, tau) per panel. tau=None -> no absolute-distrust fail-safe.
PANELS = {
    "a": dict(sycophant_ratio=0.8, coverage="full",
              curves=[("mean", "mean", None), ("dawid", "dawid", None),
                      ("esa", "esa", None)]),
    "b": dict(sycophant_ratio=1.0, coverage="full",
              curves=[("mean", "mean", None), ("dawid", "dawid", None),
                      ("esa", "esa", None), ("esa_fs", "esa", 0.5)]),
}


def curve(kind, seed, sycophant_ratio, coverage, tau):
    env = SocialBanditEnv(k=K, M=M, optimal_arm=OPT, bias_arm=BIAS_ARM, arm_gap=GAP,
                          bias_mag=BIAS_MAG, sycophant_ratio=sycophant_ratio,
                          coverage=coverage, seed=seed)
    if kind == "mean":
        return run_baseline(env, Mean(), STEPS)["regret"]
    if kind == "dawid":
        return run_baseline(env, DawidSkene(M), STEPS)["regret"]
    if kind == "esa":
        return run_esa(env, STEPS, audit_mode="active", tau=tau,
                       trust_scope="region", seed=seed)["regret"]
    raise ValueError(kind)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    args = ap.parse_args()

    out = {}
    summary = {"steps": STEPS, "seeds": args.seeds, "k": K, "M": M,
               "arm_gap": GAP, "bias_mag": BIAS_MAG, "final": {}}

    for panel, cfg in PANELS.items():
        print(f"panel {panel}: sycophant_ratio={cfg['sycophant_ratio']} "
              f"coverage={cfg['coverage']}", flush=True)
        for key, kind, tau in cfg["curves"]:
            R = np.stack([curve(kind, s, cfg["sycophant_ratio"], cfg["coverage"], tau)
                          for s in range(args.seeds)]).astype(np.float32)
            out[f"{panel}_{key}"] = R
            summary["final"][f"{panel}_{key}"] = [float(R.mean(0)[-1]),
                                                 float(R.std(0)[-1])]
            print(f"  {key:8s} final regret {R.mean(0)[-1]:8.1f} +/- {R.std(0)[-1]:5.1f}",
                  flush=True)

    np.savez_compressed("bandit_regret_curves.npz", **out)
    json.dump(summary, open("bandit_regret_summary.json", "w"), indent=2)
    print("saved bandit_regret_curves.npz + bandit_regret_summary.json")
