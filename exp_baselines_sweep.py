"""
Q6 -- baselines + sycophant-ratio sweep.
All majority-trusting aggregators (mean, median, Dawid-Skene, GAIL proxy, and the
WDPO/KLDPO/RRM analogues) collapse once bias becomes the majority. ESA (with the
divergence arm covered) stays robust. The theoretical tipping point is the ratio
at which the mean bias overtakes the latent gap: ratio* = arm_gap / bias_mag.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from social_bandit import SocialBanditEnv
from baselines import (Mean, Median, DawidSkene, GAILProxy,
                       RobustDRO_KL, RobustDRO_Wass, RRM)
from runners import run_esa, run_baseline

STEPS, SEEDS = 5000, 15
K, M, BIAS_ARM, BIAS_MAG, ARM_GAP = 10, 10, 1, 3.0, 0.5
RATIOS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
TIP = ARM_GAP / BIAS_MAG           # theoretical tipping point

def final_regret(ratio, which, seed):
    env = SocialBanditEnv(k=K, M=M, optimal_arm=0, bias_arm=BIAS_ARM,
                          bias_mag=BIAS_MAG, arm_gap=ARM_GAP,
                          sycophant_ratio=ratio, coverage="full", seed=seed)
    if which == "ESA":
        return run_esa(env, STEPS, audit_mode="active", seed=seed)["regret"][-1]
    agg = {"Mean": Mean, "Median": Median, "Dawid-Skene": lambda: DawidSkene(M),
           "GAIL": GAILProxy, "KL-DRO": RobustDRO_KL,
           "Wass-DRO": RobustDRO_Wass, "RRM": RRM}[which]()
    return run_baseline(env, agg, STEPS)["regret"][-1]

if __name__ == "__main__":
    print("Q6: baseline sweep (theoretical tipping point at ratio =", f"{TIP:.2f})")
    methods = ["Mean", "Median", "Dawid-Skene", "GAIL", "KL-DRO", "Wass-DRO", "RRM", "ESA"]
    curves = {mth: [] for mth in methods}
    for r in RATIOS:
        for mth in methods:
            vals = [final_regret(r, mth, s) for s in range(SEEDS)]
            curves[mth].append(np.mean(vals))
        print(f"  ratio={r:.1f}: " + "  ".join(f"{m}={curves[m][-1]:.0f}" for m in methods))

    plt.figure(figsize=(9.5, 6))
    styles = {"ESA": ("#6C3483", "-", 3)}
    for mth in methods:
        col, ls, lw = styles.get(mth, (None, "--", 1.8))
        plt.plot(RATIOS, curves[mth], ls, lw=lw, color=col, label=mth,
                 marker="o" if mth == "ESA" else None)
    plt.axvline(TIP, color="grey", ls=":", lw=2, label=f"theoretical tip ({TIP:.2f})")
    plt.axvline(0.5, color="black", ls=":", lw=1, alpha=0.5, label="majority (0.5)")
    plt.xlabel("sycophant ratio"); plt.ylabel(f"cumulative latent regret at T={STEPS}")
    plt.legend(ncol=2, fontsize=9); plt.grid(alpha=0.25); plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/q6_baselines_sweep.png", dpi=200)
    print("  saved paper/figures/q6_baselines_sweep.png")
