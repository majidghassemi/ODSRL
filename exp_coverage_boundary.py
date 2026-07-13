"""
Q2 -- THE failure experiment (load-bearing).
Bias sits on arm `bias_arm`. We vary only whether that arm is in the coverage set.
Theorem: source auditing recovers the optimum iff the reference covers the region
where bias could flip the decision. So ESA must RECOVER when the divergence arm is
covered and FAIL when it is not, tracking the Mean baseline in the latter case.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from social_bandit import SocialBanditEnv
from baselines import Mean
from runners import run_esa, run_baseline

STEPS, SEEDS = 5000, 20
K, M, BIAS_ARM = 10, 10, 1
COV_ON = list(range(K))                        # bias arm covered
COV_OFF = [a for a in range(K) if a != BIAS_ARM]  # bias arm uncovered

def curves(cov, mode="esa", seeds=SEEDS):
    out = np.empty((seeds, STEPS))
    finals = []
    for s in range(seeds):
        env = SocialBanditEnv(k=K, M=M, optimal_arm=0, bias_arm=BIAS_ARM,
                              bias_mag=3.0, sycophant_ratio=0.8, coverage=cov, seed=s)
        if mode == "esa":
            r = run_esa(env, STEPS, audit_mode="active", seed=s)
        else:
            r = run_baseline(env, Mean(), STEPS)
        out[s] = r["regret"]; finals.append(r["final"])
    return out, np.mean([f == 0 for f in finals])

if __name__ == "__main__":
    print("Q2: coverage boundary")
    esa_on, rec_on = curves(COV_ON, "esa")
    esa_off, rec_off = curves(COV_OFF, "esa")
    base, rec_b = curves(COV_ON, "mean")
    print(f"  ESA (divergence covered)   recovery={rec_on:.0%}")
    print(f"  ESA (divergence UNcovered) recovery={rec_off:.0%}")
    print(f"  Mean baseline              recovery={rec_b:.0%}")

    x = np.arange(STEPS)
    plt.figure(figsize=(9, 6))
    for data, lab, col, ls in [
        (esa_on, "ESA (divergence covered)", "#6C3483", "-"),
        (esa_off, "ESA (divergence UNCOVERED)", "#C0392B", "--"),
        (base, "Mean (Dogma-4)", "#D35400", ":")]:
        m, sd = data.mean(0), data.std(0)
        plt.plot(x, m, col, ls=ls, lw=2.5, label=lab)
        plt.fill_between(x, m - sd, m + sd, color=col, alpha=0.15)
    plt.xlabel("Steps"); plt.ylabel("Cumulative latent regret")
    plt.title("Coverage boundary: recovery iff the divergence region is audited")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/q2_coverage_boundary.png", dpi=200)
    print("  saved paper/figures/q2_coverage_boundary.png")
