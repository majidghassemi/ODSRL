"""
Q3 -- audit economy. Active auditing uses a horizon-independent number of reference
queries; fixed-rate (Bernoulli p_ref) grows linearly. Both recover under coverage.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from social_bandit import SocialBanditEnv
from runners import run_esa

HORIZONS = [1000, 2000, 5000, 10000, 20000, 40000, 80000]
SEEDS = 10

def audits(mode, T, seed):
    env = SocialBanditEnv(k=10, M=10, optimal_arm=0, bias_arm=1, bias_mag=3.0,
                          sycophant_ratio=0.8, coverage="full", seed=seed)
    r = run_esa(env, T, audit_mode=mode, p_ref=0.1, seed=seed)
    return r["audits"], r["final"] == 0

if __name__ == "__main__":
    print("Q3: audit economy")
    act_m, act_s, fix_m, fix_s, rec = [], [], [], [], []
    for T in HORIZONS:
        a = [audits("active", T, s) for s in range(SEEDS)]
        f = [audits("fixed", T, s) for s in range(SEEDS)]
        act = [x[0] for x in a]; fix = [x[0] for x in f]
        act_m.append(np.mean(act)); act_s.append(np.std(act))
        fix_m.append(np.mean(fix)); fix_s.append(np.std(fix))
        rec.append(np.mean([x[1] for x in a]))
        print(f"  T={T:6d}: active {np.mean(act):6.1f}  fixed {np.mean(fix):8.1f}  rec={rec[-1]:.0%}")

    x = np.array(HORIZONS)
    plt.figure(figsize=(9, 6))
    plt.errorbar(x, act_m, yerr=act_s, fmt="o-", color="#6C3483", lw=2.5,
                 label="Active (ours): horizon-independent")
    plt.errorbar(x, fix_m, yerr=fix_s, fmt="s--", color="#D35400", lw=2.5,
                 label="Fixed-rate $p_{ref}=0.1$: linear in $T$")
    plt.xlabel("Horizon $T$"); plt.ylabel("Total reference queries (audits)")
    plt.title("Audit budget vs horizon (both recover the optimum)")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/q3_audit_economy.png", dpi=200)
    print("  saved paper/figures/q3_audit_economy.png")
