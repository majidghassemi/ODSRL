"""
v2 rewrite of the v1 bandit regret comparison (run_bandit.py).

v1 ran at 100% bias and claimed ESA achieves sublinear regret -- the exact
over-claim the coverage-boundary paper retracts. v2 theory:
  * With an honest minority AND coverage, per-region active-audit ESA concentrates
    trust on the honest sources and gets sublinear latent regret; majority-trusting
    aggregators (Mean/UCB, Dawid-Skene) track the corrupted majority -> LINEAR regret.
  * At 100% HOMOGENEOUS bias there is no honest minority, so pure trust-auditing
    cannot recover (Cor. "internal-only impossibility"). The ABSOLUTE-DISTRUST
    fail-safe (tau) detects that every source in the region is bad and falls back to
    the reference -- and even that only works where the arm is COVERED.

So this figure has two panels: (A) 80% bias, covered -> ESA wins; (B) 100% bias,
covered -> pure ESA fails, ESA+fail-safe recovers via the reference. Mean and
Dawid-Skene lose in both.

Uses the v2 stack: esa.ESA (per-region trust, active audit, fail-safe) via
runners.run_esa, social_bandit.SocialBanditEnv (coverage set), baselines.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from social_bandit import SocialBanditEnv
from runners import run_esa, run_baseline
from baselines import Mean, DawidSkene

K, M = 5, 10
OPT, BIAS_ARM, GAP, BIAS_MAG = 0, 1, 0.5, 1.0
STEPS, SEEDS = 5000, 25

COL = {
    'mean': '#D35400', 'dawid': '#27AE60',
    'esa': '#6C3483', 'esa_fs': '#1F618D',
}


def env_for(seed, sycophant_ratio, coverage):
    return SocialBanditEnv(
        k=K, M=M, optimal_arm=OPT, bias_arm=BIAS_ARM, arm_gap=GAP,
        bias_mag=BIAS_MAG, sycophant_ratio=sycophant_ratio, coverage=coverage,
        seed=seed)


def curve(kind, seed, sycophant_ratio, coverage, tau=None):
    env = env_for(seed, sycophant_ratio, coverage)
    if kind == 'mean':
        return run_baseline(env, Mean(), STEPS)['regret']
    if kind == 'dawid':
        return run_baseline(env, DawidSkene(M), STEPS)['regret']
    if kind == 'esa':
        return run_esa(env, STEPS, audit_mode='active', tau=tau,
                       trust_scope='region', seed=seed)['regret']
    raise ValueError(kind)


def bands(specs, sycophant_ratio, coverage):
    """specs: list of (key, kind, tau). Returns key -> (mean, std) over seeds."""
    out = {}
    for key, kind, tau in specs:
        R = np.stack([curve(kind, s, sycophant_ratio, coverage, tau)
                      for s in range(SEEDS)])
        out[key] = (R.mean(0), R.std(0))
    return out


def draw(ax, res, labels, title):
    x = np.arange(STEPS)
    for key in labels:
        m, sd = res[key]
        lw = 3 if key.startswith('esa') else 2
        ls = '-' if key.startswith('esa') else '--'
        ax.plot(x, m, color=COL[key], lw=lw, ls=ls, label=labels[key])
        ax.fill_between(x, m - sd, m + sd, color=COL[key], alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("cumulative latent regret")
    ax.set_title(title); ax.legend(loc='upper left'); ax.grid(alpha=0.2)


if __name__ == "__main__":
    print(f"v2 bandit regret: k={K} M={M} bias_arm={BIAS_ARM} "
          f"gap={GAP} mag={BIAS_MAG}, {SEEDS} seeds x {STEPS} steps")

    # Panel A: 80% biased, bias arm covered -> ESA recovers
    resA = bands([('mean', 'mean', None),
                  ('dawid', 'dawid', None),
                  ('esa', 'esa', None)], sycophant_ratio=0.8, coverage='full')
    # Panel B: 100% homogeneous bias, covered -> pure ESA fails, fail-safe recovers
    resB = bands([('mean', 'mean', None),
                  ('dawid', 'dawid', None),
                  ('esa', 'esa', None),
                  ('esa_fs', 'esa', 0.5)], sycophant_ratio=1.0, coverage='full')

    def final(res, key):
        m, sd = res[key]
        return m[-1], sd[-1]
    print("  --- 80% biased, covered ---")
    for k in ['mean', 'dawid', 'esa']:
        print(f"    {k:8s} final regret {final(resA,k)[0]:8.1f} ± {final(resA,k)[1]:5.1f}")
    print("  --- 100% homogeneous, covered ---")
    for k in ['mean', 'dawid', 'esa', 'esa_fs']:
        print(f"    {k:8s} final regret {final(resB,k)[0]:8.1f} ± {final(resB,k)[1]:5.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    draw(ax[0], resA,
         {'mean': 'Standard UCB (Mean)', 'dawid': 'Dawid-Skene',
          'esa': 'ESA active-audit (ours)'},
         "(A) 80% biased, bias arm covered\nhonest minority + coverage → ESA wins")
    draw(ax[1], resB,
         {'mean': 'Standard UCB (Mean)', 'dawid': 'Dawid-Skene',
          'esa': 'ESA, no fail-safe', 'esa_fs': 'ESA + absolute-distrust fail-safe'},
         "(B) 100% homogeneous bias, covered\nno honest minority → pure trust fails, fail-safe recovers")
    plt.suptitle("Adversarial-bandit regret: v1 comparison corrected for the "
                 "coverage-boundary theory (v2)", fontsize=12)
    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    out = "paper/figures/q7_bandit_regret_v2.png"
    plt.savefig(out, dpi=200)
    print(f"  saved {out}")
