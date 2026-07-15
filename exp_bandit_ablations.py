"""
v2 rewrite of run_ablations.py -- three robustness ablations on the bandit, using
the v2 stack (esa.ESA per-region active-audit + fail-safe via runners.run_esa,
social_bandit.SocialBanditEnv with a coverage set).

  A. bias ratio     -- how ESA regret scales with the fraction of biased evaluators.
                       v2 message: flat 50-90%, and the ABSOLUTE-DISTRUST fail-safe
                       (tau) keeps it bounded at 95-100% where there is no honest
                       minority (the v1 claim of unconditional robustness, corrected).
  B. eta            -- sensitivity to the trust learning rate.
  C. internal noise -- robustness to an imperfect reference (sigma_ref), i.e. the
                       spot-check/verifier is noisy. Coverage-boundary theory: recovery
                       degrades gracefully as long as the reference still covers the
                       divergence arm.

Cumulative latent regret, mean +/- std over seeds. Figures -> paper/figures/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from social_bandit import SocialBanditEnv
from runners import run_esa

K, M = 5, 20
OPT, BIAS_ARM, GAP, BIAS_MAG = 0, 1, 0.5, 1.0
STEPS, SEEDS = 5000, 15
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)


def esa_regret(sycophant_ratio, eta, sigma_ref, tau, seed):
    env = SocialBanditEnv(k=K, M=M, optimal_arm=OPT, bias_arm=BIAS_ARM, arm_gap=GAP,
                          bias_mag=BIAS_MAG, sycophant_ratio=sycophant_ratio,
                          coverage="full", sigma_ref=sigma_ref, seed=seed)
    return run_esa(env, STEPS, eta=eta, audit_mode="active", tau=tau,
                   trust_scope="region", seed=seed)["regret"]


def band(ax, ratios_or_vals, curve_fn, labels, colors):
    x = np.arange(STEPS)
    for v, lab, c in zip(ratios_or_vals, labels, colors):
        R = np.stack([curve_fn(v, s) for s in range(SEEDS)])
        m, sd = R.mean(0), R.std(0)
        ax.plot(x, m, color=c, lw=2, label=lab)
        ax.fill_between(x, m - sd, m + sd, color=c, alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("cumulative latent regret")
    ax.legend(); ax.grid(alpha=0.25)


if __name__ == "__main__":
    # A. bias ratio (fail-safe on, so 100% is covered by absolute-distrust)
    print("ablation A: bias ratio")
    ratios = [0.5, 0.7, 0.9, 0.95, 1.0]
    colors = ["#27AE60", "#2980B9", "#8E44AD", "#E67E22", "#C0392B"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    band(ax, ratios,
         lambda r, s: esa_regret(r, 0.5, 0.1, 0.5, s),
         [f"{int(r*100)}% biased" for r in ratios], colors)
    ax.set_title("A. Robustness to fraction of biased evaluators\n"
                 "v2 ESA (active audit + absolute-distrust fail-safe), covered")
    fig.tight_layout(); fig.savefig(f"{OUT}/q8a_ablation_bias_ratio.png", dpi=200)
    for r in ratios:
        R = np.stack([esa_regret(r, 0.5, 0.1, 0.5, s) for s in range(SEEDS)])
        print(f"  {int(r*100)}% biased: final regret {R.mean(0)[-1]:.1f} ± {R.std(0)[-1]:.1f}")

    # B. eta sensitivity (fixed 80% biased, honest minority present)
    print("ablation B: eta")
    etas = [0.1, 0.5, 2.0]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    band(ax, etas,
         lambda e, s: esa_regret(0.8, e, 0.1, None, s),
         [rf"$\eta={e}$" for e in etas], ["#F39C12", "#6C3483", "#16A085"])
    ax.set_title(r"B. Sensitivity to trust update rate $\eta$ (80% biased, covered)")
    fig.tight_layout(); fig.savefig(f"{OUT}/q8b_ablation_eta.png", dpi=200)
    for e in etas:
        R = np.stack([esa_regret(0.8, e, 0.1, None, s) for s in range(SEEDS)])
        print(f"  eta={e}: final regret {R.mean(0)[-1]:.1f} ± {R.std(0)[-1]:.1f}")

    # C. internal (reference) noise
    print("ablation C: reference noise")
    noises = [0.0, 0.1, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    band(ax, noises,
         lambda sg, s: esa_regret(0.8, 0.5, sg, None, s),
         [rf"ref $\sigma={sg}$" for sg in noises], ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"])
    ax.set_title("C. Robustness to imperfect reference / spot-check noise\n(80% biased, covered)")
    fig.tight_layout(); fig.savefig(f"{OUT}/q8c_ablation_ref_noise.png", dpi=200)
    for sg in noises:
        R = np.stack([esa_regret(0.8, 0.5, sg, None, s) for s in range(SEEDS)])
        print(f"  sigma_ref={sg}: final regret {R.mean(0)[-1]:.1f} ± {R.std(0)[-1]:.1f}")
    print("saved q8a/q8b/q8c ablation figures ->", OUT)
