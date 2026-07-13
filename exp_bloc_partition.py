"""
Q4 -- correlation routing (Sec. B6).
(a) Oracle-less covariance clustering recovers the evaluator PARTITION even when
    colluders are the majority; it CANNOT label which bloc is honest (both clusters
    are internally coherent). Labeling needs the audit (Cor. internal-only).
(b) The drift alarm fires when a new colluding bloc forms mid-run.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from esa import ESA

def make_reports(rng, M, K, n_coll, bias_arms, bias_mag, sigma, T):
    Rstar = rng.normal(0, 1, K)
    b = np.zeros((M, K)); b[:n_coll][:, bias_arms] = bias_mag   # shared collusion bias
    Y = np.empty((T, M))
    for t in range(T):
        a = rng.integers(K)
        Y[t] = Rstar[a] + b[:, a] + rng.normal(0, sigma, M)
    truth = (np.arange(M) < n_coll).astype(int)
    return Y, truth

def cluster_acc(Y, truth, M):
    ag = ESA(1, M); ag._report_hist = list(Y)
    lab = ag.detect_blocs(n_blocs=2)
    return max((lab == truth).mean(), (lab == 1 - truth).mean())

if __name__ == "__main__":
    print("Q4: bloc partition + drift alarm")
    rng = np.random.default_rng(0)
    M, K = 20, 12
    print("  partition accuracy (oracle-less) vs colluding-majority size:")
    accs = []
    for ratio in [0.3, 0.5, 0.6, 0.8]:
        n = int(M * ratio)
        a = np.mean([cluster_acc(*make_reports(np.random.default_rng(s), M, K, n,
                     [2, 5, 9], 2.5, 0.3, 400), M) for s in range(10)])
        accs.append(a); print(f"    colluders={ratio:.0%}: accuracy={a:.3f}")

    # (b) drift alarm: phase 1 clean-ish, phase 2 a new bloc coordinates
    ag = ESA(1, M)
    alarms = []
    rng = np.random.default_rng(1)
    Rstar = rng.normal(0, 1, K)
    for t in range(1600):
        a = rng.integers(K)
        base = Rstar[a]
        y = base + rng.normal(0, 0.3, M)
        if t >= 800:
            shared = rng.normal(0, 1.0)         # colluders share a common component
            y[:12] = base + 1.5 * shared + rng.normal(0, 0.3, 12)
        ag._report_hist.append(y)
        alarms.append(1 if (t > 400 and t % 25 == 0
                            and ag.drift_alarm(window=200, thresh=1.15)) else 0)
    fired = [i for i, v in enumerate(alarms) if v]
    print(f"  drift alarm fired at steps: {fired[:6]}{' ...' if len(fired) > 6 else ''} "
          f"(bloc formed at 800)")

    plt.figure(figsize=(9, 4.5))
    plt.subplot(1, 2, 1)
    plt.bar([f"{int(r*100)}%" for r in [30, 50, 60, 80]], accs, color="#6C3483")
    plt.axhline(0.5, color="grey", ls="--", lw=1)
    plt.ylim(0, 1.05); plt.ylabel("partition accuracy")
    plt.xlabel("colluding majority"); plt.title("Partition (no oracle)")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(alarms)), np.array(alarms), color="#C0392B")
    plt.axvline(800, color="k", ls="--", lw=1, label="bloc forms")
    plt.xlabel("step"); plt.title("Drift alarm"); plt.legend()
    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/q4_bloc_partition.png", dpi=200)
    print("  saved paper/figures/q4_bloc_partition.png")
