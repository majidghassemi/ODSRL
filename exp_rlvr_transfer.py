"""
Q5 -- verifiable-subtask anchoring (RLVR), two-phase transfer.
Pipeline: learn evaluator trust on the verifiable region V (dense, exact, free
reference), FREEZE it, then deploy on the non-verifiable region N (no reference).

Coupling knob: a fraction `couple` of the biased coalition is visible on V (they
also lie on a V-arm, so the verifier catches them); the rest are honest-on-V but
biased-on-N (the "competent mathematician with an agenda on essays" -- the
Theorem-2 adversary). Transfer works when coupling is high and fails when
reliability decouples. Delta_drift ~ (1 - couple) * B.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from esa import ESA

K, M = 8, 10
V_ARMS = [0, 1, 2, 3]
N_ARMS = [4, 5, 6, 7]
OPT_N, SOC_N = 4, 5          # true optimum vs the N-arm sycophants push
V_AUDIT_ARM = 1
B, GAP = 1.0, 0.5
N_SYC = 8
TRAIN, DEPLOY, SEEDS = 3000, 3000, 20


def build_bias(couple):
    b = np.zeros((M, K))
    n_vis = int(round(couple * N_SYC))
    for m in range(N_SYC):
        b[m, SOC_N] = B                       # all push the suboptimal N-arm
        if m < n_vis:
            b[m, V_AUDIT_ARM] = B             # visible ones also lie on V -> caught
    return b


def means():
    mu = np.full(K, 0.5); mu[OPT_N] = 0.5 + GAP
    return mu


def n_recovery(couple):
    recs = []
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        b = build_bias(couple); mu = means()
        ag = ESA(K, M, trust_scope="global", audit_mode="fixed", p_ref=1.0, seed=s)
        # Phase 1: learn global trust on V (audit every pull; V is covered).
        for _ in range(TRAIN):
            a = V_ARMS[rng.integers(len(V_ARMS))]
            true_r = rng.normal(mu[a], 0.1)
            y = true_r + b[:, a] + rng.normal(0, 0.1, M)
            z = true_r + rng.normal(0, 0.01)
            loss = np.abs(y - z)
            ag.w *= np.exp(-ag.eta * loss); ag.w /= ag.w.sum(axis=-1, keepdims=True)
        w = ag.w.mean(0)
        # Phase 2: deploy on N with UCB, aggregate with frozen w (no reference).
        muN = np.zeros(len(N_ARMS)); Nn = np.zeros(len(N_ARMS))
        for t in range(DEPLOY):
            j = int(np.argmax(muN + np.sqrt(2 * np.log(t + 2) / (Nn + 1e-9))))
            a = N_ARMS[j]
            true_r = rng.normal(mu[a], 0.1)
            y = true_r + b[:, a] + rng.normal(0, 0.1, M)
            r_hat = float(np.dot(w, y))
            Nn[j] += 1; muN[j] += (r_hat - muN[j]) / Nn[j]
        recs.append(N_ARMS[int(np.argmax(muN))] == OPT_N)
    return np.mean(recs)


if __name__ == "__main__":
    print("Q5: RLVR transfer (train trust on V, deploy on N)")
    couples = [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]
    rec = []
    for c in couples:
        r = n_recovery(c); drift = (1 - c) * B; rec.append(r)
        print(f"  coupling={c:.2f}  (Delta_drift~{drift:.2f}): N-recovery={r:.0%}")
    plt.figure(figsize=(8, 5.5))
    plt.plot(couples, rec, "o-", color="#6C3483", lw=2.5)
    plt.axhline(0.5, color="grey", ls=":", lw=1)
    plt.xlabel("cross-region coupling (fraction of bias visible on V)")
    plt.ylabel("recovery on non-verifiable region N")
    plt.title("Trust transfers from verifiable to non-verifiable tasks\n"
              "iff reliability couples across regions (Prop. RLVR / Thm. 2 with C=V)")
    plt.ylim(-0.05, 1.05); plt.grid(alpha=0.25); plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    plt.savefig("paper/figures/q5_rlvr_transfer.png", dpi=200)
    print("  saved paper/figures/q5_rlvr_transfer.png")
