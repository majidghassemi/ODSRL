"""
Data generator for Figs. O/P (q4a, q4b correlation routing). NO PLOTTING HERE.

Reuses the exact experiment from exp_bloc_partition.py (same evaluators, same seeds,
same report model) and dumps both panels' data:

  a (q4a) oracle-less covariance clustering recovers the evaluator PARTITION even when
          the colluders are the majority. It CANNOT label which bloc is honest -- both
          clusters are internally coherent -- so labeling still needs the audit.
  b (q4b) the drift alarm fires when a new colluding bloc forms mid-run. Stored as the
          RAW STATISTIC (the leading-eigenvalue ratio of the residual covariance,
          recent window vs the preceding one) rather than the 0/1 alarm, so the figure
          can show how far past the threshold the statistic travels.

The statistic is recomputed here with the same formula ESA.drift_alarm uses, and each
checkpoint asserts that (ratio > thresh) equals the alarm ESA itself reports, so the
two can never drift apart silently.

Writes bloc_partition_results.json. Figs. O/P read that file and nothing else.

Run: python3 gen_bloc_partition.py --seeds 10
"""
import argparse
import json

import numpy as np

from esa import ESA

M, K = 20, 12
RATIOS = [0.3, 0.5, 0.6, 0.8]
BIAS_ARMS, BIAS_MAG, SIGMA, T_REPORTS = [2, 5, 9], 2.5, 0.3, 400

# panel b
STEPS, BLOC_FORMS, WINDOW, THRESH, CHECK_EVERY = 1600, 800, 200, 1.15, 25
N_COLLUDERS_B, SHARED_MAG = 12, 1.5


def make_reports(rng, n_coll):
    """T_REPORTS report vectors from M evaluators; the first n_coll share a bias."""
    Rstar = rng.normal(0, 1, K)
    b = np.zeros((M, K))
    b[:n_coll][:, BIAS_ARMS] = BIAS_MAG          # shared collusion bias
    Y = np.empty((T_REPORTS, M))
    for t in range(T_REPORTS):
        a = rng.integers(K)
        Y[t] = Rstar[a] + b[:, a] + rng.normal(0, SIGMA, M)
    truth = (np.arange(M) < n_coll).astype(int)
    return Y, truth


def cluster_acc(Y, truth):
    """Partition accuracy, up to a label swap (the method returns no bloc identity)."""
    ag = ESA(1, M)
    ag._report_hist = list(Y)
    lab = ag.detect_blocs(n_blocs=2)
    return max((lab == truth).mean(), (lab == 1 - truth).mean())


def lead_eig(block):
    """Leading eigenvalue of the residual correlation -- verbatim ESA.drift_alarm."""
    R = block - block.mean(axis=1, keepdims=True)   # strip per-step consensus
    C = np.nan_to_num(np.corrcoef(R.T))
    return float(np.linalg.eigvalsh(C)[-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    out = {"M": M, "K": K, "ratios": RATIOS, "seeds": args.seeds,
           "n_reports": T_REPORTS, "bias_mag": BIAS_MAG, "sigma": SIGMA,
           "partition": {"mean": [], "std": []},
           "drift": {"steps": [], "ratio": [], "alarm": [], "thresh": THRESH,
                     "bloc_forms": BLOC_FORMS, "window": WINDOW,
                     "n_colluders": N_COLLUDERS_B}}

    # --- (a) partition accuracy vs colluding-majority size ----------------------
    print(f"panel a: partition accuracy, {args.seeds} seeds/ratio", flush=True)
    for ratio in RATIOS:
        n = int(M * ratio)
        accs = [cluster_acc(*make_reports(np.random.default_rng(s), n))
                for s in range(args.seeds)]
        out["partition"]["mean"].append(float(np.mean(accs)))
        out["partition"]["std"].append(float(np.std(accs)))
        print(f"  colluders={ratio:.0%}: accuracy={np.mean(accs):.3f} "
              f"+/- {np.std(accs):.3f}", flush=True)

    # --- (b) drift alarm when a new bloc forms mid-run --------------------------
    print(f"panel b: drift alarm (bloc forms at step {BLOC_FORMS})", flush=True)
    ag = ESA(1, M)
    rng = np.random.default_rng(1)
    Rstar = rng.normal(0, 1, K)
    for t in range(STEPS):
        base = Rstar[rng.integers(K)]
        y = base + rng.normal(0, SIGMA, M)
        if t >= BLOC_FORMS:
            shared = rng.normal(0, 1.0)          # colluders share a common component
            y[:N_COLLUDERS_B] = (base + SHARED_MAG * shared
                                 + rng.normal(0, SIGMA, N_COLLUDERS_B))
        ag._report_hist.append(y)

        if t > 2 * WINDOW and t % CHECK_EVERY == 0:
            H = np.array(ag._report_hist)
            recent, prev = lead_eig(H[-WINDOW:]), lead_eig(H[-2 * WINDOW:-WINDOW])
            ratio = recent / prev if prev > 1e-9 else float("nan")
            alarm = bool(ag.drift_alarm(window=WINDOW, thresh=THRESH))
            assert alarm == (ratio > THRESH), (t, ratio, alarm)
            out["drift"]["steps"].append(t)
            out["drift"]["ratio"].append(float(ratio))
            out["drift"]["alarm"].append(int(alarm))

    fired = [s for s, a in zip(out["drift"]["steps"], out["drift"]["alarm"]) if a]
    out["drift"]["first_fire"] = fired[0] if fired else None
    print(f"  alarm fired at steps: {fired[:6]}{' ...' if len(fired) > 6 else ''}",
          flush=True)

    json.dump(out, open("bloc_partition_results.json", "w"), indent=2)
    print("saved bloc_partition_results.json")
