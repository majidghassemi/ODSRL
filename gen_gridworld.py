"""
Data generator for Fig. B (q1_mdp_gridworld). NO PLOTTING HERE.

Reuses the exact experiment from exp_mdp_gridworld.py (same evaluators, same seeds,
same episode count) and dumps the per-seed P(visit Candy) traces for the four methods
so the figure can be re-rendered without re-running the gridworld.

Writes gridworld_candy_curves.npz ([seeds, EPISODES] per method, smoothed at write
time is NOT done -- raw traces are stored; the figure does the smoothing) and
gridworld_summary.json (final P(candy) over the last 200 episodes).
"""
import json
import numpy as np

import exp_mdp_gridworld as G

KINDS = ("mean", "median", "global_esa", "region_esa")


def _worker(job):
    kind, seed = job
    cv, _, _ = G.run_single_seed(seed, kind)
    return kind, seed, cv.astype(np.float32)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=G.SEEDS,
                    help="global-trust ESA is bimodal across seeds, so its trapped "
                         "fraction needs many more than the default 15 to be stable")
    args = ap.parse_args()

    jobs = [(k, s) for k in KINDS for s in range(args.seeds)]
    print(f"Fig B data: {len(jobs)} runs ({len(KINDS)} methods x {args.seeds} seeds)"
          f" x {G.EPISODES} episodes", flush=True)

    with mp.get_context("fork").Pool(processes=min(len(jobs), 30)) as pool:
        results = pool.map(_worker, jobs)

    out, summary = {}, {}
    for k in KINDS:
        M = np.stack([cv for kind, _, cv in sorted(results, key=lambda t: t[1])
                      if kind == k])                       # [seeds, EPISODES]
        out[k] = M
        final = M[:, -200:].mean(1)                        # P(candy), last 200 eps
        summary[k] = {"final_mean": float(final.mean()), "final_std": float(final.std()),
                      "seeds": int(M.shape[0]), "episodes": int(M.shape[1])}
        print(f"  {k:11s} P(candy last200) = {final.mean():.3f} +/- {final.std():.3f}",
              flush=True)

    np.savez_compressed("gridworld_candy_curves.npz", **out)
    json.dump(summary, open("gridworld_summary.json", "w"), indent=2)
    print("saved gridworld_candy_curves.npz and gridworld_summary.json")
