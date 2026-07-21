"""
Data generator for Fig. A (q2_coverage_boundary). NO PLOTTING HERE.

Reuses the exact experiment configuration from exp_coverage_boundary.py (same env,
same seeds, same horizon) and dumps the per-seed cumulative latent regret matrices so
the figure can be re-rendered without re-running the bandit.

Three conditions:
  esa_covered   : ESA, divergence arm inside the coverage set  -> recovers
  esa_uncovered : ESA, divergence arm OUTSIDE coverage         -> tracks the mean
  mean          : naive mean aggregation (captured by the majority)

Writes coverage_boundary_curves.npz ([seeds, STEPS] per condition) and
coverage_boundary_summary.json (recovery rates).
"""
import json
import numpy as np

from exp_coverage_boundary import curves, COV_ON, COV_OFF, STEPS, SEEDS

if __name__ == "__main__":
    print(f"Fig A data: 3 conditions x {SEEDS} seeds x {STEPS} steps", flush=True)

    esa_on, rec_on = curves(COV_ON, "esa")
    print(f"  esa_covered   recovery={rec_on:.2f}", flush=True)
    esa_off, rec_off = curves(COV_OFF, "esa")
    print(f"  esa_uncovered recovery={rec_off:.2f}", flush=True)
    base, rec_b = curves(COV_ON, "mean")
    print(f"  mean          recovery={rec_b:.2f}", flush=True)

    np.savez_compressed("coverage_boundary_curves.npz",
                        esa_covered=esa_on.astype(np.float32),
                        esa_uncovered=esa_off.astype(np.float32),
                        mean=base.astype(np.float32))
    json.dump({"steps": STEPS, "seeds": SEEDS,
               "recovery": {"esa_covered": float(rec_on),
                            "esa_uncovered": float(rec_off),
                            "mean": float(rec_b)}},
              open("coverage_boundary_summary.json", "w"), indent=2)
    print("saved coverage_boundary_curves.npz and coverage_boundary_summary.json")
