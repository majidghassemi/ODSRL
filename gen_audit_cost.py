"""
Data generator for Fig. G (q3_audit_cost). NO PLOTTING HERE.

Reuses the audit-economy experiment from exp_audit_economy.py (same env, same
p_ref=0.1, same seeds). The horizon grid is extended DOWN to T=100 so the log x-axis
spans three decades; the upper end stays at T=80000 so the fixed-rate endpoint remains
the ~8000 queries reported in CHANGELOG.md.

Writes audit_cost_results.json: per-horizon mean/SD reference-query counts for the
active and fixed-rate auditors, plus the active auditor's recovery rate.
"""
import json
import numpy as np

from exp_audit_economy import audits, SEEDS

# log-spaced, ~3 decades, retaining the published 80k endpoint
HORIZONS = [100, 320, 1000, 3200, 10000, 32000, 80000]

if __name__ == "__main__":
    print(f"Fig G data: {len(HORIZONS)} horizons x {SEEDS} seeds x 2 audit modes",
          flush=True)

    out = {"horizons": HORIZONS, "seeds": SEEDS, "p_ref": 0.1,
           "active": {"mean": [], "std": []},
           "fixed": {"mean": [], "std": []},
           "active_recovery": []}

    for T in HORIZONS:
        a = [audits("active", T, s) for s in range(SEEDS)]
        f = [audits("fixed", T, s) for s in range(SEEDS)]
        act = [x[0] for x in a]
        fix = [x[0] for x in f]
        out["active"]["mean"].append(float(np.mean(act)))
        out["active"]["std"].append(float(np.std(act)))
        out["fixed"]["mean"].append(float(np.mean(fix)))
        out["fixed"]["std"].append(float(np.std(fix)))
        out["active_recovery"].append(float(np.mean([x[1] for x in a])))
        print(f"  T={T:6d}: active {np.mean(act):7.1f}  fixed {np.mean(fix):8.1f}"
              f"  recovery={out['active_recovery'][-1]:.0%}", flush=True)

    # asymptotic active budget: the plateau the figure annotates
    out["active_asymptote"] = out["active"]["mean"][-1]
    json.dump(out, open("audit_cost_results.json", "w"), indent=2)
    print(f"saved audit_cost_results.json (active asymptote {out['active_asymptote']:.0f})")
