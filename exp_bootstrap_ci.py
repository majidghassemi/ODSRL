"""
Bootstrap 95% CIs over N items for the real-grader headline numbers (GPU-free).
Adds intervals to Q5 (ESA), Q6-real (every aggregator), and the ordinal-vs-cardinal
table. AAAI reproducibility checklist item. Emits bootstrap_ci.json.
"""
import json
import numpy as np
import exp_rlvr_subjective as S
import exp_rlvr_real as RV
from baselines import Mean, Median, DawidSkene, GAILProxy, RobustDRO_KL, RobustDRO_Wass, RRM

RNG = np.random.default_rng(0)
M7, M32 = RV.MODEL_ID, "Qwen/Qwen2.5-32B-Instruct"
CACHE = json.load(open(S.CACHE2))
V, _ = RV.load_problems(120, 1); RV.attach_candidates(V, np.random.default_rng(1))
N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))


def npair(model, nsys, shield):
    c, w = [], []
    for it in N:
        fr = "shield" if shield else it["user_pref_text"]
        c.append(CACHE[S._key(model, nsys, it["q"], it["cands"]["correct"], fr)])
        w.append(CACHE[S._key(model, nsys, it["q"], it["cands"]["wrong"], fr)])
    return np.array(c), np.array(w)


def ci(per_item, B=5000):
    """95% bootstrap CI of the mean of a per-item 0/1 outcome vector."""
    per_item = np.asarray(per_item, float)
    n = len(per_item)
    means = per_item[RNG.integers(0, n, size=(B, n))].mean(1)
    return float(per_item.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# main-config N panel (2 honest-32B + 8 syco-7B) and frozen ESA trust
hNc, hNw = npair(M32, S.HONEST_N_SYS, True)
sNc, sNw = npair(M7, S.SYCO_N_SYS, False)
Nc = np.stack([hNc]*2 + [sNc]*8, 1); Nw = np.stack([hNw]*2 + [sNw]*8, 1)
w_esa = np.array(json.load(open("rlvr_subjective_results.json"))["sweep"][-1]["trust"])

out = {"n_items": len(N), "bootstrap": 5000, "q5_q6": {}, "ordinal": {}}
print(f"=== bootstrap 95% CIs (N={len(N)} items, 5000 resamples) ===\n")
print(f"{'method':30s} recovery [95% CI]")

methods = [("ESA (frozen V-trust)", lambda yc, yw: (yc @ w_esa) > (yw @ w_esa))]
for name, b in [("Mean", Mean()), ("Median", Median()), ("Dawid-Skene", DawidSkene(10)),
                ("GAIL (mimic majority)", GAILProxy()), ("KL-DRO (KLDPO-analogue)", RobustDRO_KL()),
                ("Wass-DRO (WDPO-analogue)", RobustDRO_Wass()), ("RRM", RRM())]:
    methods.append((name, (lambda bb: (lambda yc, yw: bb.process(yc) > bb.process(yw)))(b)))
for name, fn in methods:
    per = [fn(Nc[i], Nw[i]) for i in range(len(N))]
    m, lo, hi = ci(per)
    out["q5_q6"][name] = {"recovery": m, "ci95": [lo, hi]}
    print(f"  {name:30s} {m:.0%}  [{lo:.0%}, {hi:.0%}]")

# ordinal vs cardinal (natural-coupling 32B panel): per-item recovery under each trust
print("\nordinal-vs-cardinal (natural 32B panel):")
def vmarg(vsys):
    c = np.array([CACHE[S._key(M32, vsys, p["q"], p["_cands"]["correct"], "natural")] for p in V])
    w = np.array([CACHE[S._key(M32, vsys, p["q"], p["_cands"]["wrong"], "natural")] for p in V])
    return c, w
hVc, hVw = vmarg(RV.HONEST_SYS); sVc, sVw = vmarg(RV.SYCO_SYS)
hNc2, hNw2 = npair(M32, S.HONEST_N_SYS, True); sNc2, sNw2 = npair(M32, S.SYCO_N_SYS, False)
Vc = np.stack([hVc]*2 + [sVc]*8, 1); Vw = np.stack([hVw]*2 + [sVw]*8, 1)
Nc2 = np.stack([hNc2]*2 + [sNc2]*8, 1); Nw2 = np.stack([hNw2]*2 + [sNw2]*8, 1)
ETA = RV.ETA
def card_w():
    w = np.ones(10)/10
    for pi in range(len(V)):
        for y, z in [(Vc[pi], 1.), (Vw[pi], 0.)]:
            w = w*np.exp(-ETA*np.abs(y/10-z)); w /= w.sum()
    return w
def ord_w():
    w = np.ones(10)/10
    for pi in range(len(V)):
        w = w*np.exp(-ETA*(Vw[pi] >= Vc[pi])); w /= w.sum()
    return w
wc, wo = card_w(), ord_w()
for name, w in [("cardinal", wc), ("ordinal", wo)]:
    per = [(Nc2[i] @ w) > (Nw2[i] @ w) for i in range(len(N))]
    m, lo, hi = ci(per)
    out["ordinal"][name] = {"recovery": m, "ci95": [lo, hi]}
    print(f"  {name:12s} recovery {m:.0%}  [{lo:.0%}, {hi:.0%}]")

import provenance
out["provenance"] = provenance.stamp(n_n=len(N), n_v=len(V))
json.dump(out, open("bootstrap_ci.json", "w"), indent=2)
print("\nsaved bootstrap_ci.json")
