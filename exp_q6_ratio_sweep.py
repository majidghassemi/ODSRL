"""
Item 5: Q6 real-panel RATIO sweep (GPU-free, cached). Panel composition is a
resampling knob: within a grader type the scores are identical (deterministic), so
sweeping the honest/sycophant count over the cached real-grader scores gives the
tipping-point figure with REAL graders instead of the synthetic bandit.

Each majority-trusting aggregator collapses once sycophants pass its tipping point;
ESA (cardinal V-trust) holds as long as >=1 honest grader remains and the sycophant
is cardinally detectable on V. Emits q6_ratio_sweep figure + JSON with provenance.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exp_rlvr_subjective as S
import exp_rlvr_real as RV
import provenance
from baselines import Mean, Median, DawidSkene, GAILProxy, RobustDRO_KL, RobustDRO_Wass, RRM

M7, M32 = RV.MODEL_ID, "Qwen/Qwen2.5-32B-Instruct"
CACHE = json.load(open(S.CACHE2))
V, _ = RV.load_problems(120, 1); RV.attach_candidates(V, np.random.default_rng(1))
N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))
ETA, M = RV.ETA, 10


def npair(model, nsys, shield):
    c, w = [], []
    for it in N:
        fr = "shield" if shield else it["user_pref_text"]
        c.append(CACHE[S._key(model, nsys, it["q"], it["cands"]["correct"], fr)])
        w.append(CACHE[S._key(model, nsys, it["q"], it["cands"]["wrong"], fr)])
    return np.array(c), np.array(w)


def vpair(model, vsys, framing):
    c = np.array([CACHE[S._key(model, vsys, p["q"], p["_cands"]["correct"],
                 framing(p))] for p in V])
    w = np.array([CACHE[S._key(model, vsys, p["q"], p["_cands"]["wrong"],
                 framing(p))] for p in V])
    return c, w


hNc, hNw = npair(M32, S.HONEST_N_SYS, True)
sNc, sNw = npair(M7, S.SYCO_N_SYS, False)
hVc, hVw = vpair(M32, RV.HONEST_SYS, lambda p: "natural")            # honest shielded on V
sVc, sVw = vpair(M7, RV.SYCO_SYS, lambda p: p["_wrong"])             # syco V (injected/lenient, cached)


def panel(nh):
    ns = M - nh
    return (np.stack([hNc]*nh + [sNc]*ns, 1), np.stack([hNw]*nh + [sNw]*ns, 1),
            np.stack([hVc]*nh + [sVc]*ns, 1), np.stack([hVw]*nh + [sVw]*ns, 1))


def esa_trust(Vc, Vw):
    w = np.ones(Vc.shape[1]) / Vc.shape[1]
    for pi in range(len(V)):
        for y, z in [(Vc[pi], 1.), (Vw[pi], 0.)]:
            w = w * np.exp(-ETA * np.abs(y / 10 - z)); w /= w.sum()
    return w


AGG = [("Mean", Mean()), ("Median", Median()), ("Dawid-Skene", DawidSkene(M)),
       ("KL-DRO", RobustDRO_KL()), ("Wass-DRO", RobustDRO_Wass()), ("RRM", RRM())]
fracs = [i / M for i in range(M + 1)]          # sycophant fraction 0..1
curves = {name: [] for name, _ in AGG}; curves["ESA"] = []
for f in fracs:
    ns = int(round(f * M)); nh = M - ns
    if nh == 0:                                # all syco: honest signal absent
        for name, _ in AGG: curves[name].append(np.nan)
        curves["ESA"].append(np.nan); continue
    Nc, Nw, Vc, Vw = panel(nh)
    for name, b in AGG:
        curves[name].append(float(np.mean([b.process(Nc[i]) > b.process(Nw[i]) for i in range(len(N))])))
    w = esa_trust(Vc, Vw)
    curves["ESA"].append(float(np.mean((Nc @ w) > (Nw @ w))))

print("syco_frac  " + "  ".join(f"{n[:6]:6s}" for n, _ in AGG) + "  ESA")
for i, f in enumerate(fracs):
    print(f"  {f:.1f}     " + "  ".join(f"{curves[n][i]*100:5.0f}%" if not np.isnan(curves[n][i]) else "  n/a " for n, _ in AGG)
          + f"  {curves['ESA'][i]*100:5.0f}%" if not np.isnan(curves['ESA'][i]) else "   n/a")

plt.figure(figsize=(9, 5.5))
for name, _ in AGG:
    plt.plot(fracs, curves[name], "o--", lw=1.5, alpha=0.8, label=name)
plt.plot(fracs, curves["ESA"], "o-", color="#6C3483", lw=3, label="ESA (ours)")
plt.axhline(0.5, color="grey", ls=":", lw=1)
plt.xlabel("sycophant fraction of the grader panel (M=10)")
plt.ylabel("N-recovery (TruthfulQA)")
plt.ylim(-0.05, 1.05); plt.grid(alpha=0.25); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig("paper/figures/q6_ratio_sweep_real.png", dpi=200)

out = {"fracs": fracs, "curves": curves,
       "provenance": provenance.stamp(n_v=len(V), n_n=len(N),
                                      note="panel composition swept over cached real-grader scores")}
json.dump(out, open("q6_ratio_sweep_results.json", "w"), indent=2)
print("\nsaved paper/figures/q6_ratio_sweep_real.png + q6_ratio_sweep_results.json")
