"""
Q6 on the REAL grader panel (GPU-free; uses cached Q5 scores).

Converts Q6 from synthetic bandit to real LLM graders: run the aggregation baselines
(Mean, Median, Dawid-Skene, and the WDPO/KLDPO/RRM scalar analogues from baselines.py)
on the cached hetero panel (2 honest Qwen2.5-32B + 8 sycophant Qwen2.5-7B) grading the
TruthfulQA N region, and compare to ESA with frozen V-trust. Every majority-trusting
aggregator tracks the sycophantic 8/10 majority and fails; ESA, auditing the source
on the verifiable region, recovers.

NOTE (Related Work, not Experiments): full WDPO/KLDPO/RRM are preference-optimization
TRAINING methods for a different corruption model (sparse corruption, trustworthy
bulk); they are ordinal, and the ordinal-inversion result (exp_rlvr_ordinal_check.py)
shows that modality hides the dispositional bias studied here. These are labeled
scalar-reward analogues, run on the same reports, not the original methods.
"""
import json
import numpy as np
import exp_rlvr_subjective as S
import exp_rlvr_real as RV
from baselines import Mean, Median, DawidSkene, GAILProxy, RobustDRO_KL, RobustDRO_Wass, RRM

M7 = RV.MODEL_ID                  # 7B sycophants
M32 = "Qwen/Qwen2.5-32B-Instruct"  # 32B honest
CACHE = json.load(open(S.CACHE2))
N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))


def nsc(model, n_sys, shield):
    c, w = [], []
    for it in N:
        frame = "shield" if shield else it["user_pref_text"]
        c.append(CACHE[S._key(model, n_sys, it["q"], it["cands"]["correct"], frame)])
        w.append(CACHE[S._key(model, n_sys, it["q"], it["cands"]["wrong"], frame)])
    return np.array(c), np.array(w)


# main config panel: graders 0,1 = honest-32B; 2..9 = syco-7B
hNc, hNw = nsc(M32, S.HONEST_N_SYS, True)
sNc, sNw = nsc(M7, S.SYCO_N_SYS, False)
Nc = np.stack([hNc]*2 + [sNc]*8, 1)     # [nN, 10]
Nw = np.stack([hNw]*2 + [sNw]*8, 1)

# ESA frozen V-trust from the main Q5 result (couple=1)
res = json.load(open("rlvr_subjective_results.json"))
w_esa = np.array(res["sweep"][-1]["trust"])


def recovery(pick_fn):
    return float(np.mean([pick_fn(Nc[i], Nw[i]) for i in range(len(N))]))


print("=== Q6 on the REAL grader panel (2 honest-32B + 8 syco-7B, TruthfulQA N) ===")
print(f"panel: 8/10 sycophantic majority; N={len(N)} items\n")
print(f"{'method':28s} N-recovery")
recs = {}
for name, b in [("Mean (Dogma-4)", Mean()), ("Median", Median()),
                ("Dawid-Skene", DawidSkene(10)), ("GAIL (mimic majority)", GAILProxy()),
                ("KL-DRO (KLDPO-analogue)", RobustDRO_KL()),
                ("Wass-DRO (WDPO-analogue)", RobustDRO_Wass()),
                ("RRM (robust reward)", RRM())]:
    recs[name] = recovery(lambda yc, yw: b.process(yc) > b.process(yw))
    print(f"  {name:28s} {recs[name]:.0%}")
recs["ESA (frozen V-trust, ours)"] = recovery(lambda yc, yw: (yc @ w_esa) > (yw @ w_esa))
print(f"  {'ESA (frozen V-trust, ours)':28s} {recs['ESA (frozen V-trust, ours)']:.0%}")

# caption text + KL-DRO note carried into the emitted JSON so they survive into the paper
import json as _json, provenance
CAPTION = ("Q6 aggregation baselines on the REAL grader panel (2 honest Qwen2.5-32B + "
           "8 sycophant Qwen2.5-7B) over TruthfulQA. SINGLE FIXED RATIO (8/10); the "
           "tipping-point ratio sweep is exp_q6_ratio_sweep.py (also real graders). "
           "Every majority-trusting aggregator collapses to the sycophantic majority; "
           "ESA, auditing the source on the verifiable region, recovers.")
KL_DRO_NOTE = ("KL-DRO's partial 38% is not noise: entropic risk exp-tilts toward LOW "
               "scores, so it discounts the sycophant's lenient score inflation. It is "
               "exploiting the same CARDINAL channel ESA uses, just without a reference "
               "-- which is why it beats every other baseline and still fails. A "
               "corroborating data point for the cardinal-elicitation claim.")
_json.dump({"recovery": recs, "single_ratio": "8/10 fixed", "caption": CAPTION,
            "kl_dro_note": KL_DRO_NOTE, "provenance": provenance.stamp(n_n=len(N))},
           open("q6_real_panel_results.json", "w"), indent=2)
print("\nsaved q6_real_panel_results.json  (caption + KL-DRO note carried for the paper)")
