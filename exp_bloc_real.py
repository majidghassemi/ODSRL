"""
Item 7: run ESA's oracle-less bloc detection (esa.detect_blocs) on the REAL grader
panel's cached scores. If the 8-grader sycophant bloc separates from the 2 honest
graders on the consensus-centered residual covariance, Q4 gets a real-grader panel
too. If it does not, we report the noise regime and claim nothing. GPU-free.
"""
import json
import numpy as np
import exp_rlvr_subjective as S
import exp_rlvr_real as RV
import provenance
from esa import ESA

M7, M32 = RV.MODEL_ID, "Qwen/Qwen2.5-32B-Instruct"
CACHE = json.load(open(S.CACHE2))
N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))


def nscores(model, nsys, shield):
    c, w = [], []
    for it in N:
        fr = "shield" if shield else it["user_pref_text"]
        c.append(CACHE[S._key(model, nsys, it["q"], it["cands"]["correct"], fr)])
        w.append(CACHE[S._key(model, nsys, it["q"], it["cands"]["wrong"], fr)])
    return np.array(c), np.array(w)


hC, hW = nscores(M32, S.HONEST_N_SYS, True)      # honest graders 0,1
sC, sW = nscores(M7, S.SYCO_N_SYS, False)        # syco graders 2..9

# report matrix H[reports, M]: each report = 10 graders' scores on a candidate
# (both correct- and wrong-candidate reports), graders 0,1 honest / 2..9 syco.
rows = []
for i in range(len(N)):
    rows.append([hC[i], hC[i]] + [sC[i]] * 8)    # correct-candidate reports
    rows.append([hW[i], hW[i]] + [sW[i]] * 8)    # wrong-candidate reports
H = np.array(rows) + np.random.default_rng(0).normal(0, 1e-3, (len(rows), 10))  # break exact ties

ag = ESA(k=1, M=10, seed=0)
ag._report_hist = [h for h in H]
labels = ag.detect_blocs(n_blocs=2)

true = np.array([0, 0] + [1] * 8)                # honest=0, syco=1
acc = max((labels == true).mean(), (labels != true).mean())
honest_together = labels[0] == labels[1]
syco_together = len(set(labels[2:])) == 1
syco_vs_honest_split = labels[0] != labels[2]

print(f"detected labels: {labels.tolist()}")
print(f"true partition : {true.tolist()}  (0=honest, 1=syco)")
print(f"partition accuracy (label-flip invariant): {acc:.2f}")
print(f"honest graders together: {honest_together} | syco bloc together: {syco_together} "
      f"| honest vs syco split: {syco_vs_honest_split}")
verdict = ("syco bloc SEPARATES -> Q4 works on the real panel" if (acc >= 0.9 and syco_vs_honest_split)
           else "no clean separation -> report the noise regime, claim nothing")
print("->", verdict)

json.dump({"labels": labels.tolist(), "true": true.tolist(), "accuracy": float(acc),
           "honest_together": bool(honest_together), "syco_together": bool(syco_together),
           "verdict": verdict, "provenance": provenance.stamp(n_n=len(N))},
          open("bloc_real_results.json", "w"), indent=2)
print("saved bloc_real_results.json")
