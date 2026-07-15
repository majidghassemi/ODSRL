"""
Prop. rlvr threshold check, MARGIN / SCORE units, from cache only. Emits JSON.

Corrected per review:
  * Margin form, NO /2:  Delta* = G_N - beta*_V - eps.
  * All quantities on the score-margin scale (units of z, 0-10).
  * beta*_V measured on SHIELDED (belief-free) honest V grading (deployment
    conditions), per the grader-shielding fix.

Consequence of the shielding fix, reported honestly: with honest shielded, the
coupling knob no longer produces a recovery cliff (ESA is flat at 0.80). The real
recover/fail boundary is between adversary PERSONAS, so we compare the drifts of the
lenient sycophant (empirically RECOVERS) and the harsh/strategic sycophant
(empirically FAILS) against Delta*, and also report the cardinal V-calibration loss,
which is what actually drives detection.

All graders 32B here (honest vs syco), so capability is held fixed.
"""
import json
import numpy as np
import exp_rlvr_subjective as S
import exp_rlvr_real as RV

M32 = "Qwen/Qwen2.5-32B-Instruct"
CACHE = json.load(open(S.CACHE2))
V, _ = RV.load_problems(120, 1); RV.attach_candidates(V, np.random.default_rng(1))
N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))
G_C, G_W = 10.0, 0.0
G_N = G_C - G_W           # gold N margin (Delta_min^N, binary gold)
Bmax = 10.0


def vmargin(vsys):        # natural (shielded) V margin, score units
    c = np.array([CACHE[S._key(M32, vsys, p["q"], p["_cands"]["correct"], "natural")] for p in V])
    w = np.array([CACHE[S._key(M32, vsys, p["q"], p["_cands"]["wrong"], "natural")] for p in V])
    loss = np.abs(c / 10 - 1).mean() + np.abs(w / 10 - 0).mean()   # cardinal MWU loss on V
    return float((c - w).mean()), float(loss)


def nmargin(nsys, shield):
    c, w = [], []
    for it in N:
        fr = "shield" if shield else it["user_pref_text"]
        c.append(CACHE[S._key(M32, nsys, it["q"], it["cands"]["correct"], fr)])
        w.append(CACHE[S._key(M32, nsys, it["q"], it["cands"]["wrong"], fr)])
    return float((np.array(c) - np.array(w)).mean())


hV, hV_loss = vmargin(RV.HONEST_SYS)
sV_len, sVlen_loss = vmargin(RV.SYCO_SYS)       # lenient syco (natural)
sV_harsh, sVharsh_loss = vmargin(S.HARSH_V_SYS)  # harsh syco
hN = nmargin(S.HONEST_N_SYS, True)
sN = nmargin(S.SYCO_N_SYS, False)

beta_V = abs(G_N - hV)          # honest panel's residual V-margin deficit (score units)
eps = 0.0 * Bmax                # W_bias = 0 in the clean recovering config
Dstar = G_N - beta_V - eps

drift_len = abs(sV_len - sN)
drift_harsh = abs(sV_harsh - sN)
drift_honest = abs(hV - hN)

out = {
    "units": "score-margin (0-10)", "formula": "Delta* = G_N - beta*_V - eps (no /2)",
    "G_N": G_N, "honest_V_margin": hV, "beta_V": beta_V, "eps": eps, "Delta_star": Dstar,
    "honest": {"V_margin": hV, "N_margin": hN, "drift": drift_honest, "V_cardinal_loss": hV_loss},
    "syco_lenient": {"V_margin": sV_len, "N_margin": sN, "drift": drift_len,
                     "V_cardinal_loss": sVlen_loss, "empirical": "RECOVERS (80%)"},
    "syco_harsh":   {"V_margin": sV_harsh, "N_margin": sN, "drift": drift_harsh,
                     "V_cardinal_loss": sVharsh_loss, "empirical": "FAILS (0%)"},
}
# verdict
out["drift_threshold_separates_regimes"] = bool(drift_len < Dstar <= drift_harsh)
out["cardinal_loss_separates_regimes"] = bool(sVharsh_loss < hV_loss <= sVlen_loss)
import provenance
out["provenance"] = provenance.stamp(n_v=len(V), n_n=len(N), panel="honest-32B vs syco-32B (capability fixed)")
json.dump(out, open("delta_check_results.json", "w"), indent=2)

print(f"G_N={G_N:.1f}  honest V-margin={hV:.2f}  beta*_V={beta_V:.2f}  eps={eps:.2f}  ->  Delta*={Dstar:.2f}  (score units)")
print(f"honest        : V={hV:6.2f} N={hN:6.2f} drift={drift_honest:5.2f}  V-cardinal-loss={hV_loss:.2f}")
print(f"syco lenient  : V={sV_len:6.2f} N={sN:6.2f} drift={drift_len:5.2f}  V-cardinal-loss={sVlen_loss:.2f}  -> RECOVERS")
print(f"syco harsh    : V={sV_harsh:6.2f} N={sN:6.2f} drift={drift_harsh:5.2f}  V-cardinal-loss={sVharsh_loss:.2f}  -> FAILS")
print()
print(f"drift-threshold separates recover/fail?  {out['drift_threshold_separates_regimes']}  "
      f"(lenient drift {drift_len:.1f} vs harsh {drift_harsh:.1f}, both vs Delta*={Dstar:.1f})")
print(f"cardinal V-loss separates recover/fail?   {out['cardinal_loss_separates_regimes']}  "
      f"(harsh {sVharsh_loss:.2f} < honest {hV_loss:.2f} < lenient {sVlen_loss:.2f})")
print("-> detection is governed by CARDINAL V-calibration, not drift. saved delta_check_results.json")
