"""
quick_check.py -- <1 min smoke test of the three headline invariants. GPU-free.
Exits non-zero if any invariant fails. Run: python3 quick_check.py
  1. coverage boundary: ESA recovers the optimum under coverage, fails without it.
  2. active-audit budget is horizon-independent (flat across T).
  3. cardinal->ordinal inversion: ESA recovers with scalar scores, inverts with preferences.
"""
import os
import sys
import json
import numpy as np

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        FAILS.append(name)


# --- 1 & 2: synthetic bandit (fast, seeded) ---
from social_bandit import SocialBanditEnv
from runners import run_esa

def esa_final(coverage, seed):
    env = SocialBanditEnv(k=5, M=10, optimal_arm=0, bias_arm=1, arm_gap=0.5, bias_mag=1.0,
                          sycophant_ratio=0.8, coverage=coverage, seed=seed)
    return run_esa(env, 3000, audit_mode="active", trust_scope="region", tau=0.5, seed=seed)

cov = [esa_final("full", s)["final"] == 0 for s in range(5)]
unc = [esa_final([0, 2, 3, 4], s)["final"] == 0 for s in range(5)]   # bias arm 1 UNcovered
check("1. coverage recovery ON", np.mean(cov) >= 0.8, f"(covered recovery {np.mean(cov):.0%})")
check("1. coverage recovery OFF (should be low)", np.mean(unc) <= 0.4, f"(uncovered recovery {np.mean(unc):.0%})")

audits = {T: esa_final("full", 0) for T in (1000, 8000)}
a1 = run_esa(SocialBanditEnv(k=5, M=10, bias_mag=1.0, sycophant_ratio=0.8, coverage="full", seed=0),
             1000, audit_mode="active", trust_scope="region", tau=0.5, seed=0)["audits"]
a8 = run_esa(SocialBanditEnv(k=5, M=10, bias_mag=1.0, sycophant_ratio=0.8, coverage="full", seed=0),
             8000, audit_mode="active", trust_scope="region", tau=0.5, seed=0)["audits"]
check("2. active-audit budget flat across horizon", a8 <= 3 * max(a1, 1), f"(T=1k:{a1} audits, T=8k:{a8})")

# --- 3: cardinal->ordinal inversion from cached real-grader scores (if present) ---
CACHE = "rlvr_subjective_cache.json"
if os.path.exists(CACHE):
    import exp_rlvr_subjective as S, exp_rlvr_real as RV
    C = json.load(open(CACHE)); M32 = "Qwen/Qwen2.5-32B-Instruct"
    V, _ = RV.load_problems(120, 1); RV.attach_candidates(V, np.random.default_rng(1))
    N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))
    def vp(vsys):
        c = np.array([C[S._key(M32, vsys, p["q"], p["_cands"]["correct"], "natural")] for p in V])
        w = np.array([C[S._key(M32, vsys, p["q"], p["_cands"]["wrong"], "natural")] for p in V])
        return c, w
    def npr(nsys, sh):
        c, w = [], []
        for it in N:
            fr = "shield" if sh else it["user_pref_text"]
            c.append(C[S._key(M32, nsys, it["q"], it["cands"]["correct"], fr)])
            w.append(C[S._key(M32, nsys, it["q"], it["cands"]["wrong"], fr)])
        return np.array(c), np.array(w)
    hVc, hVw = vp(RV.HONEST_SYS); sVc, sVw = vp(RV.SYCO_SYS)
    hNc, hNw = npr(S.HONEST_N_SYS, True); sNc, sNw = npr(S.SYCO_N_SYS, False)
    Vc = np.stack([hVc]*2+[sVc]*8, 1); Vw = np.stack([hVw]*2+[sVw]*8, 1)
    Nc = np.stack([hNc]*2+[sNc]*8, 1); Nw = np.stack([hNw]*2+[sNw]*8, 1)
    def trust(ordinal):
        w = np.ones(10)/10
        for i in range(len(V)):
            if ordinal:
                w *= np.exp(-RV.ETA*(Vw[i] >= Vc[i]))
            else:
                for y, z in [(Vc[i], 1.), (Vw[i], 0.)]:
                    w *= np.exp(-RV.ETA*np.abs(y/10-z))
            w /= w.sum()
        return w
    card = float(((Nc @ trust(False)) > (Nw @ trust(False))).mean())
    ordi = float(((Nc @ trust(True)) > (Nw @ trust(True))).mean())
    check("3. cardinal recovers, ordinal inverts", card >= 0.5 > ordi, f"(cardinal {card:.0%}, ordinal {ordi:.0%})")
else:
    print("[SKIP] 3. ordinal inversion -- score cache absent (run exp_rlvr_subjective.py)")

print(f"\n{'ALL INVARIANTS PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
