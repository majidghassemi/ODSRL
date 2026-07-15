"""
Cardinal vs ordinal elicitation (uses ONLY cached scores from the natural-coupling
run; no GPU). Point: ESA's detection signal lives in the CARDINAL channel. The
sycophant is the BETTER math grader by pairwise PREFERENCE (V-rel 0.98 > honest
0.80); it only loses on absolute calibration (scores a wrong answer ~6.3 where honest
scores ~1.9). MWU on scalar scores eats that gap and down-weights the sycophant. But
MWU on PREFERENCES (the modality RLHF/DPO/WDPO/KLDPO use) would down-weight the HONEST
graders and ESA would INVERT.

So this is both a limitation and a finding: eliciting ABSOLUTE scores (not rankings)
is what exposes dispositional bias; rankings hide it. Run on the honest-32B vs
syco-32B natural-coupling cache.
"""
import json
import numpy as np
import exp_rlvr_subjective as S
import exp_rlvr_real as RV

M32 = "Qwen/Qwen2.5-32B-Instruct"
ETA = RV.ETA
CACHE = json.load(open(S.CACHE2))
V, _ = RV.load_problems(120, 1)
RV.attach_candidates(V, np.random.default_rng(1))
N = json.load(open("rlvr_subjective_N_Qwen2.5-32B-Instruct.json"))


def vscore(v_sys):
    c, w = [], []
    for p in V:
        kc = S._key(M32, v_sys, p["q"], p["_cands"]["correct"], "natural")
        kw = S._key(M32, v_sys, p["q"], p["_cands"]["wrong"], "natural")
        c.append(CACHE[kc]); w.append(CACHE[kw])
    return np.array(c), np.array(w)


def nscore(n_sys, shield):
    c, w = [], []
    for it in N:
        frame = "shield" if shield else it["user_pref_text"]
        kc = S._key(M32, n_sys, it["q"], it["cands"]["correct"], frame)
        kw = S._key(M32, n_sys, it["q"], it["cands"]["wrong"], frame)
        c.append(CACHE[kc]); w.append(CACHE[kw])
    return np.array(c), np.array(w)


hVc, hVw = vscore(RV.HONEST_SYS)                 # honest V (natural)
sVc, sVw = vscore(RV.SYCO_SYS)                    # syco   V (natural)
hNc, hNw = nscore(S.HONEST_N_SYS, True)           # honest N (shielded)
sNc, sNw = nscore(S.SYCO_N_SYS, False)            # syco   N (belief)

# panel: 2 honest + 8 syco
M = 10
Vc = np.stack([hVc]*2 + [sVc]*8, 1)  # [nV, M]
Vw = np.stack([hVw]*2 + [sVw]*8, 1)
Nc = np.stack([hNc]*2 + [sNc]*8, 1)  # [nN, M]
Nw = np.stack([hNw]*2 + [sNw]*8, 1)
syco = np.array([m >= 2 for m in range(M)])


def cardinal_trust():
    w = np.ones(M) / M
    for pi in range(len(V)):
        for y, z in [(Vc[pi], 1.0), (Vw[pi], 0.0)]:
            w = w * np.exp(-ETA * np.abs(y / 10 - z)); w /= w.sum()
    return w


def ordinal_trust():
    # report = preference; reference = verifier's preference (correct > wrong).
    # loss = 1 if grader ranks wrong above correct (disagrees), else 0.
    w = np.ones(M) / M
    for pi in range(len(V)):
        disagree = (Vw[pi] >= Vc[pi]).astype(float)     # grader prefers wrong -> loss 1
        w = w * np.exp(-ETA * disagree); w /= w.sum()
    return w


def recover_cardinal(w):
    return float((Nc @ w > Nw @ w).mean())


def recover_ordinal(w):
    # trust-weighted vote of preferences
    pref_c = (Nc > Nw).astype(float)               # [nN, M] grader prefers correct
    return float((pref_c @ w > (1 - pref_c) @ w).mean())


wc, wo = cardinal_trust(), ordinal_trust()
print("=== Cardinal vs Ordinal elicitation (natural-coupling cache) ===")
print(f"honest V pref-rel={ (hVc>hVw).mean():.2f}  syco V pref-rel={(sVc>sVw).mean():.2f}")
print(f"honest V scores: correct={hVc.mean():.2f} wrong={hVw.mean():.2f}")
print(f"syco   V scores: correct={sVc.mean():.2f} wrong={sVw.mean():.2f}  (leniency = calibration gap)")
print()
print(f"CARDINAL trust -> mass on syco={wc[syco].sum():.3f}  |  ESA N-recovery={recover_cardinal(wc):.0%}")
print(f"ORDINAL  trust -> mass on syco={wo[syco].sum():.3f}  |  ESA N-recovery={recover_ordinal(wo):.0%}")
print()
print("naive-mean N-recovery (cardinal):", f"{float((Nc.mean(1) > Nw.mean(1)).mean()):.0%}")
