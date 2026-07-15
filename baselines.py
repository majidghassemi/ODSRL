"""
baselines.py
Reward-aggregation baselines for the Social Bandit.

IMPORTANT / HONEST NOTE on the "new" baselines:
WDPO, KLDPO and RRM are LLM *preference-optimization* methods; they cannot be run
verbatim on a scalar-reward bandit. What is portable is their defining robustness
ASSUMPTION, which we implement as a scalar aggregator and label as an analogue:

  * RobustDRO_KL   (KLDPO-analogue): entropic-risk / KL-DRO aggregate. Pessimistic,
    down-weights over-optimistic reports. KL ambiguity ball around the report dist.
  * RobustDRO_Wass (WDPO-analogue):  epsilon-trimmed mean (Wasserstein-DRO of the
    mean is a trim/CVaR-type estimator). Robust to an eps-fraction of corruptions.
  * RRM            (RRM-analogue):    Huber-style robust reward: down-weight reports
    by deviation from the running median (separates "artifact" reports from signal).

Every one of these is MAJORITY-TRUSTING: none uses an external reference. Under an
>50% systematic bias they all track the corrupted majority, which is exactly the
point of the coverage-boundary result. Only ESA (esa.py), which audits against the
reference, escapes. Document them this way in the paper; do not claim they are the
original LLM methods.

Common interface: agent.process(y) -> scalar reward estimate.  Stateful ones
(Dawid-Skene) also read history internally.
"""
import numpy as np


class Mean:
    name = "Mean (Standard RL)"
    def process(self, y): return float(np.mean(y))


class Median:
    name = "Median"
    def process(self, y): return float(np.median(y))


class GAILProxy:
    """Imitation baseline: mimic the majority's reported value (the biased consensus)."""
    name = "GAIL (mimic majority)"
    def process(self, y):
        # majority direction = trimmed mean of the densest half
        ys = np.sort(y)
        half = len(ys) // 2
        # densest contiguous half (smallest spread) approximates the majority cluster
        spreads = [ys[i + half - 1] - ys[i] for i in range(len(ys) - half + 1)]
        i = int(np.argmin(spreads))
        return float(np.mean(ys[i:i + half]))


class DawidSkene:
    name = "Dawid-Skene"
    def __init__(self, M, em_every=50, warmup=10):
        self.w = np.ones(M) / M
        self.hist = []
        self.em_every = em_every
        self.warmup = warmup
    def process(self, y):
        self.hist.append(np.asarray(y))
        if len(self.hist) >= self.warmup and len(self.hist) % self.em_every == 0:
            D = np.array(self.hist)
            truth = np.average(D, axis=1, weights=self.w)
            err = np.mean((D - truth[:, None]) ** 2, axis=0)
            nw = 1.0 / (err + 1e-6)
            self.w = nw / nw.sum()
        return float(np.dot(self.w, y))


class RobustDRO_KL:
    """KLDPO-analogue: entropic risk R_alpha(y) = -(1/a) log E[exp(-a y)]."""
    name = "KL-DRO (KLDPO-analogue)"
    def __init__(self, alpha=1.0): self.a = alpha
    def process(self, y):
        y = np.asarray(y)
        return float(-np.log(np.mean(np.exp(-self.a * y))) / self.a)


class RobustDRO_Wass:
    """WDPO-analogue: eps-trimmed mean (drop eps-fraction each tail)."""
    name = "Wass-DRO (WDPO-analogue)"
    def __init__(self, eps=0.2): self.eps = eps
    def process(self, y):
        ys = np.sort(np.asarray(y))
        k = int(np.floor(self.eps * len(ys)))
        core = ys[k:len(ys) - k] if len(ys) - 2 * k > 0 else ys
        return float(np.mean(core))


class RRM:
    """RRM-analogue: Huber-weighted robust reward around the running median."""
    name = "RRM (robust reward)"
    def __init__(self, delta=1.0): self.delta = delta
    def process(self, y):
        y = np.asarray(y)
        med = np.median(y)
        r = np.abs(y - med)
        w = np.where(r <= self.delta, 1.0, self.delta / (r + 1e-9))
        return float(np.dot(w, y) / w.sum())


def make_baselines(M):
    return [Mean(), Median(), DawidSkene(M), GAILProxy(),
            RobustDRO_KL(), RobustDRO_Wass(), RRM()]
