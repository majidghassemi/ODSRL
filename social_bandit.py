"""
social_bandit.py
Upgraded Social Bandit for the coverage-boundary paper.

Key differences from the original run_bandit.SocialBanditEnv:
  * The reference (spot-check / verifier) is informative ONLY on a coverage set C.
    The original code always revealed the true reward on a spot-check, i.e. full
    coverage, which is exactly the "easy" regime the new theory identifies.
  * Bias is a full [M, K] matrix, so bias can be placed on covered OR uncovered
    arms (to build the failure experiment) and given bloc structure (collusion).
  * A verifiable/non-verifiable (V/N) split supports the RLVR transfer experiment,
    with independent bias on V and N so cross-region drift Delta_drift is a knob.

Convention: arm 0 (or `optimal_arm`) is the latent optimum. `socially_preferred_arm`
is the argmax of the naive-mean observed reward; when it differs from the optimum,
objective decoupling occurs (Prop. "Decoupling condition").
"""
import numpy as np


class SocialBanditEnv:
    def __init__(
        self,
        k=10,
        M=10,
        optimal_arm=0,
        arm_gap=0.5,
        sycophant_ratio=0.8,
        bias_arm=1,
        bias_mag=3.0,
        coverage="full",          # "full" | "none" | list/array of covered arm indices
        verifiable_arms=None,     # for RLVR: arms whose reference is dense+exact (sigma_ref~0)
        sigma_env=0.1,
        sigma_eval=0.1,
        sigma_ref=0.1,
        bias_matrix=None,         # optional explicit [M, K] override
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.M = M
        self.optimal_arm = optimal_arm
        self.sigma_env = sigma_env
        self.sigma_eval = sigma_eval
        self.sigma_ref = sigma_ref

        # latent means: optimum is highest by `arm_gap`
        self.true_means = np.full(k, 1.0 - arm_gap, dtype=float)
        self.true_means[optimal_arm] = 1.0

        # bias matrix b[m, a]
        if bias_matrix is not None:
            self.b = np.asarray(bias_matrix, dtype=float)
            assert self.b.shape == (M, k)
            self.is_biased = np.abs(self.b).sum(axis=1) > 1e-9
        else:
            n_syc = int(round(M * sycophant_ratio))
            self.b = np.zeros((M, k))
            # sycophants inflate `bias_arm` so the naive mean can prefer it
            self.b[:n_syc, bias_arm] = bias_mag
            self.is_biased = np.zeros(M, dtype=bool)
            self.is_biased[:n_syc] = True

        # coverage mask over arms
        self.covered = np.zeros(k, dtype=bool)
        if isinstance(coverage, str):
            if coverage == "full":
                self.covered[:] = True
            elif coverage == "none":
                self.covered[:] = False
            else:
                raise ValueError(coverage)
        else:
            self.covered[np.asarray(coverage, dtype=int)] = True

        # verifiable arms (RLVR): always covered, near-exact reference
        self.verifiable = np.zeros(k, dtype=bool)
        if verifiable_arms is not None:
            self.verifiable[np.asarray(verifiable_arms, dtype=int)] = True
            self.covered[self.verifiable] = True

        self._last_arm = None
        self._last_true = None

    # -- diagnostics -------------------------------------------------------
    @property
    def socially_preferred_arm(self):
        observed_mean = self.true_means + self.b.mean(axis=0)
        return int(np.argmax(observed_mean))

    @property
    def coverage_set(self):
        return np.where(self.covered)[0]

    def cross_region_drift(self):
        """Delta_drift proxy: max over evaluators of |mean bias on N - mean bias on V|."""
        if not self.verifiable.any() or self.verifiable.all():
            return None
        bV = self.b[:, self.verifiable].mean(axis=1)
        bN = self.b[:, ~self.verifiable].mean(axis=1)
        return float(np.max(np.abs(bN - bV)))

    # -- dynamics ----------------------------------------------------------
    def step(self, arm):
        true_r = self.rng.normal(self.true_means[arm], self.sigma_env)
        y = true_r + self.b[:, arm] + self.rng.normal(0, self.sigma_eval, self.M)
        self._last_arm = arm
        self._last_true = true_r
        return true_r, y

    def query_reference(self, arm):
        """Return z for the just-played arm if it is covered, else None.
        Verifiable arms use a near-exact reference (sigma_ref/10)."""
        if arm != self._last_arm or not self.covered[arm]:
            return None
        s = self.sigma_ref / 10.0 if self.verifiable[arm] else self.sigma_ref
        return self._last_true + self.rng.normal(0, s)

    def instant_regret(self, arm):
        return self.true_means[self.optimal_arm] - self.true_means[arm]
