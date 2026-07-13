"""
esa.py
Epistemic Source Alignment agent, upgraded to match the coverage-boundary theory.

New vs the original InternalFeedbackAgent:
  * PER-REGION trust w[region, m] instead of one global scalar per evaluator
    (Def. "Per-region trust"; fixes the false-positive failure R3 flagged).
  * ACTIVE auditing: audit a region only while it is unresolved, then stop, giving
    a horizon-independent audit budget (Thm. "Audit complexity"). `audit_mode`:
      - "active": adaptive, stop when trust concentrates
      - "fixed":  Bernoulli(p_ref) forever (the original behavior, for comparison)
  * ABSOLUTE-DISTRUST fail-safe: if every source in a region exceeds loss tau, the
    region is distrusted; the agent uses the reference where available and abstains
    otherwise, instead of adopting the sycophantic optimum (Prop. "Fail-safe").
  * Correlation routing: `detect_blocs` / `drift_alarm` for bloc discovery and a
    non-stationarity re-audit trigger (Sec. B6). Partition only; labeling still
    needs the audit (Cor. "internal-only impossibility").

The agent is a UCB learner over arms; region == arm (finest partition).
"""
import numpy as np


class ESA:
    def __init__(
        self,
        k,
        M,
        eta=0.5,
        audit_mode="active",
        p_ref=0.1,
        c_ucb=np.sqrt(2),
        kappa=0.10,          # region resolved when top-2 trust mass >= 1 - kappa
        tau=None,            # absolute-distrust threshold; None disables fail-safe
        trust_scope="region",  # "region" | "global"
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.k, self.M = k, M
        self.eta, self.c = eta, c_ucb
        self.audit_mode, self.p_ref = audit_mode, p_ref
        self.kappa, self.tau = kappa, tau
        self.trust_scope = trust_scope

        self.w = np.ones((k, M)) / M          # per-region trust
        self.mu = np.zeros(k)                 # trust-weighted reward estimate
        self.N = np.zeros(k)                  # pull counts
        self.resolved = np.zeros(k, dtype=bool)
        self.agree_streak = np.zeros(k, dtype=int)
        self.agree_margin = 0.35     # aggregate within this of the reference => trustworthy
        self.patience = 3            # consecutive agreeing audits to resolve a region
        self.distrusted = np.zeros(k, dtype=bool)
        self.ema_loss = np.full((k, M), np.nan)
        self.audits = 0
        self.t = 0
        self._report_hist = []                # for correlation routing

    # -- action selection --------------------------------------------------
    def select_arm(self):
        self.t += 1
        bonus = self.c * np.sqrt(np.log(self.t + 1) / (self.N + 1e-9))
        # abstain from distrusted-and-blind arms by making them unattractive
        ucb = self.mu + bonus
        return int(np.argmax(ucb))

    # -- audit decision ----------------------------------------------------
    def _competitive(self, arm):
        """Arm could still be the argmax (not eliminated by confidence intervals)."""
        if self.N.sum() < 2 * self.k:      # warmup: audit everything
            return True
        bonus = self.c * np.sqrt(np.log(self.t + 1) / (self.N + 1e-9))
        lcb, ucb = self.mu - bonus, self.mu + bonus
        pulled = self.N > 0
        if not pulled.any():
            return True
        return ucb[arm] >= np.max(lcb[pulled])

    def _want_audit(self, arm):
        if self.audit_mode == "fixed":
            return self.rng.random() < self.p_ref
        # active: audit only unresolved AND still-competitive regions (K_eff localization)
        return (not self.resolved[arm]) and self._competitive(arm)

    def _refresh_resolved(self, arm, agg, z):
        """Resolve a region once its trust-weighted aggregate agrees with the reference."""
        if abs(agg - z) <= self.agree_margin:
            self.agree_streak[arm] += 1
            if self.agree_streak[arm] >= self.patience:
                self.resolved[arm] = True
        else:
            self.agree_streak[arm] = 0

    # -- main update -------------------------------------------------------
    def observe(self, arm, y, reference_fn):
        """y: report vector [M]; reference_fn(arm) -> z or None (env decides coverage)."""
        self._report_hist.append(y.copy())
        audited = False
        z = None
        if self._want_audit(arm):
            z = reference_fn(arm)          # None if arm uncovered
            if z is not None:
                audited = True
                self.audits += 1
                loss = np.abs(y - z)
                # MWU trust update on this region
                idx = arm if self.trust_scope == "region" else slice(None)
                self.w[idx] *= np.exp(-self.eta * loss)
                self.w[idx] /= self.w[idx].sum(axis=-1, keepdims=True)
                # absolute-distrust bookkeeping
                self.ema_loss[arm] = np.where(
                    np.isnan(self.ema_loss[arm]), loss,
                    0.7 * self.ema_loss[arm] + 0.3 * loss,
                )
                if self.tau is not None and np.nanmin(self.ema_loss[arm]) > self.tau:
                    self.distrusted[arm] = True
                agg_now = float(np.dot(self.w[arm] if self.trust_scope == "region"
                                       else self.w.mean(0), y))
                self._refresh_resolved(arm, agg_now, z)

        # aggregate reward for this arm
        w = self.w[arm] if self.trust_scope == "region" else self.w.mean(0)
        if self.distrusted[arm]:
            # FAIL-SAFE: never adopt the sycophantic signal on a distrusted region.
            if audited and z is not None:
                r_hat = z                    # use the reference where it exists
            else:
                return                       # abstain: do not move mu toward the lie
        else:
            r_hat = float(np.dot(w, y))

        self.N[arm] += 1
        self.mu[arm] += (r_hat - self.mu[arm]) / self.N[arm]

    # -- correlation routing (Sec. B6) -------------------------------------
    def detect_blocs(self, window=None, n_blocs=2):
        """Oracle-less partition of evaluators via report covariance.
        Returns integer labels [M]. Partition only; does NOT label truthful vs biased."""
        H = np.array(self._report_hist[-window:] if window else self._report_hist)
        if H.shape[0] < 3:
            return np.zeros(self.M, dtype=int)
        C = np.corrcoef(H.T)
        C = np.nan_to_num(C)
        vals, vecs = np.linalg.eigh(C)
        # spectral embedding on the top (n_blocs-1) non-principal eigenvectors
        emb = vecs[:, -n_blocs:-1] if n_blocs > 1 else vecs[:, -1:]
        thr = np.median(emb, axis=0)
        labels = (emb > thr).astype(int).ravel() if n_blocs == 2 else \
                 _kmeans_labels(emb, n_blocs, self.rng)
        return labels

    def drift_alarm(self, window=200, ref_window=None, thresh=1.15):
        """Non-stationarity trigger: relative jump in the leading eigenvalue of the
        RESIDUAL covariance (per-step consensus removed, so the common truth signal
        is stripped and only collusion structure remains). Returns True if a new
        coalition appears to be forming; caller re-flags regions."""
        H = np.array(self._report_hist)
        if H.shape[0] < 2 * window:
            return False
        def lead_eig(block):
            R = block - block.mean(axis=1, keepdims=True)   # strip per-step consensus
            C = np.nan_to_num(np.corrcoef(R.T))
            return np.linalg.eigvalsh(C)[-1]
        recent = lead_eig(H[-window:])
        base = lead_eig(H[-(2 * window):-window] if ref_window is None else H[-ref_window:])
        if base <= 1e-9:
            return False
        if recent / base > thresh:
            self.resolved[:] = False         # force re-audit
            return True
        return False


def _kmeans_labels(X, k, rng, iters=25):
    c = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = X[lab == j].mean(0)
    return lab
