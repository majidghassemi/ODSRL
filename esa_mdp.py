"""
esa_mdp.py
MDP-side ESA mechanism for the Hopper / gridworld PPO experiments.

This is the deep-RL analogue of the bandit ESA in esa.py, upgraded from the
original global TrustMechanism to match the coverage-boundary theory:
  * PER-REGION (contextual) trust, keyed by a coarse state region, so bias in one
    part of the state space does not decay an evaluator's weight everywhere.
  * ACTIVE auditing: query the internal axiom signal only while a region is
    unresolved, giving a horizon-independent audit budget.
  * ABSOLUTE-DISTRUST fail-safe: if every source in a region exceeds loss tau, fall
    back to the reference / a conservative value instead of the sycophantic signal.

NOT RUNNABLE IN THE SANDBOX: this needs gymnasium + MuJoCo + stable-baselines3.
It is provided wired-up so it drops straight into StrategicHopperWrapper. See the
USAGE block at the bottom for the two-line change to advanced_experiments.py.
"""
import numpy as np


class ContextualTrust:
    def __init__(self, num_evaluators, n_regions=8, eta=0.05, tau=None,
                 agree_margin=0.5, patience=20):
        self.M = num_evaluators
        self.R = n_regions
        self.eta = eta
        self.tau = tau
        self.agree_margin = agree_margin
        self.patience = patience
        self.w = np.ones((n_regions, num_evaluators)) / num_evaluators
        self.resolved = np.zeros(n_regions, dtype=bool)
        self.distrusted = np.zeros(n_regions, dtype=bool)
        self.ema_loss = np.full((n_regions, num_evaluators), np.nan)
        self.streak = np.zeros(n_regions, dtype=int)
        self.audits = 0

    def region(self, obs):
        """Coarse, cheap state partition. Replace with a tiling/cluster id if you
        have one; the default hashes the sign pattern of the observation."""
        bits = (np.asarray(obs) > 0).astype(int)
        h = 0
        for b in bits[:16]:
            h = (h * 2 + int(b)) % self.R
        return h

    def want_audit(self, g):
        return not self.resolved[g]

    def update(self, obs, social_signals, internal_axiom_signal):
        """internal_axiom_signal = reference z (latent reward) if audited, else None."""
        g = self.region(obs)
        if internal_axiom_signal is not None and self.want_audit(g):
            self.audits += 1
            loss = np.abs(social_signals - internal_axiom_signal)
            self.w[g] *= np.exp(-self.eta * loss)
            self.w[g] /= self.w[g].sum()
            self.ema_loss[g] = np.where(np.isnan(self.ema_loss[g]), loss,
                                        0.7 * self.ema_loss[g] + 0.3 * loss)
            if self.tau is not None and np.nanmin(self.ema_loss[g]) > self.tau:
                self.distrusted[g] = True
            agg = float(np.dot(self.w[g], social_signals))
            if abs(agg - internal_axiom_signal) <= self.agree_margin:
                self.streak[g] += 1
                if self.streak[g] >= self.patience:
                    self.resolved[g] = True
            else:
                self.streak[g] = 0

    def trusted_reward(self, obs, social_signals, reference=None):
        g = self.region(obs)
        if self.distrusted[g]:
            # fail-safe: use the reference if present, else a conservative estimate
            return reference if reference is not None else float(np.min(social_signals))
        return float(np.dot(self.w[g], social_signals))

    def liar_weight(self, is_liar):
        return float(self.w.mean(0)[is_liar].sum())


# ---------------------------------------------------------------------------
# USAGE (drop into advanced_experiments.py StrategicHopperWrapper.__init__/step):
#
#   from esa_mdp import ContextualTrust
#   self.trust_model = ContextualTrust(self.M, n_regions=8, eta=0.05, tau=1.0)
#
#   # in step(), replace the global update/get with:
#   internal = latent_reward if np.random.rand() < 0.10 else None   # active gate below
#   if internal is not None and not self.trust_model.resolved[self.trust_model.region(obs)]:
#       self.trust_model.update(obs, social_signals, internal)
#   perceived = self.trust_model.trusted_reward(obs, social_signals, reference=internal)
#
# Everything else (PPO, plotting) is unchanged. The audit count is
# self.trust_model.audits; log it to reproduce the horizon-independence plot.
# ---------------------------------------------------------------------------
