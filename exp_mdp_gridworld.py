"""
Q1 (MDP, gridworld) -- the Sycophant Trap with PER-REGION (contextual) trust.

This is the v2 rewrite of run_sycophant.py. The old InternalFeedbackAgent kept a
single GLOBAL trust vector audited only on a binary SAFETY_VIOLATION flag; that
mechanism does not match the coverage-boundary paper. Here we wire in
`esa_mdp.ContextualTrust` (per-region trust + active auditing + fail-safe) with the
region keyed by grid cell, and show WHY per-region matters:

  A lazy sycophant is honest almost everywhere and only lies near the Candy/Lava/
  Goal cells. GLOBAL trust averages the audit signal over the whole trajectory, so
  the rare divergence at Candy is diluted and the liar stays trusted -> the agent is
  lured to the proxy goal. PER-REGION trust localizes the audit: at the Candy cell
  the liars are down-weighted and the agent recovers the true goal.

Baselines: Mean (Standard RL), Median (robust consensus), Global-ESA (continuous-audit
but one shared weight vector), Region-ESA (ours). Metric: P(visit Candy) per episode.

Runs with numpy only -- no MuJoCo. See exp_mdp_hopper.py for the continuous testbed.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from environments import (
    Evaluator, get_next_state, get_true_reward,
    START_STATE, GOAL_STATE, CANDY_STATE, LAVA_ZONES, GRID_SIZE,
)
from agents import BaseAgent
from esa_mdp import ContextualTrust

# --- config ---
SEEDS = 15
EPISODES = 3000
MAX_STEPS = 40
WINDOW = 75
P_REF = 0.10          # audit probability (spot-check of the latent reward)
ACTIONS = ['up', 'down', 'left', 'right']


class GridContextualTrust(ContextualTrust):
    """ContextualTrust with the region keyed by grid cell instead of obs sign-bits.
    scope='region' -> one trust vector per cell; scope='global' -> a single shared
    vector (n_regions=1), i.e. the old global mechanism with the SAME audit signal,
    so the comparison isolates the per-region contribution."""

    def __init__(self, M, grid=GRID_SIZE, scope='region', **kw):
        n_regions = grid * grid if scope == 'region' else 1
        super().__init__(M, n_regions=n_regions, **kw)
        self.grid = grid
        self.scope = scope

    def region(self, obs):
        if self.scope == 'global':
            return 0
        x, y = obs
        return x * self.grid + y


class ESAAgent(BaseAgent):
    """Q-learner whose perceived reward is the trust-weighted evaluator aggregate."""

    def __init__(self, actions, M, scope='region', alpha=0.05, epsilon=0.1,
                 eta=0.5, p_ref=P_REF):
        super().__init__(actions, alpha=alpha, epsilon=epsilon)
        self.trust = GridContextualTrust(M, scope=scope, eta=eta)
        self.p_ref = p_ref

    def process_feedback(self, state, feedbacks, true_reward):
        y = np.asarray(feedbacks, dtype=float)
        audited = np.random.rand() < self.p_ref
        z = true_reward if audited else None
        if z is not None:
            self.trust.update(state, y, z)          # per-region MWU on |y - z|
        return self.trust.trusted_reward(state, y, reference=z)

    def liar_weight(self, is_liar):
        return float(self.trust.w.mean(0)[is_liar].sum())


class MeanAgent(BaseAgent):
    def process_feedback(self, state, feedbacks, true_reward):
        return float(np.mean(feedbacks))


class MedianAgent(BaseAgent):
    def process_feedback(self, state, feedbacks, true_reward):
        return float(np.median(feedbacks))


def make_agent(kind, M):
    if kind == 'mean':
        return MeanAgent(ACTIONS)
    if kind == 'median':
        return MedianAgent(ACTIONS)
    if kind == 'global_esa':
        return ESAAgent(ACTIONS, M, scope='global')
    if kind == 'region_esa':
        return ESAAgent(ACTIONS, M, scope='region')
    raise ValueError(kind)


def run_single_seed(seed, kind):
    np.random.seed(seed)
    # 5 evaluators: 1 truthful + 4 lazy sycophants = 80% bias
    evaluators = [Evaluator("Truthful", "truthful")] + \
                 [Evaluator(f"Lazy{i}", "lazy_sycophant") for i in range(4)]
    is_liar = np.array([e.bias_type != "truthful" for e in evaluators])
    agent = make_agent(kind, len(evaluators))

    candy_visits = np.zeros(EPISODES)
    liar_trust = np.zeros(EPISODES)
    for ep in range(EPISODES):
        state = START_STATE
        visited_candy = 0
        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state = get_next_state(state, action)
            true_r = get_true_reward(state, action, next_state)
            feedbacks = [e.give_feedback(state, action, next_state, true_r)
                         for e in evaluators]
            r_perceived = agent.process_feedback(state, feedbacks, true_r)
            agent.update(state, action, r_perceived, next_state)
            if next_state == CANDY_STATE:
                visited_candy = 1
            state = next_state
            if state == GOAL_STATE:
                break
        candy_visits[ep] = visited_candy
        if isinstance(agent, ESAAgent):
            liar_trust[ep] = agent.liar_weight(is_liar)
    audits = agent.trust.audits if isinstance(agent, ESAAgent) else 0
    return candy_visits, liar_trust, audits


def smooth(matrix, win=WINDOW):
    out = np.array([np.convolve(r, np.ones(win) / win, mode='same') for r in matrix])
    trim = win // 2
    x = np.arange(matrix.shape[1])[trim:-trim]
    return x, out.mean(0)[trim:-trim], out.std(0)[trim:-trim]


if __name__ == "__main__":
    kinds = ['mean', 'median', 'global_esa', 'region_esa']
    labels = {'mean': 'Standard RL (Mean)', 'median': 'Robust (Median)',
              'global_esa': 'Global-trust ESA', 'region_esa': 'Region-trust ESA (ours)'}
    colors = {'mean': '#EB640B', 'median': '#20693E',
              'global_esa': '#7F8C8D', 'region_esa': '#6C3483'}

    results, audit_counts, final_candy = {}, {}, {}
    print(f"Q1 gridworld Sycophant Trap: {SEEDS} seeds x {EPISODES} episodes")
    for k in kinds:
        cv = np.zeros((SEEDS, EPISODES))
        au = []
        for s in range(SEEDS):
            cv[s], _, a = run_single_seed(s, k)
            au.append(a)
        results[k] = cv
        audit_counts[k] = np.mean(au)
        final_candy[k] = cv[:, -200:].mean()
        print(f"  {labels[k]:28s} P(candy last200)={final_candy[k]:.2f}  "
              f"audits/seed~{audit_counts[k]:.0f}")

    plt.figure(figsize=(11, 6.5))
    for k in kinds:
        x, m, sd = smooth(results[k])
        ls = '-' if k == 'region_esa' else '--'
        lw = 3 if k == 'region_esa' else 2
        plt.plot(x, m, label=labels[k], color=colors[k], linestyle=ls, linewidth=lw)
        plt.fill_between(x, m - sd, m + sd, color=colors[k], alpha=0.12)
    plt.xlabel("Episodes")
    plt.ylabel("P(visit Candy / proxy goal)")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='center right')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    os.makedirs("paper/figures", exist_ok=True)
    out = "paper/figures/q1_mdp_gridworld.png"
    plt.savefig(out, dpi=200)
    print(f"  saved {out}")
