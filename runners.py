"""runners.py -- run ESA or a baseline aggregator on a SocialBanditEnv."""
import numpy as np
from esa import ESA


def run_esa(env, T, eta=0.5, audit_mode="active", p_ref=0.1, tau=None,
            trust_scope="region", seed=0):
    ag = ESA(env.k, env.M, eta=eta, audit_mode=audit_mode, p_ref=p_ref,
             tau=tau, trust_scope=trust_scope, seed=seed)
    reg = np.empty(T)
    cum = 0.0
    for t in range(T):
        a = ag.select_arm()
        _, y = env.step(a)
        ag.observe(a, y, env.query_reference)
        cum += env.instant_regret(a)
        reg[t] = cum
    return {"regret": reg, "final": int(np.argmax(ag.mu)), "audits": ag.audits,
            "agent": ag}


def run_baseline(env, aggregator, T, c_ucb=np.sqrt(2)):
    """UCB over arms using a majority-trusting aggregator (no reference queries)."""
    k = env.k
    mu = np.zeros(k)
    N = np.zeros(k)
    reg = np.empty(T)
    cum = 0.0
    for t in range(T):
        bonus = c_ucb * np.sqrt(np.log(t + 2) / (N + 1e-9))
        a = int(np.argmax(mu + bonus))
        _, y = env.step(a)
        r_hat = aggregator.process(y)
        N[a] += 1
        mu[a] += (r_hat - mu[a]) / N[a]
        cum += env.instant_regret(a)
        reg[t] = cum
    return {"regret": reg, "final": int(np.argmax(mu)), "audits": 0}
