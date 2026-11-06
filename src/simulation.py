import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.constants import (
    N_ASSETS, HORIZON, GAMMA, ANNUAL_BUDGET, FAILURE_PENALTY, N_STATES,
    ACTIONS, ACTION_COST, INITIAL_DIST, P,
)

def simulate_network(policy_fn, n_assets=N_ASSETS, horizon=HORIZON,
                     budget=ANNUAL_BUDGET, initial_dist=None, rng=None):
    """Run one Monte Carlo trajectory. Returns per-year histories."""
    if rng is None:
        rng = np.random.default_rng()
    if initial_dist is None:
        initial_dist = INITIAL_DIST
    if not np.isclose(initial_dist.sum(), 1.0):
        raise ValueError("initial_dist must sum to 1.")
    states = rng.choice(N_STATES, size=n_assets, p=initial_dist) # List of initial states 
    # Lists that record the states 
    hist_critical = np.zeros(horizon + 1, dtype=int)
    hist_spend = np.zeros(horizon)
    hist_penalty = np.zeros(horizon)
    hist_critical[0] = (states == 4).sum()

    for t in range(horizon):
        actions = policy_fn(states, budget, t=t)               # list of action strings
        spend = sum(ACTION_COST[a] for a in actions)
        # advance each asset
        new_states = np.empty_like(states)
        for i, (s, a) in enumerate(zip(states, actions)):
            new_states[i] = rng.choice(N_STATES, p=P[a][s])
        states = new_states
        hist_critical[t + 1] = (states == 4).sum()
        hist_spend[t] = spend
        hist_penalty[t] = FAILURE_PENALTY[states].sum()

    return {"critical": hist_critical, "spend": hist_spend,
            "penalty": hist_penalty}


def total_discounted_cost(hist, gamma=GAMMA):   # prioritise fixing criticals early
    t = np.arange(len(hist["spend"]))
    return (gamma ** t * (hist["spend"] + hist["penalty"])).sum()

def run_mc(policy_fn, n_reps=10, initial_dist=None):
    runs = [simulate_network(policy_fn, rng=np.random.default_rng(i),
                            initial_dist=initial_dist)
            for i in range(n_reps)]
    crit = np.array([r["critical"] for r in runs])
    cost = np.array([total_discounted_cost(r) for r in runs])
    return crit, cost