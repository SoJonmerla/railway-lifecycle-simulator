"""MDP solver and priority-index policy for the railway maintenance problem.

The unconstrained per-asset MDP is solved by backward induction. The resulting
value function is used to compute a priority index — future cost saved per pound
spent — for each (state, action) pair. The index policy allocates the annual
budget greedily by priority.
"""
import numpy as np
from src.constants import (
    N_ASSETS, HORIZON, GAMMA, ANNUAL_BUDGET, FAILURE_PENALTY, N_STATES,
    ACTIONS, ACTION_COST, INITIAL_DIST, P,
)

def solve_unconstrained_mdp(P, cost, penalty, actions, horizon, gamma=1.0):
    """
    Solve the unconstrained per-asset MDP by backward induction.

    Parameters
    ----------
    P : dict[str, np.ndarray]   transition matrices, P[a][s, s'] = Prob(s'|s,a)
    cost : dict[str, float]     per-action cost
    penalty : np.ndarray (S,)   per-state per-year penalty
    actions : list[str]         action labels matching keys of P and cost
    horizon : int
    gamma : float               discount factor (default 1.0)

    Returns
    -------
    V  : np.ndarray (horizon+1, S)   value function; V[horizon, :] = 0
    pi : np.ndarray (horizon, S)     optimal action index at each (t, s)
    """
    n_states = N_STATES
    V = np.zeros((horizon + 1, n_states))
    pi = np.zeros((horizon, n_states), dtype=int)

    for t in range(horizon - 1, -1, -1):
        for s in range(n_states):
            q = np.array([
                cost[a] + penalty[s] + gamma * (P[a][s] @ V[t + 1])
                for a in actions
            ])
            k = int(np.argmin(q))
            V[t, s] = q[k]
            pi[t, s] = k
    return V, pi


def compute_priority_table(V, t, P, cost, penalty, actions, gamma=1.0):
    """Priority of action a in state s at year t:
        [future cost of 'do nothing' - future cost of a] / cost(a)

    Units: pounds of future cost saved per pound spent now.
    Zero for 'do nothing'. Negative if the action is worse than doing nothing.
    """
    n_states = V.shape[1]
    pri = np.zeros((n_states, len(actions)))
    for s in range(n_states):
        q_nothing = (cost["do_nothing"] + penalty[s]
                     + gamma * (P["do_nothing"][s] @ V[t + 1]))
        for k, a in enumerate(actions):
            if a == "do_nothing":
                continue
            q_a = (cost[a] + penalty[s]
                   + gamma * (P[a][s] @ V[t + 1]))
            pri[s, k] = (q_nothing - q_a) / cost[a]
    return pri


def make_index_policy(V, P, cost, penalty, actions, gamma=1.0):
    """Return a policy (states, budget, t) -> list[str] that allocates greedily
    by priority index, one action per asset per year."""
    def policy(states, budget, t=None):
        out = ["do_nothing"] * len(states)
        remaining = budget
        if t is None:
            t = 0
        pri = compute_priority_table(V, t, P, cost, penalty, actions, gamma)

        candidates = []
        for i, s in enumerate(states):
            for k, a in enumerate(actions):
                if a == "do_nothing":
                    continue
                if pri[s, k] > 0:
                    candidates.append((pri[s, k], i, k))

        candidates.sort(reverse=True)
        assigned = set()
        for _, i, k in candidates:
            if i in assigned:
                continue
            a = actions[k]
            if cost[a] <= remaining:
                out[i] = a
                remaining -= cost[a]
                assigned.add(i)
        return out
    return policy