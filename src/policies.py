import numpy as np
from src.constants import (
    N_ASSETS, HORIZON, GAMMA, ANNUAL_BUDGET, FAILURE_PENALTY, N_STATES,
    ACTIONS, ACTION_COST, INITIAL_DIST, P,
)


def policy_do_nothing(states, budget, t=None):
    return ["do_nothing"] * len(states)


def policy_worst_first(states, budget, t=None):
    """Major on worst assets first, then minor, until budget exhausted."""
    actions = ["do_nothing"] * len(states)
    remaining = budget
    order = np.argsort(-states)          # state 4 first, then 3, ...
    for i in order:
        if states[i] == 4 and remaining >= ACTION_COST["major"]:
            actions[i] = "major"; remaining -= ACTION_COST["major"]
        elif states[i] >= 3 and remaining >= ACTION_COST["minor"]:
            actions[i] = "minor"; remaining -= ACTION_COST["minor"]
    return actions


def policy_threshold(k):
    """Minor if state >= k, major if state == 4."""
    def policy(states, budget, t=None):
        actions = ["do_nothing"] * len(states)
        remaining = budget
        for i in np.argsort(-states):          # ← worst assets first
            if states[i] == 4 and remaining >= ACTION_COST["major"]:
                actions[i] = "major"; remaining -= ACTION_COST["major"]
            elif states[i] >= k and remaining >= ACTION_COST["minor"]:
                actions[i] = "minor"; remaining -= ACTION_COST["minor"]
        return actions
    return policy