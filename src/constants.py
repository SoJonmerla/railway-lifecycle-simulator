import numpy as np
# Network parameters
N_ASSETS = 1000
HORIZON = 20
GAMMA = 1                    # Left this as 1 since accidents are as bad short term and long term?
ANNUAL_BUDGET = 200.0        # total across the network
# Per-state penalty: only state 4 (Critical) carries a failure cost.
FAILURE_PENALTY = np.array([0.0, 0.0, 0.0, 0.0, 5.0], dtype=float)
N_STATES = 5                 # 0=Very Good ... 4 = Critical

STATES = ["Very Good", "Good", "Fair", "Poor", "Critical"] # Chosen depending on Track Geometry Index (TGI) or other TQI?
ACTIONS = ["do_nothing", "minor", "major"]
ACTION_COST = {"do_nothing": 0.0, "minor": 1.0, "major": 8.0}

INITIAL_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0])   # all start Very Good

# Transition matrices. No maintance tends to degrade asset, maintance tends to improve asset.
P_DO_NOTHING = np.array([
    [0.85, 0.15, 0.00, 0.00, 0.00],
    [0.00, 0.80, 0.20, 0.00, 0.00],
    [0.00, 0.00, 0.70, 0.30, 0.00],
    [0.00, 0.00, 0.00, 0.60, 0.40],
    [0.00, 0.00, 0.00, 0.00, 1.00],
])

P_MINOR = np.array([
    [0.95, 0.05, 0.00, 0.00, 0.00],
    [0.10, 0.85, 0.05, 0.00, 0.00],
    [0.00, 0.15, 0.80, 0.05, 0.00],
    [0.00, 0.00, 0.20, 0.75, 0.05],
    [0.00, 0.00, 0.00, 0.25, 0.75],
])

P_MAJOR = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00],
    [0.30, 0.70, 0.00, 0.00, 0.00],
    [0.10, 0.60, 0.30, 0.00, 0.00],
    [0.00, 0.30, 0.60, 0.10, 0.00],
    [0.00, 0.20, 0.50, 0.30, 0.00],
])

P = {"do_nothing": P_DO_NOTHING, "minor": P_MINOR, "major": P_MAJOR}

# Sanity check: each row sums to 1
for name, m in P.items():
    assert np.allclose(m.sum(axis=1), 1), f"{name} rows do not sum to 1"
