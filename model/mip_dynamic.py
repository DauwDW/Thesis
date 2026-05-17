"""
mip_dynamic.py

Dynamic priority MIP model for train rescheduling.

Replaces the endogenous PDL upgrade mechanism (C7/C8) with exogenous weights
computed in instance.py before the solve. Trains whose current measured delay
exceeds gamma receive an upgraded weight (passenger: 3, freight: 1) via the
`weights` dict — no binary pdl variables or linearisation constraints needed.

Effective delay weights (set in instance.py STEP 6):
  passenger, current_delay <  gamma  →  weight 2   (WEIGHT_PASSENGER)
  passenger, current_delay >= gamma  →  weight 3   (WEIGHT_PASSENGER + upgrade)
  freight,   any delay               →  weight 1   (WEIGHT_FREIGHT)

Differences from mip_base
--------------------------
- Headway H in C4 (mip_base uses H=0 implicitly)
- Dynamic weights are pre-computed exogenously in instance.py; this model
  is structurally identical to mip_base but accepts the extended interface
  of solver.py (Tp, Tf, H, h_stop, psl, gamma, epsilon, delta_max).

Note: Tp, Tf, H, h_stop, psl, gamma, epsilon, delta_max are accepted for
interface compatibility but are not used — weighting is fully handled via
the `weights` dict passed through instance["weights"].
"""

import gurobipy as gp
from gurobipy import GRB


def build_and_solve_model(
    T,
    Tp,
    Tf,
    S,
    Ss,
    Sl,
    path,
    sched_entry,
    sched_dep,        # solver.py: sched_dep    = instance["sched_exit"]
    RT,               # solver.py: RT           = instance["runtime"]
    DW,               # solver.py: DW           = instance["dwell"]
    H,                # accepted, not used (H=0 always)
    h_stop,           # accepted, not used
    psl,              # accepted, not used (weights already upgraded in instance.py)
    gamma,            # accepted, not used
    epsilon,          # accepted, not used
    delta_max,        # accepted, not used
    L,
    C,                # solver.py: C            = instance["conflicts"]
    in_execution,     # solver.py: in_execution = instance["occupied"]
    fix_arrival,      # solver.py: fix_arrival  = instance["fixed_entry"]
    weights,          # exogenous weights — upgraded by instance.py based on current delay
    time_limit=None,
    current_time=None,
    verbose=True,
):
    model = gp.Model("rail_rescheduling_dynamic")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # =========================================================================
    # Sets
    # =========================================================================

    TS = [(t, s) for t in T for s in path[t]]

    final_segment = {t: path[t][-1] for t in T}

    consecutive = [
        (t, path[t][k], path[t][k + 1])
        for t in T
        for k in range(len(path[t]) - 1)
    ]

    # =========================================================================
    # Variables
    # =========================================================================

    entry = {}
    exit_ = {}
    delay = {}

    for t, s in TS:

        # --- entry ---
        if (t, s) in fix_arrival:
            fixed = fix_arrival[(t, s)]
            entry[t, s] = model.addVar(
                lb=fixed, ub=fixed,
                vtype=GRB.CONTINUOUS,
                name=f"entry[{t},{s}]",
            )
        else:
            lb = max(current_time, sched_entry[(t, s)])
            entry[t, s] = model.addVar(
                lb=lb,
                vtype=GRB.CONTINUOUS,
                name=f"entry[{t},{s}]",
            )

        # --- exit ---
        exit_[t, s] = model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"exit[{t},{s}]",
        )

        # --- delay ---
        delay[t, s] = model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"delay[{t},{s}]",
        )

    y_index = [(i, j, s) for s in S for (i, j) in C[s]]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    # =========================================================================
    # Objective — exogenous weights, no pdl/q needed
    # =========================================================================

    model.setObjective(
        gp.quicksum(
            weights[t] * delay[t, final_segment[t]]
            for t in T
        ),
        GRB.MINIMIZE,
    )

    # =========================================================================
    # C1 — Segment occupation
    # =========================================================================

    for t in T:
        for s in path[t]:
            if (t, s) in in_execution:
                duration = in_execution[(t, s)]
            elif s in Sl:
                duration = RT[(t, s)]
            else:
                duration = DW[(t, s)]

            model.addConstr(
                exit_[t, s] >= entry[t, s] + duration,
                name=f"C1_occupation[{t},{s}]",
            )

    # =========================================================================
    # C2 — Path continuity
    # =========================================================================

    for t, s, s_next in consecutive:
        model.addConstr(
            entry[t, s_next] == exit_[t, s],
            name=f"C2_continuity[{t},{s},{s_next}]",
        )

    # =========================================================================
    # C3 — Delay definition
    # =========================================================================

    for t in T:
        for s in path[t]:
            model.addConstr(
                delay[t, s] >= entry[t, s] - sched_entry[(t, s)],
                name=f"C3_delay[{t},{s}]",
            )

    # =========================================================================
    # C4 — Conflicts (H=0, geen headway)
    # =========================================================================

    for s in S:
        for i, j in C[s]:
            model.addConstr(
                entry[j, s] >= exit_[i, s] - L * (1 - y[i, j, s]),
                name=f"C4a_conflict[{i},{j},{s}]",
            )
            model.addConstr(
                entry[i, s] >= exit_[j, s] - L * y[i, j, s],
                name=f"C4b_conflict[{i},{j},{s}]",
            )

    # =========================================================================
    # Solve
    # =========================================================================

    model.optimize()

    # pdl=None, q=None — parse_solution handles this gracefully
    return model, entry, exit_, delay, y, None, None, C, final_segment