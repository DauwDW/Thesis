"""
mip_dynamic.py

Dynamic priority MIP model for train rescheduling.

Extends mip_base with dynamic priority levels (PDL) via binary upgrade variables.

Effective delay weights (see MIP_Model_Dynamic.docx, Table 1):
  psl=0, pdl=0  →  weight 1   (freight, on-time)
  psl=0, pdl=1  →  weight 2   (freight, heavily delayed)
  psl=1, pdl=0  →  weight 2   (passenger, on-time)
  psl=1, pdl=1  →  weight 3   (passenger, heavily delayed)

The objective (1 + psl[t]) * delay[t, s_last] + q[t] produces these weights
because when pdl[t]=1, C8 forces q[t] = delay[t, s_last], adding an extra
delta on top of the base weight. This is correct by design.

Differences from mip_base
--------------------------
- Headway H in C4 (mip_base uses H=0 implicitly)
- Dynamic priority: pdl binary variable, q linearization variable
- C7: PDL upgrade constraints based on gamma threshold
- C8: Linearization of q[t] = pdl[t] * delay[t, s_last]

Note: Tp and Tf are accepted for interface consistency with instance.py
but are not used — weighting is handled via psl.
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
    sched_exit,
    runtime,
    dwell,
    H,
    conflicts,
    occupied,
    fixed_entry,
    psl,
    gamma,
    epsilon,
    delta_max,
    L,
    current_time,
    time_limit=None,
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
    exit  = {}
    delay = {}

    for t, s in TS:

        # --- entry ---
        if (t, s) in fixed_entry:
            fixed = fixed_entry[(t, s)]
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
        exit[t, s] = model.addVar(
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

    y_index = [(i, j, s) for s in S for (i, j) in conflicts[s]]

    y   = model.addVars(y_index, vtype=GRB.BINARY,     name="y")
    pdl = model.addVars(T,       vtype=GRB.BINARY,     name="pdl")
    q   = model.addVars(T,       vtype=GRB.CONTINUOUS, lb=0, name="q")

    # =========================================================================
    # Objective
    # =========================================================================

    model.setObjective(
        gp.quicksum(
            (1 + psl[t]) * delay[t, final_segment[t]] + q[t]
            for t in T
        ),
        GRB.MINIMIZE,
    )

    # =========================================================================
    # C1 — Segment occupation
    # =========================================================================

    for t in T:
        for s in path[t]:
            if (t, s) in occupied:
                duration = occupied[(t, s)]
            elif s in Sl:
                duration = runtime[(t, s)]
            else:
                duration = dwell[(t, s)]

            model.addConstr(
                exit[t, s] >= entry[t, s] + duration,
                name=f"C1_occupation[{t},{s}]",
            )

    # =========================================================================
    # C2 — Path continuity
    # =========================================================================

    for t, s, s_next in consecutive:
        model.addConstr(
            entry[t, s_next] >= exit[t, s],
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
    # C4 — Conflicts with headway
    # =========================================================================

    for s in S:
        for i, j in conflicts[s]:
            model.addConstr(
                entry[j, s] >= exit[i, s] + H[(i, j, s)] - L * (1 - y[i, j, s]),
                name=f"C4a_conflict[{i},{j},{s}]",
            )
            model.addConstr(
                entry[i, s] >= exit[j, s] + H[(j, i, s)] - L * y[i, j, s],
                name=f"C4b_conflict[{i},{j},{s}]",
            )

    # =========================================================================
    # C7 — Dynamic priority upgrade threshold
    # =========================================================================

    for t in T:
        s_last = final_segment[t]

        # C7a: pdl forced to 1 when delay >= gamma
        model.addConstr(
            delay[t, s_last] - gamma <= L * pdl[t],
            name=f"C7a_upgrade_force[{t}]",
        )

        # C7b: pdl forced to 0 when delay < gamma
        model.addConstr(
            delay[t, s_last] - gamma >= epsilon - L * (1 - pdl[t]),
            name=f"C7b_upgrade_block[{t}]",
        )

    # =========================================================================
    # C8 — Linearisation of q[t] = pdl[t] * delay[t, s_last]
    # =========================================================================

    for t in T:
        s_last = final_segment[t]

        # C8a: q[t] = 0 when pdl[t] = 0
        model.addConstr(
            q[t] <= delta_max * pdl[t],
            name=f"C8a_lin_upper_pdl[{t}]",
        )

        # C8b: q[t] cannot exceed actual final delay
        model.addConstr(
            q[t] <= delay[t, s_last],
            name=f"C8b_lin_upper_delay[{t}]",
        )

        # C8c: combined with C8b, forces q[t] = delay[t, s_last] when pdl[t] = 1
        model.addConstr(
            q[t] >= delay[t, s_last] - delta_max * (1 - pdl[t]),
            name=f"C8c_lin_lower[{t}]",
        )

    # =========================================================================
    # Solve
    # =========================================================================

    model.optimize()

    return model, entry, exit, delay, y, pdl, q, conflicts, final_segment