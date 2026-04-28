"""
mip_dynamic.py

Dynamic priority MIP model for train rescheduling.

Effective delay weights (see Table 1 in MIP_Model_Dynamic.docx):
  psl=0, pdl=0  →  weight 1   (low-priority, on-time freight)
  psl=0, pdl=1  →  weight 2   (low-priority, delayed freight)
  psl=1, pdl=0  →  weight 2   (high-priority passenger, on-time)
  psl=1, pdl=1  →  weight 3   (high-priority passenger, heavily delayed)

The objective (1 + psl[t]) * delta[t, s_last] + q[t] produces these weights
because when pdl=1, C8 forces q[t] = delta[t, s_last], adding an extra delta
on top of the base weight. This is correct by design.

Parameters (on top of mip_base parameters)
-------------------------------------------
psl       : dict {t: 0 or 1}   Static priority level per train
gamma     : float               Delay upgrade threshold in seconds
epsilon   : float               Small constant for strict C7b separation
delta_max : float               Max feasible delay bound (= tau_max - min arrival)
                                Used as tighter big-M in C8a and C8c.

Note: Tp and Tf (passenger/freight train subsets) are accepted for interface
consistency with mip_base and instance.py, but are not used directly in model
construction — weighting is handled via the psl parameter instead.
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
    sched_dep,
    RT,
    DW,
    H,
    h_stop,
    psl,
    gamma,
    epsilon,
    delta_max,
    L,
    C=None,
    in_execution=None,
    fix_arrival=None,
    M=None,
    time_limit=None,
    verbose=True,
    current_time=None,
):
    if in_execution is None:
        in_execution = {}
    if fix_arrival is None:
        fix_arrival = {}
    if M is None:
        M = L

    # -------------------------------------------------------------------------
    # VALIDATE fix_arrival
    # -------------------------------------------------------------------------
    if current_time is not None:
        for (t, s), arrival_time in fix_arrival.items():
            if arrival_time < current_time:
                raise ValueError(
                    f"FIX_ARRIVAL IN PAST: train {t}, seg {s}, "
                    f"{arrival_time} < {current_time}"
                )

    model = gp.Model("rail_rescheduling_dynamic_milp")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # -------------------------------------------------------------------------
    # Helper sets
    # -------------------------------------------------------------------------
    TS = [(t, s) for t in T for s in path[t]]

    # Accept external C (pre-filtered by instance.py with CONFLICT_WINDOW)
    # to avoid adding ordering constraints for train pairs far apart in time.
    # Fall back to full pairwise construction only if C is not provided.
    if C is None:
        C = {}
        for s in S:
            trains_on_s = [t for t in T if s in path[t]]
            C[s] = [
                (trains_on_s[ii], trains_on_s[jj])
                for ii in range(len(trains_on_s))
                for jj in range(ii + 1, len(trains_on_s))
            ]

    final_seg = {t: path[t][-1] for t in T}

    consecutive_pairs = [
        (t, path[t][k], path[t][k + 1])
        for t in T
        for k in range(len(path[t]) - 1)
    ]

    # -------------------------------------------------------------------------
    # Variables
    # -------------------------------------------------------------------------
    a = {}
    for t, s in TS:
        if (t, s) in fix_arrival:
            fixed_time = fix_arrival[t, s]
            a[t, s] = model.addVar(
                lb=fixed_time, ub=fixed_time,
                vtype=GRB.CONTINUOUS, name=f"a[{t},{s}]")
        else:
            lb = current_time if current_time is not None else 0.0
            a[t, s] = model.addVar(
                lb=lb,
                vtype=GRB.CONTINUOUS, name=f"a[{t},{s}]")

    d     = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="d")
    delta = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="delta")

    y_index = [(i, j, s) for s in S for (i, j) in C[s]]
    y   = model.addVars(y_index, vtype=GRB.BINARY,     name="y")
    pdl = model.addVars(T,       vtype=GRB.BINARY,     name="pdl")
    q   = model.addVars(T,       vtype=GRB.CONTINUOUS, lb=0, name="q")

    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------
    # Base weight: (1 + psl[t]) — 1 for freight, 2 for passenger.
    # When pdl[t]=1 (heavily delayed), C8 forces q[t] = delta[t, s_last],
    # adding an extra delta and producing effective weights 2 (freight) / 3 (passenger).
    model.setObjective(
        gp.quicksum(
            (1 + psl[t]) * delta[t, final_seg[t]] + q[t]
            for t in T
        ),
        GRB.MINIMIZE,
    )

    # -------------------------------------------------------------------------
    # C1 — Time consistency within a train
    # -------------------------------------------------------------------------

    # C1a — minimum running time on line segments
    for t in T:
        for s in path[t]:
            if s in Sl:
                duration = in_execution.get((t, s), RT[t, s])
                model.addConstr(
                    d[t, s] >= a[t, s] + duration,
                    name=f"C1a_run_{t}_{s}")

    # C1b — minimum dwell time on station segments (only when stopping)
    for t in T:
        for s in path[t]:
            if s in Ss:
                if (t, s) in in_execution:
                    model.addConstr(
                        d[t, s] >= a[t, s] + in_execution[t, s],
                        name=f"C1b_dwell_{t}_{s}")
                else:
                    model.addConstr(
                        d[t, s] >= a[t, s] + DW[t, s] * h_stop[t, s],
                        name=f"C1b_dwell_{t}_{s}")

    # C1c — transition between consecutive segments
    for t, s, s_next in consecutive_pairs:
        model.addConstr(
            a[t, s_next] >= d[t, s],
            name=f"C1c_transition_{t}_{s}_{s_next}")

    # -------------------------------------------------------------------------
    # C2 — No early departure (conditional on halt indicator)
    # -------------------------------------------------------------------------
    for t in T:
        for s in path[t]:
            if s in Ss:
                model.addConstr(
                    d[t, s] >= sched_dep[t, s] - M * (1 - h_stop[t, s]),
                    name=f"C2_no_early_dep_{t}_{s}")

    # -------------------------------------------------------------------------
    # C3 — Delay definition
    # -------------------------------------------------------------------------
    for t in T:
        for s in path[t]:
            model.addConstr(
                delta[t, s] >= a[t, s] - sched_entry[t, s],
                name=f"C3_delay_{t}_{s}")

    # -------------------------------------------------------------------------
    # C4 — Conflict constraints / headway enforcement
    # -------------------------------------------------------------------------
    for s in S:
        for i, j in C[s]:
            # C4a: if i before j (y=1), j must wait for i + headway
            model.addConstr(
                a[j, s] >= d[i, s] + H[i, j, s] - M * (1 - y[i, j, s]),
                name=f"C4a_{i}_{j}_{s}")
            # C4b: if j before i (y=0), i must wait for j + headway
            model.addConstr(
                a[i, s] >= d[j, s] + H[j, i, s] - M * y[i, j, s],
                name=f"C4b_{i}_{j}_{s}")

    # -------------------------------------------------------------------------
    # C7 — Dynamic priority threshold
    # -------------------------------------------------------------------------
    for t in T:
        s_last = final_seg[t]

        # C7a: pdl forced to 1 when delay >= gamma
        model.addConstr(
            delta[t, s_last] - gamma <= M * pdl[t],
            name=f"C7a_upgrade_force_{t}")

        # C7b: pdl forced to 0 when delay < gamma
        model.addConstr(
            delta[t, s_last] - gamma >= epsilon - M * (1 - pdl[t]),
            name=f"C7b_upgrade_block_{t}")

    # -------------------------------------------------------------------------
    # C8 — Linearisation of q[t] = pdl[t] * delta[t, s_last]
    # -------------------------------------------------------------------------
    for t in T:
        s_last = final_seg[t]

        # C8a: q[t] = 0 when pdl[t] = 0
        model.addConstr(
            q[t] <= delta_max * pdl[t],
            name=f"C8a_lin_upper_pdl_{t}")

        # C8b: q[t] cannot exceed actual final delay
        model.addConstr(
            q[t] <= delta[t, s_last],
            name=f"C8b_lin_upper_delta_{t}")

        # C8c: combined with C8b, forces q[t] = delta[t, s_last] when pdl[t] = 1
        model.addConstr(
            q[t] >= delta[t, s_last] - delta_max * (1 - pdl[t]),
            name=f"C8c_lin_lower_{t}")

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------
    model.optimize()

    return model, a, d, delta, y, pdl, q, C, final_seg