"""
mip_dynamic.py


Effective delay weights (see Table 1 in MIP_Model_Dynamic.docx):
  psl=0, pdl=0  →  weight 1   (low-priority, on-time freight)
  psl=0, pdl=1  →  weight 2   (low-priority, delayed freight)
  psl=1, pdl=0  →  weight 2   (high-priority passenger, on-time)
  psl=1, pdl=1  →  weight 3   (high-priority passenger, heavily delayed)

Parameters (on top of mip_base parameters)
-------------------------------------------
psl       : dict {t: 0 or 1}   Static priority level per train
gamma     : float               Delay upgrade threshold in seconds
epsilon   : float               Small constant for strict C7b separation
delta_max : float               Max feasible delay bound (= tau_max - min arrival)
                                Used as tighter big-M in C8a and C8c.
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
    in_execution=None,
    fix_arrival=None,
    M=None,
    time_limit=None,
    verbose=True,
):


    if in_execution is None:
        in_execution = {}
    if fix_arrival is None:
        fix_arrival = {}
    if M is None:
        M = L

    model = gp.Model("rail_rescheduling_dynamic_milp")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # -------------------------------------------------------------------------
    # Helper sets
    # -------------------------------------------------------------------------
    TS = [(t, s) for t in T for s in path[t]]

    C = {}
    for s in S:
        trains_on_s = [t for t in T if s in path[t]]
        C[s] = [
            (trains_on_s[a], trains_on_s[b])
            for a in range(len(trains_on_s))
            for b in range(a + 1, len(trains_on_s))
        ]

    final_seg = {t: path[t][-1] for t in T}

    consecutive_pairs = []
    for t in T:
        for k in range(len(path[t]) - 1):
            consecutive_pairs.append((t, path[t][k], path[t][k + 1]))

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
            a[t, s] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"a[{t},{s}]") #daarom constraint 5 van in doc niet meer nodig
    d = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="d")
    delta = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="delta")

    y_index = [(i, j, s) for s in S for (i, j) in C[s]]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    pdl = model.addVars(T, vtype=GRB.BINARY, name="pdl")

    q = model.addVars(T, vtype=GRB.CONTINUOUS, lb=0, name="q")

    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------
    model.setObjective(
        gp.quicksum(
            (1 + psl[t]) * delta[t, final_seg[t]] + q[t]
            for t in T),GRB.MINIMIZE)

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
            if s in Ss: #enkel bij stations
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
            # C4a: if i before j (y=1), j must wait for i + headway H[i,j,s]
            model.addConstr(
                a[j, s] >= d[i, s] + H[i, j, s] - M * (1 - y[i, j, s]),
                name=f"C4a_{i}_{j}_{s}")
            # C4b: if j before i (y=0), i must wait for j + headway H[j,i,s]
            model.addConstr(
                a[i, s] >= d[j, s] + H[j, i, s] - M * y[i, j, s],
                name=f"C4b_{i}_{j}_{s}")

    # -------------------------------------------------------------------------
    # C7 — Dynamic priority threshold
    # -------------------------------------------------------------------------
    for t in T:
        s_last = final_seg[t]

        # C7a: upgrade forced when delay >= gamma
        model.addConstr(
            delta[t, s_last] - gamma <= M * pdl[t],
            name=f"C7a_upgrade_force_{t}")

        # C7b: no upgrade when delay < gamma
        model.addConstr(
            delta[t, s_last] - gamma >= epsilon - M * (1 - pdl[t]),
            name=f"C7b_upgrade_block_{t}")

    # -------------------------------------------------------------------------
    # C8 — Linearisation
    # -------------------------------------------------------------------------
    for t in T:
        s_last = final_seg[t]

        # C8a: forces q_t = 0 when pdl_t = 0
        model.addConstr(
            q[t] <= delta_max * pdl[t],
            name=f"C8a_lin_upper_pdl_{t}")

        # C8b: q_t cannot exceed actual final delay
        model.addConstr(
            q[t] <= delta[t, s_last],
            name=f"C8b_lin_upper_delta_{t}")

        # C8c: when pdl_t = 1, combined with C8b, forces q_t = δ_{t, s_last}
        model.addConstr(
            q[t] >= delta[t, s_last] - delta_max * (1 - pdl[t]),
            name=f"C8c_lin_lower_{t}")

    # -------------------------------------------------------------------------
    # Optimize
    # -------------------------------------------------------------------------
    model.optimize()

    return model, a, d, delta, y, pdl, q, C, final_seg