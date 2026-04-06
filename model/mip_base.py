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
    w,
    L,
    M=None,
    time_limit=None,
    verbose=True
):

    if M is None:
        M = L

    model = gp.Model("rail_rescheduling_milp")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # ----------------------------
    # Helper sets
    # ----------------------------
    TS = [(t, s) for t in T for s in path[t]]

    C = {}
    for s in S:
        trains_on_s = [t for t in T if s in path[t]]
        C[s] = [(trains_on_s[a], trains_on_s[b])
                for a in range(len(trains_on_s))
                for b in range(a + 1, len(trains_on_s))]

    final_seg = {t: path[t][-1] for t in T}

    consecutive_pairs = []
    for t in T:
        for k in range(len(path[t]) - 1):
            s = path[t][k]
            s_next = path[t][k + 1]
            consecutive_pairs.append((t, s, s_next))

    # ----------------------------
    # Variables
    # ----------------------------
    a = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="a")
    d = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="d")
    delta = model.addVars(TS, vtype=GRB.CONTINUOUS, lb=0, name="delta")

    y_index = [(i, j, s) for s in S for (i, j) in C[s]]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    # ----------------------------
    # Objective
    # ----------------------------
    model.setObjective(
        gp.quicksum(w[t] * delta[t, final_seg[t]] for t in T),
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------------
    # C1 — Time consistency within a train
    # ------------------------------------------------------------------

    # # C1a — minimum running time on line segments
    for t in T:
        for s in path[t]:
            if s in Sl:
                model.addConstr(
                    d[t, s] >= a[t, s] + RT[t, s],
                    name=f"C1a_run_{t}_{s}"
                )

    # # C1b — minimum dwell time on station segments, only when stopping
    for t in T:
        for s in path[t]:
            if s in Ss:
                model.addConstr(
                    d[t, s] >= a[t, s] + DW[t, s] * h_stop[t, s],
                    name=f"C1b_dwell_{t}_{s}"
                )

    # # C1c — transition between consecutive segments
    for t, s, s_next in consecutive_pairs:
        model.addConstr(
            a[t, s_next] >= d[t, s],
            name=f"C1c_transition_{t}_{s}_{s_next}"
        )

    # ------------------------------------------------------------------
    # C2 — No early departure at station segments if stopping
    # ------------------------------------------------------------------
    for t in T:
        for s in path[t]:
            if s in Ss:
                model.addConstr(
                    d[t, s] >= sched_dep[t, s] - M * (1 - h_stop[t, s]),
                    name=f"C2_no_early_departure_{t}_{s}"
                )

    # ------------------------------------------------------------------
    # C3 — Delay definition
    # ------------------------------------------------------------------
    for t in T:
        for s in path[t]:
            model.addConstr(
                delta[t, s] >= a[t, s] - sched_entry[t, s],
                name=f"C3_delay_{t}_{s}"
            )

    # ------------------------------------------------------------------
    # C4 — Conflict constraints / headway enforcement
    # ------------------------------------------------------------------
    for s in S:
        for i, j in C[s]:
            model.addConstr(
                a[j, s] >= d[i, s] + H[i, j, s]
                - M * (1 - y[i, j, s]),
                name=f"C4a_{i}_{j}_{s}"
            )
            model.addConstr(
                a[i, s] >= d[j, s] + H[j, i, s]
                - M * y[i, j, s],
                name=f"C4b_{i}_{j}_{s}"
            )

    # ------------------------------------------------------------------
    # C5 — Domain constraints
    # Already handled by variable types
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------
    model.optimize()

    return model, a, d, delta, y, C, final_seg


# =============================================================================
# DUMMY DATA
# =============================================================================

build_and_solve_model(
    T  = ["T1", "T2"],
    Tp = ["T1"],
    Tf = ["T2"],
    S  = ["S1", "S2", "S3"],
    Ss = ["S1", "S3"],
    Sl = ["S2"],
    path = {
        "T1": ["S1", "S2", "S3"],
        "T2": ["S1", "S2", "S3"],
    },
    sched_entry = {
        ("T1", "S1"): 0,  ("T1", "S2"): 10, ("T1", "S3"): 20,
        ("T2", "S1"): 5,  ("T2", "S2"): 15, ("T2", "S3"): 25,
    },
    sched_dep = {
        ("T1", "S1"): 5,  ("T1", "S2"): 15, ("T1", "S3"): 25,
        ("T2", "S1"): 10, ("T2", "S2"): 20, ("T2", "S3"): 30,
    },
    RT = {
        ("T1", "S2"): 8,
        ("T2", "S2"): 8,
    },
    DW = {
        ("T1", "S1"): 3, ("T1", "S3"): 2,
        ("T2", "S1"): 3, ("T2", "S3"): 2,
    },
    H = {
        ("T1", "T2", "S1"): 3, ("T2", "T1", "S1"): 3,
        ("T1", "T2", "S2"): 4, ("T2", "T1", "S2"): 4,
        ("T1", "T2", "S3"): 3, ("T2", "T1", "S3"): 3,
    },
    h_stop = {
        ("T1", "S1"): 1, ("T1", "S3"): 1,
        ("T2", "S1"): 1, ("T2", "S3"): 1,
    },
    w        = {"T1": 2, "T2": 1},
    L        = 100,
)