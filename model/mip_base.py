
import gurobipy as gp
from gurobipy import GRB


def build_and_solve_model(

    T,
    S,
    Ss,
    Sl,

    path,

    sched_entry,
    sched_exit,

    runtime,
    dwell,

    conflicts,

    occupied,
    fixed_entry,
    actual_entries,

    weights,

    L,

    current_time,

    time_limit=None,
    verbose=True,
):
    """
    Static priority MIP model for train rescheduling.

    Lower bound semantics voor entry-variabelen:

      1. (t,s) in fixed_entry:
         lb = ub = current_time
         Actief segment — trein zit hier NU, entry gefixed.

      2. (t,s) in actual_entries:
         lb = ub = actual_entry[(t,s)]
         Segment al betreden, entry ligt in het verleden en kan
         niet meer aangepast worden — gefixed op werkelijke waarde.

      3. Overige segmenten:
         lb = sched_entry[(t,s)]
         Toekomstige segmenten. De continuity constraint
         entry[next] >= dep[current] garandeert automatisch dat
         deze segmenten >= current_time zijn zonder artificiële
         delay te injecteren.
    """

    model = gp.Model("rail_rescheduling")

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
    dep   = {}
    delay = {}

    for t, s in TS:

        # --- entry ---

        if (t, s) in fixed_entry:
            # 1. Actief segment — gefixed op current_time
            fixed = fixed_entry[(t, s)]
            entry[t, s] = model.addVar(
                lb=fixed, ub=fixed,
                vtype=GRB.CONTINUOUS,
                name=f"entry[{t},{s}]",
            )

        elif (t, s) in actual_entries:
            # 2. Al betreden — gefixed op werkelijke entrytijd
            ae = actual_entries[(t, s)]
            entry[t, s] = model.addVar(
                lb=ae, ub=ae,
                vtype=GRB.CONTINUOUS,
                name=f"entry[{t},{s}]",
            )

        else:
            # 3. Toekomstig segment — lb = sched_entry
            entry[t, s] = model.addVar(
                lb=sched_entry[(t, s)],
                vtype=GRB.CONTINUOUS,
                name=f"entry[{t},{s}]",
            )

        # --- departure ---

        dep[t, s] = model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"dep[{t},{s}]",
        )

        # --- delay ---

        delay[t, s] = model.addVar(
            lb=0,
            vtype=GRB.CONTINUOUS,
            name=f"delay[{t},{s}]",
        )

    y_index = [(i, j, s) for s in S for (i, j) in conflicts[s]]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    # =========================================================================
    # Objective
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

            if (t, s) in occupied:
                duration = occupied[(t, s)]
            elif s in Sl:
                duration = runtime[(t, s)]
            else:
                duration = dwell[(t, s)]

            model.addConstr(
                dep[t, s] >= entry[t, s] + duration,
                name=f"C1_occupation[{t},{s}]",
            )

    # =========================================================================
    # C2 — Path continuity
    # =========================================================================

    for t, s, s_next in consecutive:
        model.addConstr(
            entry[t, s_next] == dep[t, s],
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
    # C4 — Conflicts
    # =========================================================================

    for s in S:
        for i, j in conflicts[s]:
            model.addConstr(
                entry[j, s] >= dep[i, s] - L * (1 - y[i, j, s]),
                name=f"C4a_conflict[{i},{j},{s}]",
            )
            model.addConstr(
                entry[i, s] >= dep[j, s] - L * y[i, j, s],
                name=f"C4b_conflict[{i},{j},{s}]",
            )

    # =========================================================================
    # Solve
    # =========================================================================

    model.optimize()

    return model, entry, dep, delay, y, final_segment