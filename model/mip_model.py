"""
mip_model.py

Single MIP model for train rescheduling.

Priority handling is exogeen: de `weights` dict komt uit instance.py STEP 6
en is al aangepast op basis van priority_strategy ("static" of "dynamic").
De solver hoeft geen weet te hebben van strategy — hij minimaliseert simpelweg
de gewogen som van final-segment delays.

Lower bound semantics voor entry-variabelen:

  1. (t,s) in fixed_entry:
     lb = ub = actual_entry_time
     Actief segment — entry gefixed op werkelijke entrytijd.

  2. Overige (toekomstige) segmenten:
     lb = current_time
     Toekomstige segmenten mogen niet in het verleden starten.
     C2 continuity duwt vervolgsegmenten automatisch verder in de toekomst.
"""

import gurobipy as gp
from gurobipy import GRB
from config.settings import SOLVER_MIP_GAP


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
    expected_exit,
    halts,
    weights,
    L,
    current_time,
    time_limit=None,
    verbose=True,
):
    
    model = gp.Model("rail_rescheduling")
    #Debug
    model.Params.OutputFlag = 0
    # small_alpha = 0.00001
    # model.Params.MIPGap = SOLVER_MIP_GAP
    # epsilon = 0.000000001


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
            # Actief segment — gefixed op werkelijke entrytijd
            fixed = fixed_entry[(t, s)]
            entry[t, s] = model.addVar(
                lb=fixed, ub=fixed,
                vtype=GRB.CONTINUOUS,
                name=f"entry[{t},{s}]",
            )
        else:
            # Toekomstig segment — niet eerder dan current_time
            entry[t, s] = model.addVar(
                lb=current_time,
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
        # + epsilon * gp.quicksum(entry[t, s] for (t, s) in TS),

        # ## timespan penaliseren
        # +
        # small_alpha * gp.quicksum(
        #     dep[t, final_segment[t]] - entry[t, path[t][0]]
        #     for t in T
        # ),
        ## tussentijdse vertragingen penaliseren
        # + small_alpha * gp.quicksum(
        #     delay[t, s] for t in T for s in path[t] if s != final_segment[t]
        # )
        GRB.MINIMIZE,
    )

    # =========================================================================
    # C1 — Segment occupation
    # =========================================================================

    for t in T:
        for s in path[t]:
            if (t, s) in occupied:
                model.addConstr(
                    dep[t, s] >= current_time + occupied[(t, s)],
                    name=f"C1_occupation[{t},{s}]",
                )
            else:
                duration = runtime[(t, s)] if s in Sl else dwell[(t, s)] 
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
    # C2b — Limited early entry for first segment
    # Treinen mogen max 1 minuut te vroeg het netwerk binnenkomen
    # =========================================================================

    EARLY_ENTRY_SLACK = 0

    for t in T:

        first_seg = path[t][0]

        # Niet toepassen op reeds actieve segmenten
        if (t, first_seg) not in fixed_entry:

            model.addConstr(
                entry[t, first_seg]
                >= sched_entry[(t, first_seg)] - EARLY_ENTRY_SLACK,
                name=f"C2b_earlyentry[{t}]",
            )

    # =========================================================================
    # C3 — Delay definition
    # =========================================================================

    for t in T:
        for s in path[t]:
            model.addConstr(
                delay[t, s] >= dep[t, s] - sched_exit[(t, s)],
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
    # C5 — Minimum dwell
    # Trein mag niet vroeger vertrekken dan gepland op stationsegmenten
    # waar hij effectief halteert (geen within-station-passing)
    # =========================================================================
                        # !!!  wnr shit resultaten, zet dit in comment
    for t in T:
        for s in path[t]:
            if halts.get((t, s), False):
                model.addConstr(
                    dep[t, s] >= sched_exit[(t, s)],
                    name=f"C5_mindwell[{t},{s}]",
                )
    

    # =========================================================================
    # Warm start
    # =========================================================================
    for s in S:
        trains_on_seg = sorted(
            [t for t in T if s in path[t]],
            key=lambda t: expected_exit[(t, s)]
        )
        for k in range(len(trains_on_seg)):
            for l in range(k + 1, len(trains_on_seg)):
                i = trains_on_seg[k]
                j = trains_on_seg[l]
                if (i, j, s) in y:
                    y[i, j, s].Start = 1.0
                elif (j, i, s) in y:
                    y[j, i, s].Start = 0.0

    # =========================================================================
    # Solve
    # =========================================================================
    model.update()

    n_conflicts = sum(len(v) for v in conflicts.values())

    print(
        f"[t={current_time:.0f}] "
        f"T={len(T)} "
        f"S={len(S)} "
        f"conf={n_conflicts} "
        f"vars={model.NumVars} "
        f"constr={model.NumConstrs} "
        f"bin={len(y)}"
    )

    model.optimize()

    return model, entry, dep, delay, y, final_segment