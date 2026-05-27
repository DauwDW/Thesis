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

from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from config.settings import SOLVER_MIP_GAP, CONFLICT_WINDOW, RETRACK_CONFLICT_WINDOW, SWITCH_PENALTY


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
    platform_alternatives=None,
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
    # Retracking: platform-keuze variabelen
    # =========================================================================

    platform_alternatives = platform_alternatives or {}

    # V: visits (t, s_current) met alternatieven die ook in de huidige instance zitten.
    # Na Bug-fix (1) is s_current het HUIDIGE gekozen segment (= sleutel in path[t]).
    # P[(t, s_current)] = [s_current, alt1, alt2, ...] — volledige pool
    V: set[tuple] = set()
    P: dict[tuple, list[str]] = {}
    for (t, s_current), alts in platform_alternatives.items():
        # s_current is na Bug-fix (1) altijd het gekozen segment → check slaagt altijd
        # als instance.py correct filterde; de dubbele check is een veiligheidsnetz.
        if t in T and s_current in path.get(t, ()):
            V.add((t, s_current))
            P[(t, s_current)] = [s_current] + alts

    # V_seg: snel opzoeken welke treinen retracking-vrijheid hebben op segment s.
    # Hier gedefineerd zodat zowel alt_conflict_list als fixedalt_conflict_list
    # (Bug-fix 5) er gebruik van kunnen maken.
    V_seg: dict[str, set] = defaultdict(set)  # s → {train_ids met retrack op s}
    for (t, s) in V:
        V_seg[s].add(t)

    # Fysiek platform → alle retrackbare visits die het kunnen gebruiken
    platform_visits: dict[str, list[tuple]] = defaultdict(list)
    for (t, s_current), options in P.items():
        for p in options:
            platform_visits[p].append((t, s_current))

    # Potentiële conflict-tripels (t_i, s_i, t_j, s_j, p) tussen twee retrackbare
    # visits op hetzelfde alternatieve platform, met tijdsvenster-filter en cap.
    alt_conflict_list: list[tuple] = []
    for p, visits in platform_visits.items():
        if len(visits) <= 1:
            continue
        visits_sorted = sorted(
            visits,
            key=lambda v: expected_exit.get(v, float("inf"))
        )
        for k, (t_i, s_i) in enumerate(visits_sorted):
            for t_j, s_j in visits_sorted[k + 1:]:
                exp_i = expected_exit.get((t_i, s_i), 0)
                exp_j = expected_exit.get((t_j, s_j), 0)
                if exp_j - exp_i > RETRACK_CONFLICT_WINDOW:
                    break
                alt_conflict_list.append((t_i, s_i, t_j, s_j, p))

    # Bug-fix (5): conflicten tussen een retrackbare trein (r) die overstapt naar
    # alternatief platform p, en een VASTE trein (f) die p altijd bezet.
    #
    # platform_visits bevat alleen retrackbare visits.  Als f niet retrackbaar is
    # op p maar wel p in zijn path heeft, wordt het conflict nooit gemodelleerd:
    # C4 kent alleen conflicten op segmenten in path[t], en alternatieve platforms
    # (niet in path[r]) vallen buiten C4.  C6 (z_alt/y_alt) werkt alleen voor twee
    # retrackbare visits.
    #
    # Oplossing: bouw fixedalt_conflict_list en voeg C6e/C6f constraints toe die
    # actief zijn als r platform p kiest (x[r, s_r, p] = 1).
    fixedalt_conflict_list: list[tuple] = []   # (r, s_r, f, p)
    seen_fixedalt: set[tuple] = set()
    for (r, s_r), options in P.items():
        for p in options:
            if p == s_r:
                continue  # Huidig platform: C4 dekt dit al
            for f in T:
                if f == r:
                    continue
                if p not in path.get(f, ()):
                    continue  # f bezoekt p niet
                if f in V_seg.get(p, set()):
                    continue  # f is ook retrackbaar op p → gedekt door alt_conflict_list
                key_fa = (r, s_r, f, p)
                if key_fa in seen_fixedalt:
                    continue
                seen_fixedalt.add(key_fa)
                # Tijdsvenster-filter: gebruik expected_exit als proxy voor timing op p
                exp_r = expected_exit.get((r, s_r), float("inf"))
                exp_f = expected_exit.get((f, p), float("inf"))
                if abs(exp_r - exp_f) <= RETRACK_CONFLICT_WINDOW:
                    fixedalt_conflict_list.append(key_fa)

    # x[t, s_current, p]: visit (t, s_current) kiest platform p
    x_index = [(t, s, p) for (t, s), options in P.items() for p in options]
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x") if x_index else {}

    # z_alt[t_i, s_i, t_j, s_j, p]: beide visits kiezen p (linearisatie AND)
    # y_alt[...]: sequencing op gekozen platform
    ijp_index = [(t_i, s_i, t_j, s_j, p) for t_i, s_i, t_j, s_j, p in alt_conflict_list]
    z_alt = model.addVars(ijp_index, vtype=GRB.BINARY, name="z_alt") if ijp_index else {}
    y_alt = model.addVars(ijp_index, vtype=GRB.BINARY, name="y_alt") if ijp_index else {}

    # y_fixedalt[r, s_r, f, p]: sequencing r vs. vaste trein f op alternatief p
    y_fixedalt = (
        model.addVars(fixedalt_conflict_list, vtype=GRB.BINARY, name="y_fa")
        if fixedalt_conflict_list else {}
    )

    # =========================================================================
    # Objective
    # =========================================================================

    model.setObjective(

        gp.quicksum(
            weights[t] * delay[t, final_segment[t]]
            for t in T
        )

        # Platform-switch penalty: straf voor elke afwijking van het geplande platform.
        # Voorkomt onnodige wisselingen die nieuwe ongemodelleerde conflicten creëren.
        # Alleen actief als er retracking-variabelen zijn (x niet leeg).
        + (
            SWITCH_PENALTY * gp.quicksum(
                x[t, s_planned, p]
                for (t, s_planned), options in P.items()
                for p in options
                if p != s_planned
            )
            if x else 0
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
    #
    # Drie gevallen per paar (i, j) op segment s:
    #
    #   (a) Geen van beiden retrackbaar op s → standaard C4.
    #
    #   (b) Beide retrackbaar op s → overgeslagen; C6 (z_alt / y_alt)
    #       handelt dit af via de platform-keuze variabelen.
    #
    #   (c) Eén trein retrackbaar (r), de andere vastgelegd (f) op s →
    #       "eenzijdige" C4: de headway-constraint geldt alleen als r ook
    #       daadwerkelijk zijn geplande platform s kiest (x[r, s, s] = 1).
    #       Als r een alternatief kiest (x[r, s, s] = 0) is er geen conflict.
    # =========================================================================

    # V_seg is al gedefineerd in de retracking-sectie boven (voor fixedalt).
    for s in S:
        for i, j in conflicts[s]:
            i_rt = i in V_seg.get(s, set())  # heeft i retracking op s?
            j_rt = j in V_seg.get(s, set())  # heeft j retracking op s?

            if i_rt and j_rt:
                # Geval (b): beide retrackbaar op s.
                #
                # C6 (z_alt/y_alt) dekt conflicten op ALTERNATIEVE platforms, maar
                # alleen voor paren die door de cap/window-filter van alt_conflict_list
                # komen.  Het conflict op het geplande platform s zelf (wanneer beide
                # treinen s kiezen) moet hier expliciet worden gemodelleerd —
                # anders is het volledig ongebonden als het buiten de cap/window valt.
                #
                # Oplossing: voeg C4 toe met twee extra M-termen die de constraint
                # inactief maken zodra één van beide treinen een alternatief kiest.
                #   x[i, s, s] = 1  iff trein i zijn geplande platform s kiest
                #   x[j, s, s] = 1  iff trein j zijn geplande platform s kiest
                # De drie M-termen (y, x_i, x_j) zorgen dat de constraint trivially
                # wordt voldaan als y=0 of x_i=0 of x_j=0.
                model.addConstr(
                    entry[j, s] >= dep[i, s]
                        - L * (1 - y[i, j, s])
                        - L * (1 - x[i, s, s])
                        - L * (1 - x[j, s, s]),
                    name=f"C4a_bothrt[{i},{j},{s}]",
                )
                model.addConstr(
                    entry[i, s] >= dep[j, s]
                        - L * y[i, j, s]
                        - L * (1 - x[i, s, s])
                        - L * (1 - x[j, s, s]),
                    name=f"C4b_bothrt[{i},{j},{s}]",
                )
                # C6 handelt conflicten op alternatieve platforms af.
                continue

            if not i_rt and not j_rt:
                # Geval (a): standaard C4 — geen retracking betrokken
                model.addConstr(
                    entry[j, s] >= dep[i, s] - L * (1 - y[i, j, s]),
                    name=f"C4a_conflict[{i},{j},{s}]",
                )
                model.addConstr(
                    entry[i, s] >= dep[j, s] - L * y[i, j, s],
                    name=f"C4b_conflict[{i},{j},{s}]",
                )
            else:
                # Geval (c): eenzijdige retracking
                # r = retrackbare trein, f = vastgelegde trein op s
                r = i if i_rt else j
                # x[r, s, s] = 1 iff r kiest zijn geplande platform s
                # Als x[r, s, s] = 0, valt r op een alternatief → geen conflict op s.
                # De i/j volgorde van y[i,j,s] wordt bewaard (y=1 ↔ i vóór j):
                # beide constraints worden simpelweg gegate op x[r, s, s].
                model.addConstr(
                    entry[j, s] >= dep[i, s]
                        - L * (1 - y[i, j, s])
                        - L * (1 - x[r, s, s]),
                    name=f"C4a_onesided[{i},{j},{s}]",
                )
                model.addConstr(
                    entry[i, s] >= dep[j, s]
                        - L * y[i, j, s]
                        - L * (1 - x[r, s, s]),
                    name=f"C4b_onesided[{i},{j},{s}]",
                )

    # =========================================================================
    # C6 — Retracking: platform-keuze constraints
    # =========================================================================

    # C6a — Exact één platform per visit
    for (t, s_planned), options in P.items():
        model.addConstr(
            gp.quicksum(x[t, s_planned, p] for p in options) == 1,
            name=f"C6a_choice[{t},{s_planned}]",
        )

    # C6b/c/d — Conditonele conflicten op alternatieve platforms
    for t_i, s_i, t_j, s_j, p in alt_conflict_list:
        key = (t_i, s_i, t_j, s_j, p)

        # Linearisatie: z_alt = x_i AND x_j
        model.addConstr(
            z_alt[key] <= x[t_i, s_i, p],
            name=f"C6b_zup1[{t_i},{t_j},{p}]",
        )
        model.addConstr(
            z_alt[key] <= x[t_j, s_j, p],
            name=f"C6b_zup2[{t_i},{t_j},{p}]",
        )
        model.addConstr(
            z_alt[key] >= x[t_i, s_i, p] + x[t_j, s_j, p] - 1,
            name=f"C6b_zlow[{t_i},{t_j},{p}]",
        )

        # Headway enkel actief als beide visits platform p kiezen
        model.addConstr(
            entry[t_j, s_j] >= dep[t_i, s_i]
                - L * (1 - y_alt[key])
                - L * (1 - z_alt[key]),
            name=f"C6c_head[{t_i},{t_j},{p}]",
        )
        model.addConstr(
            entry[t_i, s_i] >= dep[t_j, s_j]
                - L * y_alt[key]
                - L * (1 - z_alt[key]),
            name=f"C6d_head[{t_i},{t_j},{p}]",
        )

    # C6e/f — Bug-fix (5): conflict retrackbare trein op alternatief vs. vaste trein
    #
    # Als retrackbare trein r platform p kiest (p ≠ s_r, p ∈ alts), maar vaste
    # trein f ook op p rijdt, bestaat er een headway-conflict dat niet in C4 of
    # C6b/c/d zit.  Deze constraints zijn actief als x[r, s_r, p] = 1 (r kiest p).
    # f is "vast" op p: f in path[f] maar NIET retrackbaar op p (niet in V_seg[p]).
    #
    #   y_fa = 0  ↔  r vóór f  (C6e actief: entry[f] >= dep[r])
    #   y_fa = 1  ↔  f vóór r  (C6f actief: entry[r] >= dep[f])
    for r, s_r, f, p in fixedalt_conflict_list:
        key_fa = (r, s_r, f, p)
        if key_fa not in y_fixedalt:
            continue
        # entry[f, p] en dep[f, p] bestaan: f heeft p in zijn path.
        # entry[r, s_r] en dep[r, s_r] bestaan: s_r zit in path[r].
        model.addConstr(
            entry[f, p] >= dep[r, s_r]
                - L * y_fixedalt[key_fa]
                - L * (1 - x[r, s_r, p]),
            name=f"C6e_fa[{r},{s_r},{f},{p}]",
        )
        model.addConstr(
            entry[r, s_r] >= dep[f, p]
                - L * (1 - y_fixedalt[key_fa])
                - L * (1 - x[r, s_r, p]),
            name=f"C6f_fa[{r},{s_r},{f},{p}]",
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

    # Warm start retracking: begin met huidig platform (geen switch)
    for (t, s_current), options in P.items():
        for p in options:
            if (t, s_current, p) in x:
                x[t, s_current, p].Start = 1.0 if p == s_current else 0.0
    for key in ijp_index:
        z_alt[key].Start = 0.0
        y_alt[key].Start = 0.0
    # Warm start y_fixedalt: y=0 → r vóór f, y=1 → f vóór r.
    # Als r eerder verwacht wordt (exp_r <= exp_f) geeft r als eerste → y=0.
    for r, s_r, f, p in fixedalt_conflict_list:
        key_fa = (r, s_r, f, p)
        if key_fa in y_fixedalt:
            exp_r = expected_exit.get((r, s_r), 0.0)
            exp_f = expected_exit.get((f, p),   0.0)
            y_fixedalt[key_fa].Start = 0.0 if exp_r <= exp_f else 1.0

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

    return model, entry, dep, delay, y, x, final_segment