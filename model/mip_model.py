"""
mip_model.py  —  Törnquist & Persson geïnspireerde resource-choice formulering.

Elk bezoek (t, s) vraagt exact één fysiek platform/resource p uit zijn optionset.
Alle conflicten — gepland én via retracking — worden afgehandeld door één
generieke headway-constraint, gefilterd op de resource-keuzes van beide visits.

Variabelen
----------
  entry[t, s]       start van bezoek (t, s)
  dep[t, s]         einde (vertrek) van bezoek (t, s)
  delay[t, s_fin]   vertraging op eindsegment van trein t  (Törnquist: z_{n_i})
  use[t, s, p]      binair — bezoek (t, s) gebruikt platform p
  ord[...]          binair — volgorde tussen twee bezoeken op gedeelde resource

Constraints
-----------
  C1  segment-bezetting (rijtijd / verblijftijd)
  C2  padcontinuïteit
  C2b geen vroege inreis (eerste toekomstig segment)
  C3  vertragingsdefinitie (alleen eindsegment)
  C5  minimum verblijftijd op geplande stopplaatsen
  C6a resourcekeuze per bezoek (sum = 1)
  C6b headway tussen elke twee bezoeken die hetzelfde platform kunnen kiezen
"""

from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB
from config.settings import RETRACK_CONFLICT_WINDOW, SWITCH_PENALTY


def build_and_solve_model(
    T, S, Ss, Sl,
    path, sched_entry, sched_exit, runtime, dwell,
    conflicts, occupied, fixed_entry, expected_exit, halts,
    weights, L, current_time,
    time_limit=None, verbose=True, platform_alternatives=None,
):
    model = gp.Model("rail_rescheduling")
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # ── Sets ─────────────────────────────────────────────────────────────────
    TS            = [(t, s) for t in T for s in path[t]]
    final_segment = {t: path[t][-1] for t in T}
    consecutive   = [
        (t, path[t][k], path[t][k + 1])
        for t in T for k in range(len(path[t]) - 1)
    ]

    # ── Tijdvariabelen ────────────────────────────────────────────────────────
    entry = {}
    dep   = {}
    for t, s in TS:
        if (t, s) in fixed_entry:
            v = fixed_entry[t, s]
            entry[t, s] = model.addVar(lb=v, ub=v, name=f"entry[{t},{s}]")
        else:
            entry[t, s] = model.addVar(lb=current_time, name=f"entry[{t},{s}]")
        dep[t, s] = model.addVar(lb=0, name=f"dep[{t},{s}]")

    # Vertraging alleen op eindsegment per trein (Törnquist: z_{n_i})
    delay = {
        (t, final_segment[t]): model.addVar(lb=0, name=f"delay[{t}]")
        for t in T
    }

    # ── Resource-keuzevariabelen ──────────────────────────────────────────────
    platform_alternatives = platform_alternatives or {}

    # options[t, s]: geordende lijst van bruikbare platforms (gepland eerst)
    options: dict[tuple, list] = {}
    for t in T:
        for s in path[t]:
            alts = platform_alternatives.get((t, s), [])
            options[t, s] = list(dict.fromkeys([s] + [a for a in alts if a != s]))

    # resource_events[p]: alle bezoeken (t, s) die platform p kunnen kiezen
    resource_events: dict[str, list] = defaultdict(list)
    for (t, s), platforms in options.items():
        for p in platforms:
            resource_events[p].append((t, s))

    # use[t, s, p] = 1  iff bezoek (t, s) kiest platform p
    # Gurobi verwijdert singleton-opties automatisch via presolve.
    use_index = [
        (t, s, p) for (t, s), platforms in options.items() for p in platforms
    ]
    use = model.addVars(use_index, vtype=GRB.BINARY, name="use")

    # C6a — exact één platform per bezoek
    for (t, s), platforms in options.items():
        model.addConstr(
            gp.quicksum(use[t, s, p] for p in platforms) == 1,
            name=f"C6a[{t},{s}]",
        )

    # ── Objectief ─────────────────────────────────────────────────────────────
    switch_cost = SWITCH_PENALTY * gp.quicksum(
        use[t, s, p]
        for (t, s), platforms in options.items()
        for p in platforms
        if p != s
    )

    model.setObjective(
        gp.quicksum(weights[t] * delay[t, final_segment[t]] for t in T) + switch_cost,
        GRB.MINIMIZE,
    )

    # ── C1 — Segmentbezetting ─────────────────────────────────────────────────
    for t in T:
        for s in path[t]:
            if (t, s) in occupied:
                model.addConstr(
                    dep[t, s] >= current_time + occupied[t, s],
                    name=f"C1[{t},{s}]",
                )
            else:
                duration = runtime[t, s] if s in Sl else dwell[t, s]
                model.addConstr(
                    dep[t, s] >= entry[t, s] + duration,
                    name=f"C1[{t},{s}]",
                )

    # ── C2 — Padcontinuïteit ──────────────────────────────────────────────────
    for t, s, s_next in consecutive:
        model.addConstr(entry[t, s_next] == dep[t, s], name=f"C2[{t},{s}]")

    # ── C2b — Geen vroege inreis ───────────────────────────────────────────────
    for t in T:
        s0 = path[t][0]
        if (t, s0) not in fixed_entry:
            model.addConstr(
                entry[t, s0] >= sched_entry[t, s0], name=f"C2b[{t}]"
            )

    # ── C3 — Vertragingsdefinitie (alleen eindsegment) ────────────────────────
    for t in T:
        s_fin = final_segment[t]
        model.addConstr(
            delay[t, s_fin] >= dep[t, s_fin] - sched_exit[t, s_fin],
            name=f"C3[{t}]",
        )

    # ── C5 — Minimum verblijftijd op geplande stops ───────────────────────────
    for t in T:
        for s in path[t]:
            if halts.get((t, s), False):
                model.addConstr(
                    dep[t, s] >= sched_exit[t, s], name=f"C5[{t},{s}]"
                )

    # ── C6b — Resource-conflicten ─────────────────────────────────────────────
    #
    # Voor elk paar bezoeken (e1, e2) dat hetzelfde platform p kan kiezen:
    #
    #   entry[e2] ≥ dep[e1] − L·(1−ord) − L·(2−use[e1,p]−use[e2,p])
    #   entry[e1] ≥ dep[e2] − L·ord     − L·(2−use[e1,p]−use[e2,p])
    #
    # De gate L·(2−use[e1,p]−use[e2,p]) = 0 alleen als beide p kiezen;
    # anders ontspant de constraint automatisch.
    #
    # Dit vervangt C4 (3 gevallen) + C6b/c/d/e/f (z_alt-linearisatie).
    # Geplande conflicten (stap 1) en retracking-conflicten (stap 2) worden
    # uniform behandeld; seen_pairs voorkomt dubbele toevoeging.

    seen_pairs: set[tuple]          = set()
    order_vars: dict[tuple, gp.Var] = {}

    def add_conflict(e1: tuple, e2: tuple, p: str) -> None:
        t1, s1 = e1
        t2, s2 = e2
        if t1 == t2:
            return
        fwd = (t1, s1, t2, s2, p)
        if fwd in seen_pairs or (t2, s2, t1, s1, p) in seen_pairs:
            return
        seen_pairs.add(fwd)

        o    = model.addVar(vtype=GRB.BINARY, name=f"ord[{t1},{s1},{t2},{s2},{p}]")
        gate = 2 - use[t1, s1, p] - use[t2, s2, p]
        order_vars[fwd] = o

        model.addConstr(
            entry[t2, s2] >= dep[t1, s1] - L * (1 - o) - L * gate,
            name=f"Ca[{t1},{s1},{t2},{s2},{p}]",
        )
        model.addConstr(
            entry[t1, s1] >= dep[t2, s2] - L * o - L * gate,
            name=f"Cb[{t1},{s1},{t2},{s2},{p}]",
        )

    # Stap 1: geplande conflicten (altijd meenemen; opgebouwd in instance.py
    #          met CONFLICT_WINDOW op basis van expected_entry)
    for p, train_pairs in conflicts.items():
        for t1, t2 in train_pairs:
            add_conflict((t1, p), (t2, p), p)

    # Stap 2: alternatieve resource-conflicten (RETRACK_CONFLICT_WINDOW;
    #          voegt retracking-paren toe; seen_pairs slaat overlap met stap 1 over)
    for p, events in resource_events.items():
        events_sorted = sorted(events, key=lambda e: expected_exit.get(e, float("inf")))
        for k, e1 in enumerate(events_sorted):
            exp1 = expected_exit.get(e1, float("inf"))
            for e2 in events_sorted[k + 1:]:
                if expected_exit.get(e2, float("inf")) - exp1 > RETRACK_CONFLICT_WINDOW:
                    break
                add_conflict(e1, e2, p)

    # ── Warm start ────────────────────────────────────────────────────────────
    # use: begin op het geplande platform (geen switches)
    for (t, s), platforms in options.items():
        for p in platforms:
            use[t, s, p].Start = 1.0 if p == s else 0.0

    # ord: timetablevolgorde op basis van expected_exit
    for (t1, s1, t2, s2, p), o in order_vars.items():
        exp1 = expected_exit.get((t1, s1), 0.0)
        exp2 = expected_exit.get((t2, s2), 0.0)
        o.Start = 1.0 if exp1 <= exp2 else 0.0

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.update()
    n_planned = sum(len(v) for v in conflicts.values())
    print(
        f"[t={current_time:.0f}] "
        f"T={len(T)} S={len(S)} "
        f"planned_conf={n_planned} resource_pairs={len(seen_pairs)} "
        f"vars={model.NumVars} constr={model.NumConstrs}"
    )
    model.optimize()

    return model, entry, dep, delay, use, final_segment
