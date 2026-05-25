"""
model/fcfs.py

Berekent de objective-waarde van een FCFS-dispatching oplossing voor
een gegeven MIP-instance. Gebruikt als baseline om te beslissen of de
MIP-oplossing voldoende verbetering biedt om toegepast te worden.

FCFS-principe:
  Op elk segment gaan treinen in volgorde van hun expected_entry.
  Een trein die wil binnenkomen terwijl een eerdere trein nog bezig is,
  wacht tot die eerdere trein vertrokken is.

Algoritme: vaste-punt iteratie
  1. Initialiseer entry-tijden uit expected_entry (of fixed_entry voor
     actieve segmenten).
  2. C1: bereken dep uit entry + bezettingsduur.
  3. C2: propageer dep[t,s] → entry[t,s_next] langs het pad.
  4. C4: per segment, sorteer treinen op entry; trein k wacht op
     dep van trein k-1 indien nodig.
  5. Herhaal tot stabiel (geen wijzigingen) of max_iterations bereikt.

De berekening gebruikt dezelfde inputs als de MIP — runtime, dwell,
sched_exit, weights — zodat de vergelijking met solution.objective
zinvol is.
"""

from __future__ import annotations


def compute_fcfs_objective(
    instance: dict,
    max_iterations: int = 50,
) -> tuple[float, bool]:
    """
    Bereken de gewogen som van eindsegment-vertragingen onder FCFS.

    Parameters
    ----------
    instance : dict
        MIP-instance dictionary zoals geproduceerd door build_instance().
    max_iterations : int
        Maximaal aantal vaste-punt iteraties (default 50).

    Returns
    -------
    objective : float
        Gewogen som van eindsegment-vertragingen (zelfde eenheid als
        solution.objective). Returns 0.0 als instance leeg is.
    converged : bool
        True als de vaste-punt iteratie convergeerde binnen max_iterations.
        False signaleert een complexe situatie waar de FCFS-schatting
        onbetrouwbaar is — de caller kan in dat geval beter de
        MIP-oplossing wel toepassen.
    """
    T = instance["T"]
    if not T:
        return 0.0, True

    S              = instance["S"]
    Sl             = instance["Sl"]
    path           = instance["path"]
    runtime        = instance["runtime"]
    dwell          = instance["dwell"]
    halts          = instance["halts"]
    sched_exit     = instance["sched_exit"]
    fixed_entry    = instance["fixed_entry"]
    occupied       = instance["occupied"]
    expected_entry = instance["expected_entry"]
    weights        = instance["weights"]
    current_time   = instance["current_time"]

    # ------------------------------------------------------------------
    # Helper — bereken dep[t,s] uit entry[t,s]
    #
    # C5 (dep >= sched_exit voor halterende treinen) geldt in de MIP
    # ONVOORWAARDELIJK — ook voor segmenten die nu occupied zijn. Een
    # trein die op een station staat te wachten mag niet eerder
    # vertrekken dan gepland, ook al is zijn remaining-tijd al voorbij.
    # ------------------------------------------------------------------
    def _compute_dep(t, s, entry_ts):
        if (t, s) in occupied:
            d = current_time + occupied[(t, s)]
        elif s in Sl:
            d = entry_ts + runtime[(t, s)]
        else:
            d = entry_ts + dwell[(t, s)]

        # C5: halterende treinen mogen niet voor sched_exit vertrekken
        if halts.get((t, s), False):
            d = max(d, sched_exit[(t, s)])
        return d

    # ------------------------------------------------------------------
    # Initialisatie
    #
    # Belangrijk: expected_entry kan een waarde uit een vorige MIP-run
    # bevatten die nu in het verleden ligt (mip_entry_for() retourneert de
    # opgeslagen waarde zonder rekening te houden met current_time). De
    # MIP zelf heeft lb=current_time op zijn entry-variabelen, dus de FCFS
    # baseline moet diezelfde ondergrens respecteren — anders produceren
    # we een infeasible "oplossing" met kunstmatig lage objective.
    #
    # Voor niet-gestarte treinen geldt al expected_entry >= sched_entry
    # (zie instance.py: expected_entry = mip_entry of sched_entry+current_delay),
    # dus de C2b-ondergrens uit de MIP hoeft niet apart geclampt te worden.
    # ------------------------------------------------------------------


    entry: dict = {
        (t, s): fixed_entry.get((t, s), max(current_time, expected_entry[(t, s)]))
        for t in T
        for s in path[t]
    }

    # Pre-compute trains per segment, gesorteerd op entry-tijd (wordt elke iteratie geüpdatet)
    trains_per_segment: dict = {s: [t for t in T if s in path[t]] for s in S}

    # ------------------------------------------------------------------
    # Vaste-punt iteratie
    # ------------------------------------------------------------------
    EPS = 1e-6
    converged = False

    for _ in range(max_iterations):
        changed = False

        # --- C1 + C2: bereken dep en propageer naar volgende segment ---
        dep: dict = {}
        for t in T:
            for k, s in enumerate(path[t]):
                dep[(t, s)] = _compute_dep(t, s, entry[(t, s)])

                # C2: dep[t,s] → entry[t,s_next]
                if k + 1 < len(path[t]):
                    s_next = path[t][k + 1]
                    if (t, s_next) in fixed_entry:
                        continue
                    if dep[(t, s)] > entry[(t, s_next)] + EPS:
                        entry[(t, s_next)] = dep[(t, s)]
                        changed = True

        # --- C4: FCFS conflictresolutie per segment ---
        for s, trains_on_s in trains_per_segment.items():
            if len(trains_on_s) < 2:
                continue

            ordered = sorted(trains_on_s, key=lambda t: entry[(t, s)])

            for k in range(1, len(ordered)):
                prev = ordered[k - 1]
                curr = ordered[k]
                if (curr, s) in fixed_entry:
                    continue
                if entry[(curr, s)] < dep[(prev, s)] - EPS:
                    entry[(curr, s)] = dep[(prev, s)]
                    changed = True

        if not changed:
            converged = True
            break

    # ------------------------------------------------------------------
    # Objective: gewogen som van eindsegment-vertragingen
    # ------------------------------------------------------------------
    objective = 0.0
    for t in T:
        final_seg = path[t][-1]
        final_dep = _compute_dep(t, final_seg, entry[(t, final_seg)])
        delay = max(0.0, final_dep - sched_exit[(t, final_seg)])
        objective += weights[t] * delay

    return objective, converged
