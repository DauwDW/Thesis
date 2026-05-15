
"""
instance.py

Builds a compact MILP rescheduling instance from the current system state.

Terminology:
    entry(t,s) = moment train enters segment s
    exit(t,s)  = moment train leaves segment s

Trein-selectie (STEP 1):
    Delayed treinen die al gestart zijn: altijd includeren.
    Delayed treinen die nog niet gestart zijn: alleen als sched_entry
      van het eerste resterende segment nog in [current_time, horizon_end]
      valt. Als de starttijd in het verleden ligt, kan de MIP de entry
      niet meer beïnvloeden — uitsluiten voorkomt artificiële
      objective-explosie via de continuity constraint.
    Affected treinen: zelfde logica.

Lower bound semantics (zie mip_base.py):
    fixed_entry[(t,s)]    → lb = ub = current_time   (actief segment)
    actual_entries[(t,s)] → lb = ub = actual_entry   (al betreden, niet actief)
    overige segmenten     → lb = sched_entry          (geen artificiële delay)
"""

from config.settings import (
    L,
    EPSILON,
    DELTA_MAX,
    PSL_PASSENGER,
    PSL_FREIGHT,
    RESCHEDULING_HORIZON,
    CONFLICT_WINDOW,
)

from domain.segment import SegmentType
from domain.train import TrainType


# ============================================================================
# Helpers
# ============================================================================

def has_started(state, train) -> bool:
    try:
        state.actual_entry(train.id, train.path[0])
        return True
    except KeyError:
        return False


def planned_entry(timetable, train_id, segment):
    return timetable.scheduled_arrival(train_id, segment)


def planned_exit(timetable, segments, train_id, segment):
    if segments[segment].seg_type == SegmentType.BETWEEN_STATION:
        return (
            timetable.scheduled_arrival(train_id, segment)
            + timetable.running_time(train_id, segment)
        )
    return (
        timetable.scheduled_arrival(train_id, segment)
        + timetable.dwell_time(train_id, segment)
    )


# ============================================================================
# Main
# ============================================================================

def build_instance(
    state,
    timetable,
    trains,
    segments,
    current_time,
    weight_passenger,
    weight_freight,
    gamma,
):

    horizon_end = current_time + RESCHEDULING_HORIZON

    # =========================================================================
    # STEP 1 — Relevant trains
    # =========================================================================

    delayed = []

    for train in trains.values():
        if state.is_finished(train.id):
            continue
        if state.current_delay(train.id) > 0:
            delayed.append(train)

    delayed_ids = {t.id for t in delayed}

    affected = []

    for train in trains.values():
        if train.id in delayed_ids:
            continue
        if state.is_finished(train.id):
            continue
        remaining = set(state.remaining_path(train.id))
        for d in delayed:
            if remaining & set(state.remaining_path(d.id)):
                affected.append(train)
                break

    relevant = []

    # --- delayed trains ---
    for train in delayed:
        remaining = state.remaining_path(train.id)
        if not remaining:
            continue

        if has_started(state, train):
            relevant.append(train)
            continue

        # Nog niet gestart: alleen als starttijd nog in de toekomst ligt
        first_seg  = remaining[0]
        start_time = planned_entry(timetable, train.id, first_seg)

        if current_time <= start_time <= horizon_end:
            relevant.append(train)

    # --- affected trains ---
    for train in affected:
        remaining = state.remaining_path(train.id)
        if not remaining:
            continue

        if has_started(state, train):
            relevant.append(train)
            continue

        first_seg  = remaining[0]
        start_time = planned_entry(timetable, train.id, first_seg)

        if current_time <= start_time <= horizon_end:
            relevant.append(train)

    # deduplicate
    relevant = list({t.id: t for t in relevant}.values())

    # =========================================================================
    # STEP 2 — Sets
    # =========================================================================

    T  = [t.id for t in relevant]
    Tp = [t.id for t in relevant if t.train_type == TrainType.PASSENGER]
    Tf = [t.id for t in relevant if t.train_type == TrainType.FREIGHT]

    path = {
        t.id: tuple(state.remaining_path(t.id))
        for t in relevant
    }

    S  = sorted({s for t in T for s in path[t]})
    Ss = {s for s in S if segments[s].seg_type == SegmentType.STATION}
    Sl = {s for s in S if segments[s].seg_type == SegmentType.BETWEEN_STATION}

    # =========================================================================
    # STEP 3 — Timing parameters
    # =========================================================================

    sched_entry = {}
    sched_exit  = {}
    runtime     = {}
    dwell       = {}

    for train in relevant:
        for seg in path[train.id]:

            sched_entry[(train.id, seg)] = (
                timetable.scheduled_arrival(train.id, seg)
            )
            sched_exit[(train.id, seg)] = (
                planned_exit(timetable, segments, train.id, seg)
            )

            if seg in Sl:
                runtime[(train.id, seg)] = timetable.running_time(train.id, seg)
            if seg in Ss:
                dwell[(train.id, seg)]   = timetable.dwell_time(train.id, seg)

    # =========================================================================
    # STEP 4 — Active segments + actual entries
    #
    # fixed_entry[(t, seg)]:
    #   Huidig actief segment — MIP fixeert entry op current_time.
    #
    # occupied[(t, seg)]:
    #   Resterende bezettingstijd van het actieve segment.
    #
    # actual_entries[(t, seg)]:
    #   Segmenten die de trein al betreden heeft maar nog in path[t] zitten.
    #   MIP fixeert entry op de werkelijke entrytijd — ligt in het verleden
    #   en kan niet meer aangepast worden.
    # =========================================================================

    fixed_entry    = {}
    occupied       = {}
    actual_entries = {}

    for train in relevant:
        seg = state.current_segment(train.id)

        if seg is None:
            continue
        if seg not in path[train.id]:
            continue

        try:
            state.actual_exit(train.id, seg)
            continue
        except KeyError:
            pass

        try:
            actual_entry_time = state.actual_entry(train.id, seg)
        except KeyError:
            continue

        if seg in Sl:
            duration = runtime[(train.id, seg)]
        else:
            duration = dwell[(train.id, seg)]

        remaining_time = (actual_entry_time + duration) - current_time

        fixed_entry[(train.id, seg)] = current_time
        occupied[(train.id, seg)]    = max(0.0, remaining_time)

    # Segmenten die al betreden zijn maar niet het actieve segment
    for train in relevant:
        for seg in path[train.id]:
            if (train.id, seg) in fixed_entry:
                continue
            try:
                ae = state.actual_entry(train.id, seg)
                actual_entries[(train.id, seg)] = ae
            except KeyError:
                pass
    

    # =========================================================================
    # STEP 5 — Conflicts
    # =========================================================================

    def actual_or_sched_entry(train_id, seg):
        try:
            return state.actual_entry(train_id, seg)
        except KeyError:
            return sched_entry[(train_id, seg)]

    conflicts = {s: [] for s in S}

    for seg in S:
        trains_on_seg = [t for t in T if seg in path[t]]
        for i in range(len(trains_on_seg)):
            for j in range(i + 1, len(trains_on_seg)):
                t1 = trains_on_seg[i]
                t2 = trains_on_seg[j]
                e1 = actual_or_sched_entry(t1, seg)
                e2 = actual_or_sched_entry(t2, seg)
                if abs(e1 - e2) <= CONFLICT_WINDOW:
                    conflicts[seg].append((t1, t2))

    # =========================================================================
    # STEP 6 — Weights
    # =========================================================================

    weights = {}
    psl     = {}

    for train in relevant:
        current_delay = state.current_delay(train.id)

        if train.train_type == TrainType.PASSENGER:
            upgrade           = 1 if current_delay >= gamma else 0
            weights[train.id] = weight_passenger + upgrade
            psl[train.id]     = PSL_PASSENGER + upgrade
        else:
            weights[train.id] = weight_freight
            psl[train.id]     = PSL_FREIGHT

    # =========================================================================
    # Diagnostiek
    # =========================================================================

    print(f"Treinen in instance: {len(T)}")
    print(f"  Gestart: {sum(1 for t in relevant if has_started(state, t))}, "
          f"Nog niet gestart: {sum(1 for t in relevant if not has_started(state, t))}")
    delays = [(t.id, state.current_delay(t.id)) for t in relevant]
    delays.sort(key=lambda x: -x[1])
    print(f"  Top 5 vertraagd: {delays[:5]}")
    print(f"  Gemiddelde delay: {sum(d for _, d in delays) / max(1, len(delays)):.0f}s")

    # Niet-gestarte treinen met sched_entry in verleden
    # Check alle treinen in instance op grote implied delays
    print(f"\n--- Instance diagnostiek t={current_time:.0f}s ---")
    for train in relevant:
        t_id = train.id
        remaining_segs = path.get(t_id, ())
        if not remaining_segs:
            continue
        last_seg = remaining_segs[-1]
        se_last = sched_entry.get((t_id, last_seg), 0)
        
        # Wat is de effectieve lb voor het eerste segment?
        first_seg = remaining_segs[0]
        if (t_id, first_seg) in fixed_entry:
            lb_first = fixed_entry[(t_id, first_seg)]
            kind = "fixed"
        elif (t_id, first_seg) in actual_entries:
            lb_first = actual_entries[(t_id, first_seg)]
            kind = "actual"
        else:
            lb_first = sched_entry.get((t_id, first_seg), 0)
            kind = "sched"
        
        implied_delay = max(0, lb_first - se_last)
        if implied_delay > 500:
            print(f"  🔴 trein {t_id}: lb_first={lb_first:.0f}s ({kind}) "
                f"sched_last={se_last:.0f}s "
                f"implied_delay>={implied_delay:.0f}s "
                f"n_segs={len(remaining_segs)}")

    # =========================================================================
    # RETURN
    # =========================================================================

    return dict(
        T=T,
        Tp=Tp,
        Tf=Tf,
        S=S,
        Ss=Ss,
        Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_exit=sched_exit,
        runtime=runtime,
        dwell=dwell,
        occupied=occupied,
        fixed_entry=fixed_entry,
        actual_entries=actual_entries,
        conflicts=conflicts,
        weights=weights,
        psl=psl,
        L=L,
        epsilon=EPSILON,
        delta_max=DELTA_MAX,
        gamma=gamma,
        current_time=current_time,
    )