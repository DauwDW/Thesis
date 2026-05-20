"""
instance.py

Builds a compact MILP rescheduling instance from the current system state.

Terminology:
    entry(t,s) = moment train enters segment s
    exit(t,s)  = moment train leaves segment s

Trein-selectie (STEP 1):


Lower bound semantics (zie mip_base.py):
    fixed_entry[(t,s)]    → lb = ub = current_time   (actief segment)
    actual_entries[(t,s)] → lb = ub = actual_entry   (al betreden, niet actief)
    overige segmenten     → lb = sched_entry          (geen artificiële delay)

Priority weights (STEP 6):
    "static"  → weights[t] = weight_passenger of weight_freight (fixed per type)
    "dynamic" → idem + upgrade_weight bovenop als state.current_delay(t) >= gamma
                (exogene upgrade op basis van observed delay)
"""
# !!! check of je hier beter ipv scheduled exit mip_exit gebruikt

from config.settings import (
    L,
    EPSILON,
    DELTA_MAX,
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


# def planned_entry(timetable, train_id, segment):
#     return timetable.scheduled_entry(train_id, segment)


# # def planned_exit(timetable, segments, train_id, segment):
# #     if segments[segment].seg_type == SegmentType.BETWEEN_STATION:
# #         return (
# #             timetable.scheduled_entry(train_id, segment)
# #             + timetable.running_time(train_id, segment)
# #         )
# #     return (
# #         timetable.scheduled_entry(train_id, segment)
# #         + timetable.dwell_time(train_id, segment)
# #     )


# ============================================================================
# Main
# ============================================================================

def build_instance(
    state,
    timetable,
    trains,
    segments,
    current_time,
    priority_strategy,
    weight_passenger,
    weight_freight,
    upgrade_weight,
    gamma,
):
    if priority_strategy not in ("static", "dynamic"):
        raise ValueError(
            f"Unknown priority_strategy: '{priority_strategy}'. "
            f"Use 'static' or 'dynamic'."
        )

    horizon_end = current_time + RESCHEDULING_HORIZON

    # =========================================================================
    # STEP 1 — Relevant trains
    # =========================================================================

    relevant = []

    for train in trains.values():
        if state.is_finished(train.id): #Skip trains that have already completed their full route
            continue
        remaining = state.remaining_path(train.id)
        if not remaining:
            continue
        if has_started(state, train): #trein die al aan het rijden is sws toevoegen
            relevant.append(train)
            continue
        first_seg  = remaining[0] 
        start_time = timetable.scheduled_entry(train.id, first_seg)
        if start_time <= horizon_end:# treinen die nog niet begonnen zijn maar hun planned entry wel binnen de horizon ligt
            relevant.append(train)

    # =========================================================================
    # STEP 2 — Sets
    # =========================================================================

    T  = [t.id for t in relevant]
    Tp = [t.id for t in relevant if t.train_type == TrainType.PASSENGER]
    Tf = [t.id for t in relevant if t.train_type == TrainType.FREIGHT]

    # Als je het volledige path wil simuleren
    # path = {
    #     t.id: tuple(state.remaining_path(t.id))
    #     for t in relevant
    # }

    # Enkel segmenten die binnen de rescheduling horizon liggen

    path = {}
    for train in relevant:# Loop over every train that was selected in STEP 1
        current_delay = state.current_delay(train.id)
        truncated = [] #Initialize an empty list that will hold the segments to include for this train
        for seg in state.remaining_path(train.id):
            mip_entry = state.mip_entry_for(train.id, seg)
            if mip_entry is not None:
                expected = mip_entry
            else:
                expected = timetable.scheduled_entry(train.id, seg) + current_delay
            truncated.append(seg)
            if expected > horizon_end:
                break
        path[train.id] = tuple(truncated)
    

    S  = sorted({s for t in T for s in path[t]})
    Ss = sorted({s for s in S if segments[s].seg_type == SegmentType.STATION})
    Sl = sorted({s for s in S if segments[s].seg_type == SegmentType.BETWEEN_STATION})

    # =========================================================================
    # STEP 3 — Timing parameters
    # =========================================================================

    sched_entry = {}
    sched_exit  = {}
    runtime     = {}
    dwell       = {}

    for train in relevant:
        for seg in path[train.id]:
            sched_entry[(train.id, seg)] = timetable.scheduled_entry(train.id, seg)
            sched_exit[(train.id, seg)]  = timetable.scheduled_exit(train.id, seg)

            if seg in Sl:
                runtime[(train.id, seg)] = timetable.running_time(train.id, seg)
            if seg in Ss:
                dwell[(train.id, seg)] = timetable.dwell_time(train.id, seg)    # !!! check of dwell_time dezelfde waarde geeft als de simulator gebruikt voor dwell segmets

    halts = {
        (train.id, seg): train.halts_at(seg)
        for train in relevant
        for seg in path[train.id]
        if seg in Ss
    }

    # =========================================================================
    # STEP 4 — Active segments
    #
    # fixed_entry[(t, seg)]:
    #   Huidig actief segment — MIP fixeert entry op current_time.
    #
    # occupied[(t, seg)]:
    #   Resterende bezettingstijd van het actieve segment.
    # =========================================================================

    fixed_entry = {}
    occupied    = {}

    for train in relevant:
        seg = state.current_segment(train.id)

        if seg is None:
            continue
        if seg not in path[train.id]:
            continue

        try:
            actual_entry_time = state.actual_entry(train.id, seg)
        except KeyError:
            continue

        try:
            duration = state.sampled_duration(train.id, seg)
        except KeyError:
            if seg in Sl:
                duration = runtime[(train.id, seg)]
            else:
                duration = dwell[(train.id, seg)]

        remaining_time = (actual_entry_time + duration) - current_time

        fixed_entry[(train.id, seg)] = actual_entry_time
        occupied[(train.id, seg)]    = max(0.0, remaining_time)

        
    # =========================================================================
    # STEP 4b - Warm Start
    # =========================================================================
    expected_exit = {}
    for train in relevant:
        for seg in path[train.id]:
            if (train.id, seg) in occupied:
                expected_exit[(train.id, seg)] = current_time + occupied[(train.id, seg)]
            else:
                expected_exit[(train.id, seg)] = sched_exit[(train.id, seg)]


    # =========================================================================
    # STEP 5 — Conflicts
    # =========================================================================


    # # Alle mogelijke conflicten:
    # conflicts = {s: [] for s in S}

    # for seg in S:
    #     trains_on_seg = [t for t in T if seg in path[t]]
    #     for i in range(len(trains_on_seg)):
    #         for j in range(i + 1, len(trains_on_seg)):
    #             conflicts[seg].append((trains_on_seg[i], trains_on_seg[j]))
        
    # Conflict-window
    expected_entry = {}

    for train in relevant:
        current_delay = state.current_delay(train.id)
        for seg in path[train.id]:
            if (train.id, seg) in fixed_entry:
                expected_entry[(train.id, seg)] = fixed_entry[(train.id, seg)]
            else:
                mip_entry = state.mip_entry_for(train.id, seg)
                if mip_entry is not None:
                    expected_entry[(train.id, seg)] = mip_entry
                else:
                    expected_entry[(train.id, seg)] = sched_entry[(train.id, seg)] + current_delay

    conflicts = {s: [] for s in S}

    for seg in S:
        trains_on_seg = sorted(
            [t for t in T if seg in path[t]],
            key=lambda t: expected_entry[(t, seg)]
        )
        for i in range(len(trains_on_seg)):
            for j in range(i + 1, len(trains_on_seg)):
                t1 = trains_on_seg[i]
                t2 = trains_on_seg[j]
                if expected_entry[(t2, seg)] - expected_entry[(t1, seg)] <= CONFLICT_WINDOW:
                    conflicts[seg].append((t1, t2))
                else:
                    break
    

    

    # =========================================================================
    # STEP 6 — Weights
    #
    # static:  base label per treintype, geen upgrade
    # dynamic: base label + upgrade_weight als current_delay >= gamma
    # =========================================================================

    weights = {}
    n_upgraded = 0

    for train in relevant:
        if train.train_type == TrainType.PASSENGER:
            base = weight_passenger
        else:
            base = weight_freight

        if priority_strategy == "dynamic" and state.current_delay(train.id) >= gamma:
            weights[train.id] = base + upgrade_weight
            n_upgraded += 1
        else:
            weights[train.id] = base



    # # =========================================================================
    # # Diagnostiek
    # # =========================================================================

    # print(f"Treinen in instance: {len(T)}")
    # print(f"  Gestart: {sum(1 for t in relevant if has_started(state, t))}, "
    #       f"Nog niet gestart: {sum(1 for t in relevant if not has_started(state, t))}")
    # delays = [(t.id, state.current_delay(t.id)) for t in relevant]
    # delays.sort(key=lambda x: -x[1])
    # print(f"  Top 5 vertraagd: {delays[:5]}")
    # print(f"  Gemiddelde delay: {sum(d for _, d in delays) / max(1, len(delays)):.0f}s")

    # if priority_strategy == "dynamic":
    #     print(f"  Dynamic upgrades (delay >= {gamma}s): "
    #           f"{n_upgraded}/{len(relevant)} treinen +{upgrade_weight}")

    # # Niet-gestarte treinen met sched_entry in verleden
    # # Check alle treinen in instance op grote implied delays
    # print(f"\n--- Instance diagnostiek t={current_time:.0f}s ---")
    # for train in relevant:
    #     t_id = train.id
    #     remaining_segs = path.get(t_id, ())
    #     if not remaining_segs:
    #         continue
    #     last_seg = remaining_segs[-1]
    #     se_last = sched_entry.get((t_id, last_seg), 0)

    #     # Wat is de effectieve lb voor het eerste segment?
    #     first_seg = remaining_segs[0]
    #     if (t_id, first_seg) in fixed_entry:
    #         lb_first = fixed_entry[(t_id, first_seg)]
    #         kind = "fixed"
    #     elif (t_id, first_seg) in actual_entries:
    #         lb_first = actual_entries[(t_id, first_seg)]
    #         kind = "actual"
    #     else:
    #         lb_first = sched_entry.get((t_id, first_seg), 0)
    #         kind = "sched"

    #     implied_delay = max(0, lb_first - se_last)
    #     if implied_delay > 500:
    #         print(f"  🔴 trein {t_id}: lb_first={lb_first:.0f}s ({kind}) "
    #               f"sched_last={se_last:.0f}s "
    #               f"implied_delay>={implied_delay:.0f}s "
    #               f"n_segs={len(remaining_segs)}")

    #         # ---- Uitgebreide trein-geschiedenis ----
    #         full_path = train.path
    #         first_path_seg = full_path[0]
    #         sched_first    = timetable.scheduled_arrival(t_id, first_path_seg)

    #         # Wat heeft deze trein al gedaan?
    #         done_segs = []
    #         for seg in full_path:
    #             try:
    #                 ae = state.actual_entry(t_id, seg)
    #             except KeyError:
    #                 break
    #             try:
    #                 ax = state.actual_exit(t_id, seg)
    #             except KeyError:
    #                 ax = None
    #             done_segs.append((seg, ae, ax))

    #         # Type + waar staat hij nu?
    #         train_type = train.train_type.value  # 'P' of 'F'
    #         is_synth   = t_id >= 900000

    #         print(f"      type={train_type} synth={is_synth} "
    #               f"sched_first={sched_first:.0f}s "
    #               f"path_len={len(full_path)} done={len(done_segs)}")

    #         if done_segs:
    #             first_done = done_segs[0]
    #             last_done  = done_segs[-1]
    #             # Vertraging op het eerste segment ten opzichte van planning
    #             first_planned = timetable.scheduled_arrival(t_id, first_done[0])
    #             start_delay   = first_done[1] - first_planned
    #             print(f"      eerste_segment: {first_done[0][:40]} "
    #                   f"actual_entry={first_done[1]:.0f}s "
    #                   f"sched_entry={first_planned:.0f}s "
    #                   f"start_delay={start_delay:.0f}s")
    #             print(f"      laatste_voltooide: {last_done[0][:40]} "
    #                   f"entry={last_done[1]:.0f}s "
    #                   f"exit={last_done[2]}")
    #         else:
    #             # Trein nog niet gestart maar zit toch in 'fixed' — dat zou niet moeten
    #             print(f"      ⚠ geen actual_entry maar kind={kind}")
    #     # ---- Bottleneck-analyse ----
    # print("\n--- Bottleneck candidates ---")
    # stuck_count = {}
    # for train in relevant:
    #     t_id = train.id
    #     cur_seg = state.current_segment(t_id)
    #     if cur_seg is None:
    #         continue
    #     try:
    #         ae = state.actual_entry(t_id, cur_seg)
    #     except KeyError:
    #         continue
        
    #     # Schatting van fysieke duur op huidig segment
    #     if cur_seg in Sl:
    #         phys_dur = runtime.get((t_id, cur_seg), 0)
    #     else:
    #         phys_dur = dwell.get((t_id, cur_seg), 0)
        
    #     waited = current_time - ae
    #     if waited > phys_dur + 60:  # meer dan 60s langer dan zou moeten
    #         # Wat is zijn next_seg?
    #         try:
    #             idx = train.path.index(cur_seg)
    #             if idx + 1 < len(train.path):
    #                 next_seg = train.path[idx + 1]
    #                 stuck_count[next_seg] = stuck_count.get(next_seg, 0) + 1
    #         except ValueError:
    #             pass

    # # Top 5 bottleneck segmenten
    # top = sorted(stuck_count.items(), key=lambda x: -x[1])[:5]
    # for seg, n in top:
    #     print(f"  {n} treinen wachten op: {seg[:60]}")
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
        runtime=runtime,
        halts = halts,
        dwell=dwell,
        sched_exit=sched_exit,
        expected_exit=expected_exit,
        occupied=occupied,
        fixed_entry=fixed_entry,
        conflicts=conflicts,
        weights=weights,
        L=L,
        current_time=current_time,
    )