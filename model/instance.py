""" 
instance.py

Builds a reduced MILP instance from the current SystemState.
"""

from config.settings import L, GAMMA, EPSILON, DELTA_MAX, WEIGHT_PASSENGER, WEIGHT_FREIGHT, PSL_PASSENGER, PSL_FREIGHT, RESCHEDULING_HORIZON, CONFLICT_WINDOW
from domain.segment import SegmentType
from domain.train   import TrainType


def build_instance(state, timetable, trains, segments, current_time):

    # =========================================================================
    # STEP 1 — Select relevant trains (delayed + affected + within horizon)
    # =========================================================================

    delayed_trains = []
    for t in trains.values():
        if state.current_delay(t.id) > 0 and not state.is_finished(t.id):
            delayed_trains.append(t)

    delayed_ids = {t.id for t in delayed_trains}

    affected_trains = []
    for t in trains.values():
        if t.id in delayed_ids or state.is_finished(t.id):
            continue
        remaining_t = set(state.remaining_path(t.id))
        for d in delayed_trains:
            if remaining_t & set(state.remaining_path(d.id)):
                affected_trains.append(t)
                break

    # Horizon filter
    relevant_trains = []
    for t in (delayed_trains + affected_trains):
        remaining = state.remaining_path(t.id)
        if not remaining:
            continue
        try:
            planned_start = timetable.scheduled_arrival(t.id, remaining[0])
        except (KeyError, ValueError):
            continue
        if planned_start <= current_time + RESCHEDULING_HORIZON:
            relevant_trains.append(t)

    # Deduplicate
    relevant_trains = list({t.id: t for t in relevant_trains}.values())

    T  = [t.id for t in relevant_trains]
    Tp = [t.id for t in relevant_trains if t.train_type == TrainType.PASSENGER]
    Tf = [t.id for t in relevant_trains if t.train_type == TrainType.FREIGHT]

    # =========================================================================
    # STEP 2 — Remaining paths only
    # =========================================================================

    path = {
        t.id: tuple(state.remaining_path(t.id))
        for t in relevant_trains
    }

    # =========================================================================
    # STEP 3 — In-execution + fix_arrival
    # =========================================================================

    in_execution = {}
    fix_arrival  = {}

    for t in relevant_trains:
        seg = state.current_segment(t.id)

        if seg is None or seg not in path[t.id]:
            continue

        try:
            state.actual_exit(t.id, seg)
            continue
        except KeyError:
            pass

        try:
            entry_time = state.actual_entry(t.id, seg)
        except KeyError:
            continue

        # sched_dep = timetable.scheduled_departure(t.id, seg)
        # if current_time > sched_dep:
        #     continue

        if segments[seg].seg_type == SegmentType.BETWEEN_STATION:
            full_duration = timetable.running_time(t.id, seg)
        else:
            try:
                full_duration = timetable.dwell_time(t.id, seg)
            except ValueError:
                continue

        expected_exit = entry_time + full_duration
        remaining     = expected_exit - current_time

        if remaining <= 0:
            continue

        fix_arrival[(t.id, seg)]  = current_time
        in_execution[(t.id, seg)] = max(1.0, remaining)

    # =========================================================================
    # STEP 4 — Segment sets
    # =========================================================================

    S  = sorted({s for t_id in T for s in path[t_id]})
    Ss = {s for s in S if segments[s].seg_type == SegmentType.STATION}
    Sl = {s for s in S if segments[s].seg_type == SegmentType.BETWEEN_STATION}

    # =========================================================================
    # STEP 5 — Timetable parameters
    # =========================================================================

    sched_entry = {}
    for t in relevant_trains:        # loop over alle relevante treinen
        for s in path[t.id]:         # loop over elk resterend segment van die trein
            sched_entry[(t.id, s)] = timetable.scheduled_arrival(t.id, s)

    sched_dep = {}
    for t in relevant_trains:
        for s in path[t.id]:
            sched_dep[(t.id, s)] = timetable.scheduled_departure(t.id, s)

    RT = {}
    for t in relevant_trains:
        for s in path[t.id]:
            if segments[s].seg_type == SegmentType.BETWEEN_STATION:
                RT[(t.id, s)] = timetable.running_time(t.id, s)

    DW = {}
    for t in relevant_trains:
        for s in path[t.id]:
            if segments[s].seg_type == SegmentType.STATION:
                DW[(t.id, s)] = timetable.dwell_time(t.id, s)

    h_stop = {}
    for t in relevant_trains:
        for s in path[t.id]:
            if segments[s].seg_type == SegmentType.STATION:
                h_stop[(t.id, s)] = timetable.halts_at(t.id, s)

    # =========================================================================
    # STEP 6 — Conflict sets
    # =========================================================================

    def actual_or_sched(t_id, s):
        try:
            return state.actual_entry(t_id, s)
        except KeyError:
            return sched_entry[(t_id, s)]

    C = {}
    for s in S:
        C[s] = []

    for s in S:
        trains_on_s = [t_id for t_id in T if s in path[t_id]]
        for i in range(len(trains_on_s)):
            for j in range(i + 1, len(trains_on_s)):
                t1, t2 = trains_on_s[i], trains_on_s[j]
                e1 = actual_or_sched(t1, s)
                e2 = actual_or_sched(t2, s)
                if abs(e1 - e2) <= CONFLICT_WINDOW:
                    C[s].append((t1, t2))
    # Enkel die paren krijgen een volgorde-constraint in de MIP. Treinparen die ver uit elkaar zitten hoeven niet geordend te worden — dat spaart binaire variabelen.
    # =========================================================================
    # STEP 7 — Headway
    # =========================================================================

    H = {}
    for s in S:
        for (i, j) in C[s]:
            H[(i, j, s)] = 0
            H[(j, i, s)] = 0

    # =========================================================================
    # STEP 8 — Weights
    # =========================================================================

    w = {
        t.id: WEIGHT_PASSENGER if t.train_type == TrainType.PASSENGER else WEIGHT_FREIGHT
        for t in relevant_trains
    }

    psl = {
        t.id: PSL_PASSENGER if t.train_type == TrainType.PASSENGER else PSL_FREIGHT
        for t in relevant_trains
    }

    # =========================================================================
    # RETURN
    # =========================================================================

    return dict(
        T=T, Tp=Tp, Tf=Tf,
        S=S, Ss=Ss, Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_dep=sched_dep,
        RT=RT,
        DW=DW,
        H=H,
        h_stop=h_stop,
        w=w,
        psl=psl,
        L=L,
        gamma=GAMMA,
        epsilon=EPSILON,
        delta_max=DELTA_MAX,
        in_execution=in_execution,
        fix_arrival=fix_arrival,
        current_time = current_time,
        C=C
    )