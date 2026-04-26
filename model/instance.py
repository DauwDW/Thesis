"""
instance.py

Translates the current SystemState into the parameter sets
required by the MILP model (mip_base.py / mip_dynamic.py).

Responsible for:
- Filtering to only relevant trains (delayed or affected)
- Trimming each train's path to remaining segments only
- Building conflict sets C_s
- Computing headway parameters H based on train type combinations
"""
from config.settings import L, GAMMA, EPSILON, DELTA_MAX
from domain.segment import SegmentType

# Headway lookup based on train type combinations (in seconds)
HEADWAY_TABLE = {
    ("P", "P"): 180,   # passenger following passenger
    ("P", "F"): 240,   # freight following passenger
    ("F", "P"): 300,   # passenger following freight 
    ("F", "F"): 240,   # freight following freight ALLEMAAL NOG TE BEPALEN   
}


def get_headway(type_i, type_j):
    """
    Returns the required headway when train j follows train i on a segment.
    type_i, type_j: 'P' (passenger) or 'F' (freight)
    """
    return HEADWAY_TABLE.get((type_i, type_j), 180)  # default to 180 if not found


# Main function
def build_instance(state, timetable, trains, segments, current_time): #state komt uit de simulatie
    """
    Builds the MILP parameter sets from the current SystemState.

    Parameters
    ----------
    state     : SystemState   — current simulation state (positions, delays, current time)
    timetable : Timetable     — original scheduled arrival/departure times
    trains    : list[Train]   — all Train objects
    segments  : list[Segment] — all Segment objects
    current_time : float      — current simulation time in seconds

    Returns
    -------
    A dictionary with all MILP parameters ready to pass into build_and_solve_model()
    """

   # Step 1 — Find delayed trains
    delayed_trains = [
        t for t in trains.values()
        if state.current_delay(t.id) > 0
        and not state.is_finished(t.id)]

    # Step 2 — Find affected trains (remaining path overlaps with delayed train)
    delayed_ids = set(t.id for t in delayed_trains)

    affected_trains = []
    for t in trains:
        if t.id in delayed_ids:
            continue  # already included
        if state.is_finished(t.id):
            continue  # train already completed its journey

        remaining_t = state.remaining_path(t.id)

        for d in delayed_trains:
            remaining_d = state.remaining_path(d.id)
            if set(remaining_t) & set(remaining_d):  # overlap exists
                affected_trains.append(t)
                break

   # Step 3 — Build relative train set T (delayed + affected, not yet finished)
    relevant_trains = [
        t for t in delayed_trains + affected_trains
        if not state.is_finished(t.id)]

    T  = [t.id for t in relevant_trains]
    Tp = [t.id for t in relevant_trains if t.train_type == "P"]
    Tf = [t.id for t in relevant_trains if t.train_type == "F"]

    # Step 4 — Build remaining paths per train
    path = {t.id: state.remaining_path(t.id) for t in relevant_trains}
    # Step 4b — Detect in-execution operations 
    in_execution = {}
    fix_arrival  = {}
 
    for t in relevant_trains:
        # TODO: NAAM CHECKEN MET DAUW — state.current_segment(t.id)
        seg = state.current_segment(t.id)
 
        if seg is None:
            continue  # train is not currently mid-segment, nothing to fix
 
        if seg not in path[t.id]:
            continue  # segment already completed, not in remaining path
 
        # How long has the train already been on this segment?
        actual_arrival_time = state.actual_entry(t.id, seg)
        elapsed = current_time - actual_arrival_time
 
        # Full planned duration on this segment (running time or dwell time)
        if segments[seg].seg_type == SegmentType.BETWEEN_STATION:
            full_duration = timetable.running_time(t.id, seg)
        else:
            full_duration = timetable.dwell_time(t.id, seg)

        # Remaining time = full duration minus what has already elapsed
        # Minimum of 1 to avoid zero-duration operations in the MILP
        remaining = max(1, full_duration - elapsed)
 
        in_execution[t.id, seg] = remaining
        fix_arrival[t.id, seg]  = current_time  # arrival is now, cannot be moved        
 

    # Step 5 — Build segment sets
    all_segs = set(s for segs in path.values() for s in segs)

    S  = list(all_segs)
    Ss = set(s for s in S if segments[s].seg_type == SegmentType.STATION)
    Sl = set(s for s in S if segments[s].seg_type == SegmentType.BETWEEN_STATION)

    # Step 6 — Scheduled times from timetable (never change)
    sched_entry = {
        (t.id, s): timetable.scheduled_arrival(t.id, s)
        for t in relevant_trains for s in path[t.id]}
    sched_dep = {
        (t.id, s): timetable.scheduled_departure(t.id, s)
        for t in relevant_trains for s in path[t.id]}

    # Step 7 — Running times and dwell times
    RT = {
        (t.id, s): timetable.running_time(t.id, s)
        for t in relevant_trains for s in path[t.id]
        if segments[s].seg_type == SegmentType.BETWEEN_STATION}
    DW = {
        (t.id, s): timetable.dwell_time(t.id, s)
        for t in relevant_trains for s in path[t.id]
        if segments[s].seg_type == SegmentType.STATION}

    # Step 8 — Halt indicators (does train stop at this station?)
    h_stop = {
        (t.id, s): timetable.halts_at(t.id, s)
        for t in relevant_trains for s in path[t.id]
        if segments[s].seg_type == SegmentType.STATION}

    # Step 9 — Headway parameters H based on train type combinations
    train_type = {t.id: t.train_type for t in relevant_trains}

    C = {}
    for s in S:
        trains_on_s = [t_id for t_id in T if s in path[t_id]]
        C[s] = [
            (trains_on_s[a], trains_on_s[b])
            for a in range(len(trains_on_s))
            for b in range(a + 1, len(trains_on_s))]

    H = {}
    for s in S:
        for (i, j) in C[s]:
            H[i, j, s] = get_headway(train_type[i], train_type[j])
            H[j, i, s] = get_headway(train_type[j], train_type[i])

   # Step 10 — Priority weights (static: based on train type) passenger bvb dubbel zo belangrijk als freight
    w = {
        t.id: 2 if t.train_type == "P" else 1
        for t in relevant_trains}
    psl = {
        t.id: 1 if t.train_type == "P" else 0
        for t in relevant_trains}

    # Return all parameters as a dictionary
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
        fix_arrival=fix_arrival)