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


# =============================================================================
# Headway lookup based on train type combinations (in minutes)
# Adjust these values to match your domain knowledge / NMBS data
# =============================================================================
HEADWAY_TABLE = {
    ("P", "P"): 3,   # passenger following passenger
    ("P", "F"): 4,   # freight following passenger
    ("F", "P"): 5,   # passenger following freight 
    ("F", "F"): 4,   # freight following freight ALLEMAAL NOG TE BEPALEN   
}


def get_headway(type_i, type_j):
    """
    Returns the required headway when train j follows train i on a segment.
    type_i, type_j: 'P' (passenger) or 'F' (freight)
    """
    return HEADWAY_TABLE.get((type_i, type_j), 3)  # default to 3 if not found


# =============================================================================
# Main function
# =============================================================================

def build_instance(state, timetable, trains, segments): #state komt uit de simulatie
    """
    Builds the MILP parameter sets from the current SystemState.

    Parameters
    ----------
    state     : SystemState   — current simulation state (positions, delays, current time)
    timetable : Timetable     — original scheduled arrival/departure times
    trains    : list[Train]   — all Train objects
    segments  : list[Segment] — all Segment objects

    Returns
    -------
    A dictionary with all MILP parameters ready to pass into build_and_solve_model()
    """

    # -------------------------------------------------------------------------
    # Step 1 — Find delayed trains
    # -------------------------------------------------------------------------
    delayed_trains = [
        t for t in trains
        if state.current_delay(t.id) > 0
        and not state.is_finished(t.id)
    ]

    # -------------------------------------------------------------------------
    # Step 2 — Find affected trains (remaining path overlaps with delayed train)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 3 — Build relative train set T (delayed + affected, not yet finished)
    # -------------------------------------------------------------------------
    relevant_trains = [
        t for t in delayed_trains + affected_trains
        if not state.is_finished(t.id)
    ]

    T  = [t.id for t in relevant_trains]
    Tp = [t.id for t in relevant_trains if t.train_type == "P"]
    Tf = [t.id for t in relevant_trains if t.train_type == "F"]

    # -------------------------------------------------------------------------
    # Step 4 — Build remaining paths per train
    # -------------------------------------------------------------------------
    path = {t.id: state.remaining_path(t.id) for t in relevant_trains}

    # -------------------------------------------------------------------------
    # Step 5 — Build segment sets
    # -------------------------------------------------------------------------
    all_segs = set(s for segs in path.values() for s in segs)

    S  = list(all_segs)
    Ss = [s for s in S if segments[s].seg_type == "station"]
    Sl = [s for s in S if segments[s].seg_type == "line"]

    # -------------------------------------------------------------------------
    # Step 6 — Scheduled times from timetable (never change)
    # -------------------------------------------------------------------------
    sched_entry = {
        (t.id, s): timetable.scheduled_arrival(t.id, s)
        for t in relevant_trains for s in path[t.id]
    }
    sched_dep = {
        (t.id, s): timetable.scheduled_departure(t.id, s)
        for t in relevant_trains for s in path[t.id]
    }

    # -------------------------------------------------------------------------
    # Step 7 — Running times and dwell times
    # -------------------------------------------------------------------------
    RT = {
        (t.id, s): timetable.running_time(t.id, s)
        for t in relevant_trains for s in path[t.id]
        if segments[s].seg_type == "line"
    }
    DW = {
        (t.id, s): timetable.dwell_time(t.id, s)
        for t in relevant_trains for s in path[t.id]
        if segments[s].seg_type == "station"
    }

    # -------------------------------------------------------------------------
    # Step 8 — Halt indicators (does train stop at this station?)
    # -------------------------------------------------------------------------
    h_stop = {
        (t.id, s): timetable.halts_at(t.id, s)
        for t in relevant_trains for s in path[t.id]
        if segments[s].seg_type == "station"
    }

    # -------------------------------------------------------------------------
    # Step 9 — Headway parameters H based on train type combinations
    # -------------------------------------------------------------------------
    train_type = {t.id: t.train_type for t in relevant_trains}

    C = {}
    for s in S:
        trains_on_s = [t_id for t_id in T if s in path[t_id]]
        C[s] = [
            (trains_on_s[a], trains_on_s[b])
            for a in range(len(trains_on_s))
            for b in range(a + 1, len(trains_on_s))
        ]

    H = {}
    for s in S:
        for (i, j) in C[s]:
            H[i, j, s] = get_headway(train_type[i], train_type[j])
            H[j, i, s] = get_headway(train_type[j], train_type[i])

    # -------------------------------------------------------------------------
    # Step 10 — Priority weights (static: based on train type) passenger dubbel zo belangrijk als freight
    # -------------------------------------------------------------------------
    w = {
        t.id: 2 if t.train_type == "P" else 1
        for t in relevant_trains
    }

    # -------------------------------------------------------------------------
    # Return all parameters as a dictionary
    # -------------------------------------------------------------------------
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
    )