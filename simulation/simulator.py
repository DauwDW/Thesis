from __future__ import annotations

import logging

import numpy as np

from simulation.dispatcher import Dispatcher
from simulation.event_queue import EventQueue, TrainEntered, TrainReadyToExit
from simulation.state import SystemState

logger = logging.getLogger(__name__)


# Objective-drempel waarboven een run als deadlock wordt beschouwd.
# Wanneer de MIP-objective deze waarde overschrijdt, gooit de simulator
# een DeadlockDetected exception. run_simulation.py vangt deze op en
# markeert de run als incomplete. Zie thesis §X voor motivatie.
_DEADLOCK_OBJECTIVE_THRESHOLD = 6000000000.0
_DEADLOCK_TIME_LIMIT = 150000
_DEADLOCK_CONSECUTIVE_FAILURES = 2


class DeadlockDetected(RuntimeError):
    """
    Raised wanneer de MIP-objective de deadlock-drempel overschrijdt.

    Dit wijst op een circulaire blokkering in de bloksectie-simulatie —
    een bekende beperking van enkelvoudige bloksecties op bidirectionele
    corridors (zie thesis §X). De run wordt als incomplete gemarkeerd.
    """
    pass


class Simulator:
    """
    Discrete-event simulator voor treinvertragingen op de Brusselse corridor.

    Volgorde-principe:
      De dispatcher beheert alleen bezetting — geen prioriteitswachtrij.
      Volgorde op conflicterende segmenten is FCFS op event-tijd.
      De MIP stuurt volgorde via mip_schedule in SystemState: TrainReadyToExit-
      en TrainEntered-events worden gepusht op max(MIP-tijd, phys_ready,
      current_time).

    Resource queueing:
      Bij blokkering komt een trein in de waiting-list van het volgende
      segment (Dispatcher._waiting). Er wordt geen retry-event gepusht;
      de dispatcher wekt de hoogste-prioriteitswachter zodra het segment
      vrijkomt (zie _wake_next). Prioriteit binnen de queue is gebaseerd
      op mip_entry uit SystemState, met FIFO als fallback.
    """
#!!!!! verwijder hierna strict_order
    def __init__(self, trains, segments, timetable, controller, seed=None, queue_mode: str = "fsfs"):
        self._trains = trains
        self._segments = segments
        self._timetable = timetable
        self._controller = controller
        self._rng = np.random.default_rng(seed)
        self._state = SystemState(trains=trains, timetable=timetable, start_time=0.0)
        self._queue = EventQueue()
        self._dispatcher = Dispatcher(timetable=timetable, segments=segments, trains=trains, queue_mode=queue_mode)
        self._solutions = []

    # ------------------------------------------------------------------
    # Publieke interface
    # ------------------------------------------------------------------

    def run(self):
        self._initialise()
        while self._queue:
            event = self._queue.pop()
            self._state.advance_time(event.time)
            # Deadlock detectie
            if self._state.current_time > _DEADLOCK_TIME_LIMIT:
                raise DeadlockDetected(
                    f"t={self._state.current_time:.0f}s — vermoedelijke deadlock."
                )
        
            if isinstance(event, TrainEntered):
                if not self._is_stale_entered(event):   #misschien overbodig --> dubbelcheck
                    self._handle_entered(event)
            elif isinstance(event, TrainReadyToExit):
                if not self._is_stale_ready_to_exit(event):  #misschien overbodig --> dubbelcheck
                    self._handle_ready_to_exit(event)
            else:
                raise TypeError(type(event))
                # Eindcontrole: alle treinen moeten hun eindsegment bereikt en verlaten hebben.
        unfinished = [
            train_id for train_id in self._trains
            if not self._state.is_finished(train_id)
        ]
        if unfinished:
            preview = ", ".join(str(t) for t in unfinished[:10])
            suffix = "" if len(unfinished) <= 10 else f", ... (+{len(unfinished) - 10})"
            not_started = [t for t in unfinished if not self._state._actual[t]]
            in_waiting  = {seg: list(w) for seg, w in self._dispatcher._waiting.items() if w}
            occupied    = {s: o for s, o in self._dispatcher._occupied.items() if o is not None}
            print(f"[DEADLOCK] niet gestart: {len(not_started)}/{len(unfinished)}")
            print(f"[DEADLOCK] bezette segmenten op eindtijd: {occupied}")
            print(f"[DEADLOCK] waiting lists: {sum(len(w) for w in in_waiting.values())} treinen verdeeld over {len(in_waiting)} segmenten")
            raise DeadlockDetected(
                f"Event-queue leeg op t={self._state.current_time:.0f}s, maar "
                f"{len(unfinished)} trein(en) bereikten hun eindsegment niet: "
                f"[{preview}{suffix}] — vermoedelijke deadlock."
            )
        return self._state
    
       # ------------------------------------------------------------------
    # Initialisatie
    # ------------------------------------------------------------------

    def _initialise(self):
        for train_id in sorted(self._trains):
            train = self._trains[train_id]
            first_seg = train.first_segment
            entry_time = self._timetable.scheduled_entry(train_id, first_seg)
            self._queue.push(TrainEntered(
                time=entry_time,
                train_id=train_id,
                segment_id=first_seg,
            ))

    # ------------------------------------------------------------------
    # Stale-event checks
    # ------------------------------------------------------------------ 

    def _is_stale_entered(self, event):
        if self._state.is_finished(event.train_id):
            return True
        if self._state.current_segment(event.train_id) is not None:
            return True
        remaining = self._state.remaining_path(event.train_id)
        return not remaining or remaining[0] != event.segment_id

    def _is_stale_ready_to_exit(self, event):
        return self._state.current_segment(event.train_id) != event.segment_id

 

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_entered(self, event: TrainEntered) -> None:
        current_time = event.time       # Overbodig want al in run() toch?

        # ==========================================================
        # DIAGNOSTIC:
        # detect REAL artificial waiting
        # ==========================================================

        # ----------------------------------------------------------
        # 1. Zoek vorige segment van deze trein
        # ----------------------------------------------------------

        prev_seg = self._previous_segment(
            event.train_id,
            event.segment_id
        )

        prev_exit = None

        if prev_seg is not None:

            try:
                prev_exit = self._state.actual_exit(
                    event.train_id,
                    prev_seg
                )
            except KeyError:
                pass

        # ----------------------------------------------------------
        # 2. Zoek wanneer segment effectief vrij werd
        # ----------------------------------------------------------

        segment_free_since = None

        occupied_by = self._dispatcher._occupied[event.segment_id]

        if occupied_by is None:

            latest_exit = -float("inf")

            for train_id in self._trains:

                try:
                    exit_time = self._state.actual_exit(
                        train_id,
                        event.segment_id
                    )

                    latest_exit = max(latest_exit, exit_time)

                except KeyError:
                    pass

            if latest_exit > -float("inf"):
                segment_free_since = latest_exit

        # ----------------------------------------------------------
        # 3. Bereken earliest feasible entry
        # ----------------------------------------------------------

        candidates = []

        if prev_exit is not None:
            candidates.append(prev_exit)

        if segment_free_since is not None:
            candidates.append(segment_free_since)

        if prev_exit is not None and candidates:

            earliest_feasible = max(candidates)

            artificial_wait = event.time - earliest_feasible

            # alleen loggen als echt substantieel
            if artificial_wait > 1:

                print(
                    f"[ARTIFICIAL WAIT] "
                    f"train={event.train_id} "
                    f"segment={event.segment_id} "
                    f"prev_exit={prev_exit} "
                    f"segment_free_since={segment_free_since} "
                    f"earliest_feasible={earliest_feasible:.1f} "
                    f"scheduled_entry={event.time:.1f} "
                    f"extra_wait={artificial_wait:.1f}s"
                )

        ### einde diagnostiek !!!

        if not self._dispatcher.request_entry(
            event.train_id, event.segment_id, current_time, state=self._state,
        ):
            # In de waiting-list gezet door request_entry — dispatcher wekt
            # ons via _wake_next zodra het segment vrijkomt.
            return

        self._dispatcher.confirm_entry(event.train_id, event.segment_id)
        self._state.record_entry(
            train_id=event.train_id,
            segment_id=event.segment_id,
            time=current_time,
        )
        ready_time = self._compute_ready_time(event.train_id, event.segment_id, current_time)
        self._queue.push(TrainReadyToExit(
            time=ready_time,
            train_id=event.train_id,
            segment_id=event.segment_id,
        ))

    def _handle_ready_to_exit(self, event: TrainReadyToExit) -> None:
        current_time = event.time
        next_seg = self._next_segment(event.train_id, event.segment_id)

        # --- terminaal segment ---
        if next_seg is None:
            self._dispatcher.release(event.train_id, event.segment_id)
            self._state.record_exit(
                train_id=event.train_id,
                segment_id=event.segment_id,
                time=current_time,
            )
            self._wake_next(event.segment_id, current_time)
            result = self._controller.step(self._state, current_time)
            self._apply_result(result)
            return

        # --- geblokkeerd ---
        if not self._dispatcher.request_entry(
            event.train_id, next_seg, current_time, state=self._state,
        ):
            # In de waiting-list van next_seg — dispatcher wekt ons.
            result = self._controller.step(self._state, current_time)
            self._apply_result(result)
            return

        # --- atomaire transfer ---
        self._dispatcher.release(event.train_id, event.segment_id)
        self._state.record_exit(
            train_id=event.train_id,
            segment_id=event.segment_id,
            time=current_time,
        )
        self._dispatcher.confirm_entry(event.train_id, next_seg)
        self._state.record_entry(
            train_id=event.train_id,
            segment_id=next_seg,
            time=current_time,
        )
        ready_time = self._compute_ready_time(event.train_id, next_seg, current_time)
        self._queue.push(TrainReadyToExit(
            time=ready_time,
            train_id=event.train_id,
            segment_id=next_seg,
        ))
        self._wake_next(event.segment_id, current_time)
        result = self._controller.step(self._state, current_time)
        self._apply_result(result)

    # ------------------------------------------------------------------
    # Wake-up van wachters in dispatcher-queue
    # ------------------------------------------------------------------

    def _wake_next(self, segment_id: str, current_time: float) -> None:
        """
        Wek de hoogste-prioriteits-wachter op een net-vrijgekomen segment.

        De wachter zit nog steeds op zijn vorige segment (current_segment),
        tenzij hij nog niet gestart is — dan wachtte hij op zijn first_segment
        via een TrainEntered-event. We pushen het juiste event-type op
        current_time, zodat de wachter onmiddellijk opnieuw request_entry doet.
        """
        next_id = self._dispatcher.next_waiter(segment_id, state=self._state, current_time=current_time)
        if next_id is None:
            return

        current_seg = self._state.current_segment(next_id)

        if current_seg is None:
            self._queue.cancel_train_entered(next_id, segment_id)
            self._queue.push(TrainEntered(
                time=current_time, train_id=next_id, segment_id=segment_id,
            ))
        else:
            self._queue.cancel_ready_to_exit(next_id, current_seg)
            self._queue.push(TrainReadyToExit(
                time=current_time, train_id=next_id, segment_id=current_seg,
            ))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Controller resultaat verwerken
    # ------------------------------------------------------------------

    def _apply_result(self, result) -> None:
        if result.action == "rescheduled":
            if (result.solution.objective is not None and
                    result.solution.objective > _DEADLOCK_OBJECTIVE_THRESHOLD):
                raise DeadlockDetected(
                    f"MIP-objective {result.solution.objective:.0f} overschrijdt "
                    f"drempel {_DEADLOCK_OBJECTIVE_THRESHOLD:.0f} op "
                    f"t={self._state.current_time:.0f}s — vermoedelijke deadlock."
                )
            self._solutions.append(result.solution)
            self._apply_solution(result.solution)
            self._dispatcher.notify_reschedule(self._state.current_time)

        if self._controller.consecutive_failures >= _DEADLOCK_CONSECUTIVE_FAILURES:
            raise DeadlockDetected(
                f"Solver gaf {self._controller.consecutive_failures} opeenvolgende "
                f"infeasible/unknown resultaten op t={self._state.current_time:.0f}s "
                f"— vermoedelijke deadlock."
            )

    def _apply_solution(self, solution) -> None:
        """
        Verwerk een MIP-oplossing: sla het volledige plan op in SystemState
        en herplan het direct volgende event per trein.

        Voor elke trein in solution.arrival:
          1. Sla complete (entry, departure) schedule op via record_mip_schedule.
          2. Push één event op basis van huidige positie:
             - Niet gestart  → TrainEntered op max(mip_entry, current_time)
             - Actief        → TrainReadyToExit op max(mip_dep_curr,
                               current_time, phys_ready)

        Vervolgsegmenten worden niet expliciet gepusht; ze volgen via
        _handle_ready_to_exit → _compute_ready_time, dat het opgeslagen
        MIP-plan raadpleegt.
        """
        current_time = self._state.current_time

        # Groepeer entries en departures per trein
        train_data = {}  # train_id -> {seg: (mip_entry, mip_departure)}
        for (train_id, segment_id), mip_entry in solution.entry.items(): #!!! Check solution.arrival.items()
            mip_dep = solution.exit.get((train_id, segment_id))
            if mip_dep is None:
                continue
            if train_id not in train_data:
                train_data[train_id] = {}
            train_data[train_id][segment_id] = (mip_entry, mip_dep)

        for train_id, seg_schedule in train_data.items():
            if train_id not in self._trains:
                continue

            # Sla nieuw MIP-plan op (overschrijft oude)
            self._state.clear_mip_schedule(train_id)
            for seg_id, (mip_e, mip_d) in seg_schedule.items():
                self._state.record_mip_schedule(train_id, seg_id, mip_e, mip_d)
            
            # Trein wordt herplanned naar een nieuwe ready-time — pas op zijn
            # nieuwe TrainReadyToExit-event mag hij opnieuw in een queue komen.
            self._dispatcher.remove_from_queues(train_id)

            current_seg = self._state.current_segment(train_id)

            # ---- Niet gestart ----
            if current_seg is None:
                if self._state.is_finished(train_id):
                    continue
                remaining = self._state.remaining_path(train_id)
                if not remaining:
                    continue
                first_seg = remaining[0]
                mip_entry = seg_schedule.get(first_seg, (None, None))[0]
                if mip_entry is None:
                    continue

                sched_entry_first = self._timetable.scheduled_entry(train_id, first_seg)
                safe_entry = max(mip_entry, current_time, sched_entry_first)
                self._queue.cancel_train_entered(train_id, first_seg)
                self._queue.push(TrainEntered(
                    time=safe_entry, train_id=train_id, segment_id=first_seg,
                ))

            # ---- Actief ----
            else:
                mip_dep_curr = seg_schedule.get(current_seg, (None, None))[1]
                if mip_dep_curr is None:
                    continue

                try:
                    ae = self._state.actual_entry(train_id, current_seg)
                    sd = self._state.sampled_duration(train_id, current_seg)
                    phys_ready = ae + sd
                except KeyError:
                    phys_ready = current_time

                safe_entry = max(mip_dep_curr, current_time, phys_ready)    #!!! heel belangrijk mip_dep_curr misschien uitcommenten

                self._queue.cancel_ready_to_exit(train_id, current_seg)
                self._queue.push(TrainReadyToExit(
                    time=safe_entry, train_id=train_id, segment_id=current_seg,
                ))

    # ------------------------------------------------------------------
    # Hulpfuncties
    # ------------------------------------------------------------------

    def _next_segment(self, train_id: int, segment_id: str):
        path = self._trains[train_id].path
        try:
            idx = path.index(segment_id)
        except ValueError:
            return None
        return path[idx + 1] if idx + 1 < len(path) else None

    def _compute_ready_time(self, train_id, segment_id, entry_time):
        # Sample altijd (voor sampled_duration-tracking)
        duration = sample_duration(
            train=self._trains[train_id],
            segment=self._segments[segment_id],
            timetable=self._timetable,
            entry_time=entry_time,
            rng=self._rng,
        )
        self._state.record_sampled_duration(train_id, segment_id, duration)

        # Twee ondergrenzen !!! C2 constraint
        phys_ready = entry_time + duration

        min_exit = self._dispatcher.min_exit_time(
            train_id=train_id,
            segment_id=segment_id,
            entry_time=entry_time,
            state=self._state,
        )

        return max(phys_ready, min_exit)
    

#!!! helper diagnostiek
    def _previous_segment(
        self,
        train_id: int,
        segment_id: str,
    ) -> str | None:

        train = self._trains[train_id]
        path  = train.path

        try:
            idx = path.index(segment_id)
        except ValueError:
            return None

        if idx == 0:
            return None

        return path[idx - 1]

# =============================================================================
# Hulpfuncties (module-niveau)
# =============================================================================

from domain.segment import SegmentType
from reality.sampling import sample_running_time, seconds_to_period


def _is_passing(train, segment) -> bool:
    """
    True als de trein dit stationssegment passeert zonder te stoppen.

    Passing-segmenten worden gemodelleerd als instantane bezetting
    (PASSING_DURATION = 1s in reality.sampling) — voldoende voor
    headway-conflictdetectie zonder de aanliggende rijtijden te vervormen.
    """
    return segment.is_station and not train.halts_at(segment.id)


def sample_duration(train, segment, timetable, entry_time: float, rng) -> float:
    """
    Sample de fysieke bezettingsduur van een segment.

    Drie gevallen:
      1. Stationspassing  → 1s (via sample_running_time met is_passing=True)
      2. Stationsstop     → geplande dwell uit timetable (C2-conform)
      3. Lijnsegment      → stochastisch via reality.sampling,
                            fallback op gepland tijdsverschil uit timetable.

    De ondergrens van 1s voorkomt dat events op exact dezelfde
    tijdstempel landen, wat de event-queue ordening verstoort.
    """
    # 1. Stationspassing
    if _is_passing(train, segment):
        return 1 

    # 2. Stationsstop
    if segment.is_station:
            if _seconds_to_period(entry_time) in ("MORNING PEAK", "EVENING PEAK"):
                return 120
            return 60

    # 3. Lijnsegment
    sampled = sample_running_time(
        section    = segment.id,
        train_type = train.train_subtype.value,
        dynamics   = train.dynamics_at(segment.id),
        period     = _seconds_to_period(entry_time),
        rng        = rng,
    )
    if sampled is not None:
        return max(1.0, float(sampled))

    row = timetable.get(train.id, segment.id)
    return max(1.0, float(row.exit_seconds - row.entry_seconds))
    # return timetable.scheduled_exit(train.id, segment.id) - timetable.scheduled_entry(train.id, segment.id)


_seconds_to_period = seconds_to_period
