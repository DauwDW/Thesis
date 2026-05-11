from __future__ import annotations

import logging

import numpy as np

from simulation.dispatcher import Dispatcher
from simulation.event_queue import EventQueue, TrainEntered, TrainReadyToExit
from simulation.state import SystemState

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 30

# Objective-drempel waarboven een run als deadlock wordt beschouwd.
# Wanneer de MIP-objective deze waarde overschrijdt, gooit de simulator
# een DeadlockDetected exception. run_simulation.py vangt deze op en
# markeert de run als incomplete. Zie thesis §X voor motivatie.
_DEADLOCK_OBJECTIVE_THRESHOLD = 600000.0


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
      De MIP stuurt volgorde impliciet via TrainReadyToExit-tijden.

    Smart retry:
      Bij blokkering retried een trein op expected_release_time van het
      volgende segment. Als expected_release stale is (< current_time),
      wordt _POLL_INTERVAL gebruikt als fallback om een 1-seconde
      polling loop te vermijden.
    """

    def __init__(self, trains, segments, timetable, controller, seed=None):
        self._trains = trains
        self._segments = segments
        self._timetable = timetable
        self._controller = controller
        self._rng = np.random.default_rng(seed)

        self._state = SystemState(trains=trains, timetable=timetable, start_time=0.0)
        self._queue = EventQueue()
        self._dispatcher = Dispatcher(timetable=timetable, segments=segments)

    # ------------------------------------------------------------------
    # Publieke interface
    # ------------------------------------------------------------------

    def run(self):
        self._initialise()
        while self._queue:
            event = self._queue.pop()
            self._state.advance_time(event.time)
            if isinstance(event, TrainEntered):
                if not self._is_stale_entered(event):
                    self._handle_entered(event)
            elif isinstance(event, TrainReadyToExit):
                if not self._is_stale_ready_to_exit(event):
                    self._handle_ready_to_exit(event)
            else:
                raise TypeError(type(event))
        return self._state

    # ------------------------------------------------------------------
    # Stale-event checks
    # ------------------------------------------------------------------

    def _is_stale_entered(self, event: TrainEntered) -> bool:
        if self._state.is_finished(event.train_id):
            return True
        if self._state.current_segment(event.train_id) is not None:
            return True
        remaining = self._state.remaining_path(event.train_id)
        return not remaining or remaining[0] != event.segment_id

    def _is_stale_ready_to_exit(self, event: TrainReadyToExit) -> bool:
        return self._state.current_segment(event.train_id) != event.segment_id

    # ------------------------------------------------------------------
    # Initialisatie
    # ------------------------------------------------------------------

    def _initialise(self):
        for train_id in sorted(self._trains):
            train = self._trains[train_id]
            first_seg = train.first_segment
            entry_time = self._timetable.scheduled_arrival(train_id, first_seg)
            self._queue.push(TrainEntered(
                time=entry_time,
                train_id=train_id,
                segment_id=first_seg,
            ))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_entered(self, event: TrainEntered) -> None:
        current_time = event.time

        if not self._dispatcher.request_entry(event.train_id, event.segment_id, current_time):
            release = self._dispatcher.expected_release_time(event.segment_id)
            retry = release if (release is not None and release > current_time) \
                    else current_time + _POLL_INTERVAL
            self._queue.push(TrainEntered(
                time=retry,
                train_id=event.train_id,
                segment_id=event.segment_id,
            ))
            return

        self._dispatcher.confirm_entry(event.train_id, event.segment_id)
        self._state.record_entry(
            train_id=event.train_id,
            segment_id=event.segment_id,
            time=current_time,
        )
        ready_time = self._compute_ready_time(event.train_id, event.segment_id, current_time)
        self._dispatcher.set_expected_release(event.segment_id, ready_time)
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
            result = self._controller.step(self._state, current_time)
            self._apply_result(result)
            return

        # --- geblokkeerd ---
        if not self._dispatcher.request_entry(event.train_id, next_seg, current_time):
            release = self._dispatcher.expected_release_time(next_seg)
            retry = release if (release is not None and release > current_time) \
                    else current_time + _POLL_INTERVAL
            self._dispatcher.set_expected_release(event.segment_id, retry)
            self._queue.push(TrainReadyToExit(
                time=retry,
                train_id=event.train_id,
                segment_id=event.segment_id,
            ))
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
        self._dispatcher.set_expected_release(next_seg, ready_time)
        self._queue.push(TrainReadyToExit(
            time=ready_time,
            train_id=event.train_id,
            segment_id=next_seg,
        ))
        result = self._controller.step(self._state, current_time)
        self._apply_result(result)

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
            self._apply_solution(result.solution)

    def _apply_solution(self, solution) -> None:
        """
        Verwerk een MIP-oplossing: herplan toekomstige entries.

        A) Niet-gestarte trein, eerste resterende segment:
           Cancel TrainEntered + push nieuw op safe_entry.
        B) Actieve trein geblokkeerd op current_seg voor next_seg:
           Cancel TrainReadyToExit + push nieuw op safe_entry.
        C) Alles anders: geen actie.
        """
        current_time = self._state.current_time

        for (train_id, segment_id), mip_entry in solution.arrival.items():

            if train_id not in self._trains:
                continue

            try:
                self._state.actual_entry(train_id, segment_id)
                continue
            except KeyError:
                pass

            safe_entry = max(mip_entry, current_time)
            current_seg = self._state.current_segment(train_id)

            if current_seg is None:
                if self._state.is_finished(train_id):
                    continue
                remaining = self._state.remaining_path(train_id)
                if remaining and segment_id == remaining[0]:
                    self._queue.cancel(train_id, segment_id)
                    self._queue.push(TrainEntered(
                        time=safe_entry,
                        train_id=train_id,
                        segment_id=segment_id,
                    ))
            else:
                next_seg = self._next_segment(train_id, current_seg)
                if next_seg == segment_id:
                    self._queue.cancel_ready_to_exit(train_id, current_seg)
                    self._queue.push(TrainReadyToExit(
                        time=safe_entry,
                        train_id=train_id,
                        segment_id=current_seg,
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

    def _compute_ready_time(self, train_id: int, segment_id: str, entry_time: float) -> float:
        duration = sample_duration(
            train=self._trains[train_id],
            segment=self._segments[segment_id],
            timetable=self._timetable,
            entry_time=entry_time,
            rng=self._rng,
        )
        min_exit = self._dispatcher.min_exit_time(train_id, segment_id, entry_time)
        return max(entry_time + duration, min_exit)


# =============================================================================
# Hulpfuncties (module-niveau)
# =============================================================================

from domain.segment import SegmentType
from reality.sampling import sample_running_time


def sample_duration(train, segment, timetable, entry_time: float, rng) -> float:
    """
    Sample de fysieke bezettingsduur van een segment.

    STATION: gepland verblijf via timetable.dwell_time() (fallback: 60s).
    Overige segmenten: stochastisch via reality.sampling, fallback op
    gepland tijdsverschil uit timetable.
    """
    if segment.seg_type == SegmentType.STATION:
        try:
            return max(1.0, float(timetable.dwell_time(train.id, segment.id)))
        except Exception:
            return 60.0

    dynamics = train.dynamics_at(segment.id)
    period = _seconds_to_period(entry_time)

    if dynamics is not None:
        sampled = sample_running_time(
            section=segment.id,
            train_type=train.train_subtype.value,
            dynamics=dynamics,
            period=period,
            rng=rng,
        )
        if sampled is not None:
            return max(1.0, float(sampled))

    row = timetable.get(train.id, segment.id)
    return max(1.0, float(row.exit_seconds - row.entry_seconds))


def _seconds_to_period(seconds: float) -> str:
    hour = (seconds % 86400) / 3600
    if hour < 6:    return "NIGHT"
    if hour < 9:    return "MORNING PEAK"
    if hour < 16:   return "DAYTIME"
    if hour < 19:   return "EVENING PEAK"
    return "EVENING"