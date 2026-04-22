# simulation/simulator.py
#
# Simulator — centrale event-loop van de rescheduling simulatie.
#
# Verantwoordelijkheden:
#   1. Initialiseer de EventQueue op basis van de geplande timetable
#   2. Verwerk events één voor één (TrainEntered, TrainExited)
#   3. Sample werkelijke rijtijden via data/running_distributions.py
#   4. Handhaaf blokbezetting en FCFS-volgorde via de Dispatcher
#   5. Handhaaf dwell-tijdconstraints via dispatcher.min_exit_time()
#   6. Roep controller.step() aan na elk TrainExited-event
#   7. Verwerk het ControllerResult:
#        - 'rescheduled'   → _apply_solution: herbouw toekomstige events
#        - 'fcfs_fallback' → event_queue.apply_fcfs + dispatcher.reorder
#        - 'skipped'       → geen actie
#   8. Registreer entries en exits in SystemState
#
# Tijdsconventie:
#   Alle tijden in seconden (float), consistent met Timetable en SystemState.
#
# Architectuur:
#   Simulator kent SystemState, EventQueue, Dispatcher en Controller.
#   SystemState, EventQueue en Dispatcher zijn passief — de Simulator
#   is de enige die hun toestand muteert.
#
# Aanroepvolgorde bij TrainExited (kritiek voor correctheid):
#   1. dispatcher.release(train_id, segment_id)
#   2. state.record_exit(train_id, segment_id, time)
#   3. Voor het volgende segment (indien aanwezig):
#      a. dispatcher.enqueue(train_id, next_seg)
#      b. min_time = dispatcher.min_exit_time(train_id, segment_id, entry_time)
#      c. if dispatcher.request_entry(train_id, next_seg, current_time):
#             schedule TrainEntered(next_seg, current_time)
#         else:
#             schedule TrainExited(segment_id, max(current_time, min_time) + POLL)
#   4. controller.step() → verwerk resultaat
#
# Aanroepvolgorde bij TrainEntered:
#   1. dispatcher.confirm_entry(train_id, segment_id)
#   2. state.record_entry(train_id, segment_id, time)
#   3. Schedule TrainExited op gesamplede duur, maar nooit eerder dan
#      dispatcher.min_exit_time()

from __future__ import annotations

import heapq
import logging

import numpy as np

from data.running_distributions import sample_running_time
from domain                     import Timetable
from domain.segment             import SegmentType
from simulation.dispatcher      import Dispatcher
from simulation.event_queue     import EventQueue, TrainEntered, TrainExited
from simulation.state           import SystemState

logger = logging.getLogger(__name__)

# Pollinginterval (seconden): hoe lang een trein wacht voor een bezet segment
# voor hij opnieuw controleert. Klein genoeg voor nauwkeurigheid,
# groot genoeg om oneindige lussen te vermijden.
_POLL_INTERVAL = 1.0


class Simulator:
    """
    Discrete-event simulator voor treinvertragingen op de Brusselse corridor.

    Verwerkt events in chronologische volgorde. Bij elk TrainExited-event
    wordt gecontroleerd of het volgende segment beschikbaar is (dispatcher).
    Als niet, wordt de exit uitgesteld tot het segment vrij is.
    Na elke succesvolle exit wordt de controller aangeroepen.

    Parameters
    ----------
    trains     : dict[int, Train]   — alle treinobjecten
    segments   : dict[str, Segment] — alle segmentobjecten
    timetable  : Timetable          — geplande tijden (onveranderlijk)
    controller : Controller         — rescheduling controller
    seed       : int | None         — random seed voor reproduceerbaarheid
    """

    def __init__(
        self,
        trains:     dict,
        segments:   dict,
        timetable:  Timetable,
        controller,
        seed:       int | None = None,
    ) -> None:
        self._trains      = trains
        self._segments    = segments
        self._timetable   = timetable
        self._controller  = controller
        self._rng         = np.random.default_rng(seed)

        self._state: SystemState = SystemState(
            trains     = trains,
            timetable  = timetable,
            start_time = 0.0,
        )
        self._queue:      EventQueue = EventQueue()
        self._dispatcher: Dispatcher = Dispatcher(
            timetable = timetable,
            segments  = segments,
        )

    # ==========================================================================
    # Publieke interface
    # ==========================================================================

    def run(self) -> SystemState:
        """
        Initialiseert de queue en voert de event-loop uit tot alle events
        verwerkt zijn.

        Returns
        -------
        SystemState — eindtoestand na afloop van de simulatie
        """
        self._initialise()
        self._run_loop()
        return self._state

    # ==========================================================================
    # Initialisatie
    # ==========================================================================

    def _initialise(self) -> None:
        """
        Vult de EventQueue met één TrainEntered-event per trein voor het
        eerste segment, op de geplande entry-tijd.

        Bij het eerste segment meldt de trein zich ook aan in de
        dispatcher-wachtrij, zodat de FCFS-volgorde al vanaf het begin
        correct is.
        """
        for train_id, train in self._trains.items():
            first_seg     = train.first_segment
            planned_entry = self._timetable.scheduled_arrival(train_id, first_seg)

            # Aanmelden in dispatcher-wachtrij voor het eerste segment
            self._dispatcher.enqueue(train_id, first_seg)

            self._queue.push(TrainEntered(
                time       = planned_entry,
                train_id   = train_id,
                segment_id = first_seg,
            ))

        logger.info(
            f"Queue geïnitialiseerd: {len(self._queue)} events "
            f"voor {len(self._trains)} treinen"
        )


    # ==========================================================================
    # Event handlers
    # ==========================================================================

    def _handle_entered(self, event: TrainEntered) -> None:
        """
        Verwerkt een TrainEntered-event:
          1. Bevestig entry in dispatcher (segment bezet, trein uit wachtrij)
          2. Registreer entry in SystemState
          3. Sample werkelijke verblijfsduur
          4. Schedule TrainExited, maar nooit eerder dan min_exit_time()
        """
        # 1. Dispatcher: segment bezet markeren
        self._dispatcher.confirm_entry(event.train_id, event.segment_id)

        # 2. State: registreer entry
        self._state.record_entry(
            train_id   = event.train_id,
            segment_id = event.segment_id,
            time       = event.time,
        )

        # 3. Sample verblijfsduur
        sampled_duration = self._sample_duration(
            event.train_id, event.segment_id, event.time
        )
        sampled_exit = event.time + sampled_duration

        # 4. Dwell-tijdconstraint: nooit eerder dan min_exit_time()
        min_exit = self._dispatcher.min_exit_time(
            train_id   = event.train_id,
            segment_id = event.segment_id,
            entry_time = event.time,
        )
        exit_time = max(sampled_exit, min_exit)

        self._queue.push(TrainExited(
            time       = exit_time,
            train_id   = event.train_id,
            segment_id = event.segment_id,
        ))

    def _handle_exited(self, event: TrainExited) -> None:
        """
        Verwerkt een TrainExited-event:
          1. Geef segment vrij in dispatcher
          2. Registreer exit in SystemState
          3. Bepaal het volgende segment
          4. Als er een volgend segment is:
               a. Meldt trein aan in wachtrij
               b. Check beschikbaarheid via dispatcher
               c. Als beschikbaar: schedule TrainEntered onmiddellijk
               d. Als bezet/wachtrij: schedule uitgestelde poging
          5. Roep controller aan (enkel na succesvolle, definitieve exit)
          6. Verwerk ControllerResult
        """
        # 1. Dispatcher: segment vrijgeven
        self._dispatcher.release(event.train_id, event.segment_id)

        # 2. State: registreer exit
        self._state.record_exit(
            train_id   = event.train_id,
            segment_id = event.segment_id,
            time       = event.time,
        )

        # 3. Bepaal volgend segment
        next_seg = self._next_segment(event.train_id, event.segment_id)

        if next_seg is not None:
            # 4a. Aanmelden in wachtrij voor volgend segment
            self._dispatcher.enqueue(event.train_id, next_seg)

            # 4b-d. Check beschikbaarheid en schedule entry of wacht
            self._try_enter_next(
                train_id    = event.train_id,
                segment_id  = next_seg,
                current_time = event.time,
                prev_seg    = event.segment_id,
            )

        # 5 & 6. Controller aanroepen en resultaat verwerken
        result = self._controller.step(self._state, event.time)

        if result.action == "rescheduled":
            self._apply_solution(result.solution)
        elif result.action == "fcfs_fallback":
            self._queue.apply_fcfs(result.fcfs_order)
            self._dispatcher.reorder(result.fcfs_order)
        # "skipped" → geen actie

    # ==========================================================================
    # Segmenttoegang
    # ==========================================================================

    def _try_enter_next(
        self,
        train_id:     int,
        segment_id:   str,
        current_time: float,
        prev_seg:     str,
    ) -> None:
        """
        Probeert een trein het volgende segment te laten betreden.

        Als het segment beschikbaar is (niet bezet, trein staat eerste in
        wachtrij), wordt een TrainEntered geplant op current_time.

        Als het segment bezet of de wachtrij nog niet aan de beurt is,
        wordt een uitgestelde TrainExited geplant op het huidige segment
        na _POLL_INTERVAL seconden — de simulator controleert dan opnieuw.

        Opmerking: het "uitgestelde TrainExited" is conceptueel een
        wachtperiode, geen echte exit. De exit is al geregistreerd in
        SystemState. De herpoging werkt als een polling-mechanisme.

        Parameters
        ----------
        train_id     : int   — treinnummer
        segment_id   : str   — het volgende segment dat de trein wil betreden
        current_time : float — huidige simulatietijd
        prev_seg     : str   — het segment dat de trein net verlaten heeft
                               (gebruikt voor min_exit_time bij herpoging)
        """
        if self._dispatcher.request_entry(train_id, segment_id, current_time):
            # Segment beschikbaar: schedule directe entry
            self._queue.push(TrainEntered(
                time       = current_time,
                train_id   = train_id,
                segment_id = segment_id,
            ))
        else:
            # Segment bezet of wachtrij: poll na _POLL_INTERVAL seconden
            # We hergebruiken TrainExited op het vorige segment als polling-event.
            # De dispatcher weet dat de exit al geregistreerd is — release()
            # wordt niet opnieuw aangeroepen omdat _occupied al None is.
            logger.debug(
                f"Trein {train_id}: '{segment_id}' niet beschikbaar op t={current_time:.0f}s "
                f"— wacht {_POLL_INTERVAL}s"
            )
            self._queue.push(_WaitingEntry(
                time       = current_time + _POLL_INTERVAL,
                train_id   = train_id,
                segment_id = segment_id,   # het segment dat de trein wil betreden
                prev_seg   = prev_seg,
            ))

    def _handle_waiting(self, event: _WaitingEntry) -> None:
        """
        Verwerkt een wacht-event: herprobeert het segment te betreden.

        Als het segment nu beschikbaar is, wordt een TrainEntered geplant.
        Anders wordt opnieuw een _WaitingEntry geplant na _POLL_INTERVAL.
        """
        self._state.advance_time(event.time)

        if self._dispatcher.request_entry(event.train_id, event.segment_id, event.time):
            self._queue.push(TrainEntered(
                time       = event.time,
                train_id   = event.train_id,
                segment_id = event.segment_id,
            ))
        else:
            logger.debug(
                f"Trein {event.train_id}: '{event.segment_id}' nog niet beschikbaar "
                f"op t={event.time:.0f}s — wacht opnieuw {_POLL_INTERVAL}s"
            )
            self._queue.push(_WaitingEntry(
                time       = event.time + _POLL_INTERVAL,
                train_id   = event.train_id,
                segment_id = event.segment_id,
                prev_seg   = event.prev_seg,
            ))

    # ==========================================================================
    # Hulp: volgend segment
    # ==========================================================================

    def _next_segment(self, train_id: int, segment_id: str) -> str | None:
        """
        Geeft het segment dat volgt op segment_id in het pad van de trein,
        of None als het het laatste segment is.
        """
        train = self._trains[train_id]
        path  = train.path

        try:
            idx = path.index(segment_id)
        except ValueError:
            logger.warning(
                f"Trein {train_id}: '{segment_id}' niet gevonden in pad"
            )
            return None

        if idx + 1 >= len(path):
            return None  # laatste segment

        return path[idx + 1]

    # ==========================================================================
    # Duursampling
    # ==========================================================================

    def _sample_duration(
        self,
        train_id:   int,
        segment_id: str,
        entry_time: float,
    ) -> float:
        """
        Samplet de werkelijke verblijfsduur van een trein op een segment.

        Voor lijnsegmenten: sample uit empirische rijtijdverdeling via
        data/running_distributions.py. Fallback op geplande rijtijd als
        geen verdeling beschikbaar is.

        Voor stationssegmenten: gebruik altijd de geplande dwell-tijd.
        De werkelijke exittijd wordt vervolgens gecorrigeerd door
        dispatcher.min_exit_time() in _handle_entered.

        Parameters
        ----------
        train_id   : int
        segment_id : str
        entry_time : float — actuele entrytijd (seconden), voor periodeberekening

        Returns
        -------
        float — verblijfsduur in seconden (altijd > 0)
        """
        segment = self._segments[segment_id]
        train   = self._trains[train_id]

        if segment.seg_type == SegmentType.STATION:
            try:
                return self._timetable.dwell_time(train_id, segment_id)
            except (KeyError, ValueError):
                logger.warning(
                    f"Trein {train_id}: geen dwell_time voor '{segment_id}' "
                    f"— fallback op 60s"
                )
                return 60.0

        # Lijnsegment: sample uit empirische verdeling
        dynamics = train.dynamics_at(segment_id)
        period   = _seconds_to_period(entry_time)

        if dynamics is None:
            logger.debug(
                f"Trein {train_id}: geen dynamics voor '{segment_id}' "
                f"— fallback op geplande rijtijd"
            )
            return self._planned_duration(train_id, segment_id)

        sampled = sample_running_time(
            section    = segment_id,
            train_type = train.train_subtype.value,
            dynamics   = dynamics,
            period     = period,
            rng        = self._rng,
        )

        if sampled is None:
            return self._planned_duration(train_id, segment_id)

        return sampled

    def _planned_duration(self, train_id: int, segment_id: str) -> float:
        """
        Geplande verblijfsduur als fallback: exit_seconds − entry_seconds.

        Returns
        -------
        float — geplande duur in seconden (minimaal 1.0)
        """
        try:
            entry = self._timetable.scheduled_arrival(train_id, segment_id)
            exit_ = self._timetable.scheduled_departure(train_id, segment_id)
            return max(1.0, exit_ - entry)
        except (KeyError, ValueError):
            logger.warning(
                f"Trein {train_id}: geen geplande tijden voor '{segment_id}' "
                f"— fallback op 60s"
            )
            return 60.0

    # ==========================================================================
    # MIP-oplossing toepassen
    # ==========================================================================

    def _apply_solution(self, solution) -> None:
        """
        Verwerkt een MIP-oplossing in de EventQueue.

        Voor elk (train_id, segment_id) paar in de oplossing:
          - Als het segment nog niet afgerond is in SystemState:
              Verwijder bestaande TrainEntered en TrainExited events
              voor dat (train_id, segment_id) en voeg nieuwe toe op
              de door de solver berekende tijden.

        Segmenten waarvoor al een actual_exit geregistreerd is worden
        nooit aangeraakt.

        Parameters
        ----------
        solution : Solution — output van model/solution.py
        """
        to_reschedule: set[tuple[int, str]] = set()

        for (train_id, segment_id), new_entry in solution.arrival.items():
            if train_id not in self._trains:
                continue
            try:
                self._state.actual_exit(train_id, segment_id)
                continue  # exit al geregistreerd — niet aanraken
            except KeyError:
                pass

            new_exit = solution.departure.get((train_id, segment_id))
            if new_exit is None:
                continue

            to_reschedule.add((train_id, segment_id))

        if not to_reschedule:
            return

        # Filter queue: verwijder events voor te herbouwen segmenten
        # (ook _WaitingEntry events voor deze treinen)
        remaining = []
        for entry in self._queue._heap:
            ev = entry.event
            if (ev.train_id, ev.segment_id) in to_reschedule:
                continue
            remaining.append(entry)

        heapq.heapify(remaining)
        self._queue._heap = remaining

        # Voeg nieuwe events toe op MIP-tijden
        for (train_id, segment_id) in to_reschedule:
            new_entry_time = solution.arrival[(train_id, segment_id)]
            new_exit_time  = solution.departure[(train_id, segment_id)]

            try:
                self._state.actual_entry(train_id, segment_id)
                # Entry al geregistreerd — geen nieuw TrainEntered
            except KeyError:
                self._queue.push(TrainEntered(
                    time       = new_entry_time,
                    train_id   = train_id,
                    segment_id = segment_id,
                ))

            self._queue.push(TrainExited(
                time       = new_exit_time,
                train_id   = train_id,
                segment_id = segment_id,
            ))

        logger.debug(
            f"apply_solution: {len(to_reschedule)} (trein, segment) paren herbouwd"
        )

    # ==========================================================================
    # Gewijzigde _run_loop om _WaitingEntry te verwerken
    # ==========================================================================

    def _run_loop(self) -> None:
        """
        Hoofdlus: pop events in chronologische volgorde en verwerk ze.
        Ondersteunt drie event-types: TrainEntered, TrainExited, _WaitingEntry.
        """
        n_processed = 0

        while self._queue:
            event = self._queue.pop()
            self._state.advance_time(event.time)

            if isinstance(event, TrainEntered):
                self._handle_entered(event)
            elif isinstance(event, TrainExited):
                self._handle_exited(event)
            elif isinstance(event, _WaitingEntry):
                self._handle_waiting(event)

            n_processed += 1

        logger.info(f"Simulatie klaar — {n_processed} events verwerkt")

    # ==========================================================================
    # Diagnostiek
    # ==========================================================================

    def __repr__(self) -> str:
        return (
            f"Simulator("
            f"treinen={len(self._trains)}, "
            f"queue={len(self._queue)} events, "
            f"state={self._state.summary()}, "
            f"dispatcher={self._dispatcher})"
        )


# =============================================================================
# Intern wacht-event
# =============================================================================

from dataclasses import dataclass, field  # noqa: E402 — na klassedefinitie om circulaire ref te vermijden


@dataclass(order=False)
class _WaitingEntry:
    """
    Intern polling-event: trein wacht op toegang tot een segment.

    Niet zichtbaar buiten simulator.py. Wordt aangemaakt door _try_enter_next
    als request_entry() False teruggeeft, en verwerkt door _handle_waiting.

    Attributen
    ----------
    time       : float — tijdstip van de herpoging
    train_id   : int   — treinnummer
    segment_id : str   — het segment dat de trein wil betreden
    prev_seg   : str   — het segment dat de trein al verlaten heeft
    """
    time:       float
    train_id:   int
    segment_id: str
    prev_seg:   str
    kind: str = field(default="waiting", init=False, repr=False)


# =============================================================================
# Hulpfunctie: seconden → dagperiode
# =============================================================================

def _seconds_to_period(seconds: float) -> str:
    """
    Zet een tijdstip in seconden (vanaf middernacht) om naar een dagperiode.

    Periodes consistent met data/timetable.py add_period() en dispatcher.py.
    """
    hour = (seconds % 86400) / 3600

    if hour < 6:
        return "NIGHT"
    elif hour < 9:
        return "MORNING PEAK"
    elif hour < 16:
        return "DAYTIME"
    elif hour < 19:
        return "EVENING PEAK"
    else:
        return "EVENING"