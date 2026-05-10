# simulation/simulator.py
#
# Simulator — centrale event-loop van de rescheduling simulatie.
#
# Verantwoordelijkheden:
#   1. Initialiseer de EventQueue op basis van de geplande timetable
#   2. Verwerk events één voor één (TrainEntered, TrainExited)
#   3. Sample werkelijke rijtijden via reality/sampling.py
#   4. Handhaaf blokbezetting en volgorde via de Dispatcher
#   5. Handhaaf dwell-tijdconstraints via dispatcher.min_exit_time()
#   6. Roep controller.step() aan na elk TrainExited-event
#   7. Verwerk het ControllerResult:
#        - 'rescheduled'   → _apply_solution: herbouw toekomstige events
#        - 'fcfs_fallback' → dispatcher.reorder (volgorde aanpassen)
#        - 'skipped'       → geen actie
#   8. Registreer entries en exits in SystemState
#
# Tijdsconventie:
#   TrainEntered.time = geplande/MIP entry-tijd
#   TrainExited.time  = werkelijke exit-tijd (entry + gesamplede duur)
#   state.record_entry() ontvangt de werkelijke entry-tijd (current_time
#   op het moment van verwerking, niet de geplande tijd)
#
# Architectuur:
#   Simulator kent SystemState, EventQueue, Dispatcher en Controller.
#   SystemState, EventQueue en Dispatcher zijn passief — de Simulator
#   is de enige die hun toestand muteert.
#
# Aanroepvolgorde bij TrainEntered:
#   1. dispatcher.request_entry() — heeft trein voorrang?
#      - Nee → TrainEntered opnieuw plannen na _POLL_INTERVAL
#      - Ja  → dispatcher.confirm_entry()
#              state.record_entry(werkelijke tijd = current_time)
#              TrainExited plannen op gesamplede duur
#
# Aanroepvolgorde bij TrainExited:
#   1. dispatcher.release()
#   2. state.record_exit()
#   3. Voor volgend segment (indien aanwezig en nog niet gepland):
#      a. dispatcher.enqueue(planned_time)
#      b. TrainEntered plannen op geplande tijd
#   4. controller.step() → verwerk resultaat

from __future__ import annotations

import logging

import numpy as np

from reality.sampling import sample_running_time
from domain                     import Timetable
from domain.segment             import SegmentType
from simulation.dispatcher      import Dispatcher
from simulation.event_queue     import EventQueue, TrainEntered, TrainExited
from simulation.state           import SystemState

logger = logging.getLogger(__name__)

# Pollinginterval (seconden): hoe lang een trein wacht als hij geen voorrang
# heeft van de dispatcher. Klein genoeg voor nauwkeurigheid.
_POLL_INTERVAL = 30.0


class Simulator:
    """
    Discrete-event simulator voor treinvertragingen op de Brusselse corridor.

    Verwerkt events in chronologische volgorde. Bij elk TrainEntered-event
    wordt via de dispatcher gecontroleerd of de trein voorrang heeft.
    Als niet, wordt het event uitgesteld. Na elke TrainExited wordt de
    controller aangeroepen.

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

        Meldt elke trein ook aan in de dispatcher-wachtrij zodat de
        volgorde al vanaf het begin correct is.
        """
        for train_id, train in self._trains.items():
            first_seg     = train.first_segment
            planned_entry = self._timetable.scheduled_arrival(train_id, first_seg)

            # Aanmelden in dispatcher-wachtrij op geplande tijd
            self._dispatcher.enqueue(train_id, first_seg, planned_entry)

            # TrainEntered op geplande tijd
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
    # Event-loop
    # ==========================================================================

    def _run_loop(self) -> None:
        """
        Hoofdlus: pop events in chronologische volgorde en verwerk ze.
        """
        n_processed = 0

        while self._queue:
            event = self._queue.pop()                
            self._state.advance_time(event.time)

            if isinstance(event, TrainEntered):
                self._handle_entered(event)
            elif isinstance(event, TrainExited):
                self._handle_exited(event)

            n_processed += 1

        logger.info(f"Simulatie klaar — {n_processed} events verwerkt")

    # ==========================================================================
    # Event handlers
    # ==========================================================================

    def _handle_entered(self, event: TrainEntered) -> None:
        """
        Verwerkt een TrainEntered-event.

        Controleert eerst via de dispatcher of de trein voorrang heeft.
        Als niet, wordt het event uitgesteld met _POLL_INTERVAL.
        Als wel, wordt entry bevestigd, geregistreerd en TrainExited gepland.

        De werkelijke entry-tijd is current_time (het moment van verwerking),
        niet de geplande tijd in het event.
        """
        current_time = event.time

        # 1. Check of trein voorrang heeft (volgorde + segment vrij)
        if not self._dispatcher.request_entry(event.train_id, event.segment_id, current_time):
            # Geen voorrang of segment bezet — uitstellen
            # Gebruik state.current_time zodat retry altijd in de toekomst ligt
            retry_time = self._state.current_time + _POLL_INTERVAL
            logger.debug(
                f"Trein {event.train_id}: geen toegang tot '{event.segment_id}' "
                f"op t={current_time:.0f}s — herpoging op t={retry_time:.0f}s"
            )
            self._queue.push(TrainEntered(
                time       = retry_time,
                train_id   = event.train_id,
                segment_id = event.segment_id,
            ))
            return

        # 2. Dispatcher: segment bezet markeren
        self._dispatcher.confirm_entry(event.train_id, event.segment_id)

        # 3. State: registreer werkelijke entry-tijd
        self._state.record_entry(
            train_id   = event.train_id,
            segment_id = event.segment_id,
            time       = current_time,
        )

        # 4. Sample werkelijke verblijfsduur
        sampled_duration = self._sample_duration(
            event.train_id, event.segment_id, current_time
        )
        sampled_exit = current_time + sampled_duration

        # 5. C2 constraint: nooit eerder vertrekken dan min_exit_time
        min_exit  = self._dispatcher.min_exit_time(
            train_id   = event.train_id,
            segment_id = event.segment_id,
            entry_time = current_time,
        )
        exit_time = max(sampled_exit, min_exit)

        # 6. Plan TrainExited op werkelijke exittijd
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
        3. Roep controller aan en verwerk resultaat
        4. Plan TrainEntered voor volgend segment op geplande tijd
            (na _apply_solution, enkel als nog niet gepland)
        """
        # 1. Dispatcher: segment vrijgeven
        self._dispatcher.release(event.train_id, event.segment_id)

        # 2. State: registreer exit
        self._state.record_exit(
            train_id   = event.train_id,
            segment_id = event.segment_id,
            time       = event.time,
        )

        # 3. Controller aanroepen en resultaat verwerken
        result = self._controller.step(self._state, event.time)
        if result.action == "rescheduled":
            self._apply_solution(result.solution)
        elif result.action == "fcfs_fallback":
            self._dispatcher.reorder(result.fcfs_order)

        # 4. Volgend segment plannen (na _apply_solution)
        next_seg = self._next_segment(event.train_id, event.segment_id)
        if next_seg is not None:
            planned_entry = self._timetable.scheduled_arrival(event.train_id, next_seg)
            actual_entry  = max(planned_entry, self._state.current_time)
            self._dispatcher.enqueue(event.train_id, next_seg, planned_entry)
            if not self._queue.has_entered(event.train_id, next_seg):
                self._queue.push(TrainEntered(
                    time       = actual_entry,
                    train_id   = event.train_id,
                    segment_id = next_seg,
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
            return None

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
        return sample_duration(
            train     = self._trains[train_id],
            segment   = self._segments[segment_id],
            timetable = self._timetable,
            entry_time= entry_time,
            rng       = self._rng,
        )

    
    def _apply_solution(self, solution) -> None:

        current_time = self._state.current_time
        to_reschedule = []

        # ==========================================================
        # 1. FILTER geldig
        # ==========================================================
        for (train_id, segment_id), mip_entry in solution.arrival.items():

            if train_id not in self._trains:
                continue

            # skip als al gestart
            try:
                self._state.actual_entry(train_id, segment_id)
                continue
            except KeyError:
                pass

            mip_exit = solution.departure.get((train_id, segment_id))
            if mip_exit is None:
                continue

            # 🔥 FORCE NO PAST
            safe_entry = max(mip_entry, current_time)

            # 🔥 GEEN exit pushen → simulator bepaalt die
            to_reschedule.append((train_id, segment_id, safe_entry))

        if not to_reschedule:
            return

        # ==========================================================
        # 2. CANCEL ALLES VEILIG
        # ==========================================================
        for train_id, segment_id, _ in to_reschedule:
            self._queue.cancel(train_id, segment_id)

        # ==========================================================
        # 3. PUSH ALLEEN ENTRY EVENTS
        # ==========================================================
        for train_id, segment_id, safe_entry in to_reschedule:

            if safe_entry <= 0:
                print(f"🚨 SKIP zero entry: {train_id}, {segment_id}")
                continue

            self._queue.push(TrainEntered(
                time       = safe_entry,
                train_id   = train_id,
                segment_id = segment_id,
            ))

        # ==========================================================
        # 4. DISPATCHER CONSISTENT MAKEN
        # ==========================================================
        seg_order = {}

        for train_id, segment_id, safe_entry in to_reschedule:
            seg_order.setdefault(segment_id, []).append((safe_entry, train_id))

        reorder_dict = {
            seg: [t for _, t in sorted(entries)]
            for seg, entries in seg_order.items()
        }

        if reorder_dict:
            self._dispatcher.reorder(reorder_dict)

        print(f"[DEBUG] apply_solution: {len(to_reschedule)} updates toegepast")


# =============================================================================
# Hulpfunctie: duursampling
# =============================================================================

def sample_duration(train, segment, timetable, entry_time: float, rng) -> float:
    """
    Samplet de verblijfsduur van een trein op een segment.

    Losse module-level functie zodat ook triggers.py ze kan importeren
    zonder afhankelijkheid van de Simulator-instantie.

    Voor stationssegmenten: geplande dwell-tijd.
    Voor lijnsegmenten: sample via reality module, fallback op geplande duur.

    Parameters
    ----------
    train      : Train
    segment    : Segment
    timetable  : Timetable
    entry_time : float — werkelijke entrytijd (voor periodeberekening)
    rng        : np.random.Generator

    Returns
    -------
    float — verblijfsduur in seconden (altijd >= 1.0)
    """
    if segment.seg_type == SegmentType.STATION:
        try:
            return timetable.dwell_time(train.id, segment.id)
        except (KeyError, ValueError):
            logger.warning(
                f"Trein {train.id}: geen dwell_time voor '{segment.id}' "
                f"— fallback op 60s"
            )
            return 60.0

    # Lijnsegment: sample uit empirische verdeling
    dynamics = train.dynamics_at(segment.id)
    period   = _seconds_to_period(entry_time)

    if dynamics is not None:
        sampled = sample_running_time(
            section    = segment.id,
            train_type = train.train_subtype.value,
            dynamics   = dynamics,
            period     = period,
            rng        = rng,
        )
        if sampled is not None:
            return sampled

    logger.debug(
        f"Trein {train.id}: geen data voor '{segment.id}' "
        f"— fallback op geplande rijtijd"
    )
    st = timetable.get(train.id, segment.id)
    return max(1.0, st.exit_seconds - st.entry_seconds)


# =============================================================================
# Hulpfunctie: seconden → dagperiode
# =============================================================================

def _seconds_to_period(seconds: float) -> str:
    """
    Zet een tijdstip in seconden (vanaf middernacht) om naar een dagperiode.

    Periodes consistent met data/timetable.py add_period().
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