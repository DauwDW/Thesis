from __future__ import annotations

import logging

import numpy as np

from reality.sampling import sample_running_time, seconds_to_period
from simulation.dispatcher import Dispatcher
from simulation.event_queue import EventQueue, TrainEntered, TrainReadyToExit
from simulation.state import SystemState

logger = logging.getLogger(__name__)


# Drempelwaarden voor deadlock-detectie.
_DEADLOCK_OBJECTIVE_THRESHOLD  = 6_000_000_000.0
_DEADLOCK_TIME_LIMIT           = 200_000          # seconden simulatietijd
_DEADLOCK_CONSECUTIVE_FAILURES = 2                # opeenvolgende solver-fouten


class DeadlockDetected(RuntimeError):
    """
    Raised wanneer de simulator een (vermoedelijke) deadlock detecteert.

    Drie triggers:
      - MIP-objective overschrijdt _DEADLOCK_OBJECTIVE_THRESHOLD
      - Simulatietijd overschrijdt _DEADLOCK_TIME_LIMIT
      - Event-queue leeg terwijl treinen het netwerk niet verlaten hebben
      - _DEADLOCK_CONSECUTIVE_FAILURES opeenvolgende solver-fouten

    run_simulation.py vangt deze op en markeert de run als incomplete.
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

    def __init__(self, trains, segments, timetable, controller, seed=None, queue_mode: str = "fsfs"):
        self._trains      = trains
        self._segments    = segments
        self._timetable   = timetable
        self._controller  = controller
        self._rng         = np.random.default_rng(seed)
        self._state       = SystemState(trains=trains, timetable=timetable, start_time=0.0)
        self._queue       = EventQueue()
        self._dispatcher  = Dispatcher(timetable=timetable, segments=segments, trains=trains, queue_mode=queue_mode)
        self._solutions   = []

    # ------------------------------------------------------------------
    # Publieke interface
    # ------------------------------------------------------------------

    def run(self):
        self._initialise()
        while self._queue:
            event = self._queue.pop()
            self._state.advance_time(event.time)

            if self._state.current_time > _DEADLOCK_TIME_LIMIT:
                raise DeadlockDetected(
                    f"t={self._state.current_time:.0f}s — t > 200.000, deadlock."
                )

            if isinstance(event, TrainEntered):
                if not self._is_stale_entered(event):
                    self._handle_entered(event)
            elif isinstance(event, TrainReadyToExit):
                if not self._is_stale_ready_to_exit(event):
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
            suffix  = "" if len(unfinished) <= 10 else f", ... (+{len(unfinished) - 10})"
            not_started = [t for t in unfinished if not self._state._actual[t]]
            in_waiting  = {seg: list(w) for seg, w in self._dispatcher._waiting.items() if w}
            print(f"[DEADLOCK] niet gestart: {len(not_started)}/{len(unfinished)}")
            print(f"[DEADLOCK] bezette segmenten op eindtijd: "
                  f"{ {s: o for s, o in self._dispatcher._occupied.items() if o is not None} }")
            print(f"[DEADLOCK] waiting lists: "
                  f"{sum(len(w) for w in in_waiting.values())} treinen "
                  f"over {len(in_waiting)} segmenten")
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
            train      = self._trains[train_id]
            first_seg  = train.first_segment
            entry_time = self._timetable.scheduled_entry(train_id, first_seg)
            self._queue.push(TrainEntered(
                time=entry_time,
                train_id=train_id,
                segment_id=first_seg,
            ))

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
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_entered(self, event: TrainEntered) -> None:
        current_time = event.time

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
        next_seg     = self._next_segment(event.train_id, event.segment_id)

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

        Voor elke trein in solution.entry:
          1. Sla complete (entry, departure) schedule op via record_mip_schedule.
          2. Push één event op basis van huidige positie:
             - Niet gestart  → TrainEntered op max(mip_entry, current_time,
                               scheduled_entry)  — scheduled_entry als veiligheidsnet
             - Actief        → TrainReadyToExit op max(mip_dep_curr,
                               current_time, phys_ready)

        Vervolgsegmenten worden niet expliciet gepusht; ze volgen via
        _handle_ready_to_exit → _compute_ready_time, dat het opgeslagen
        MIP-plan raadpleegt.
        """
        current_time = self._state.current_time

        # Groepeer entries en departures per trein
        train_data: dict[int, dict[str, tuple[float, float]]] = {}
        for (train_id, segment_id), mip_entry in solution.entry.items():
            mip_dep = solution.exit.get((train_id, segment_id))
            if mip_dep is None:
                continue
            train_data.setdefault(train_id, {})[segment_id] = (mip_entry, mip_dep)  #.setdefault: Maak deze key aan als hij nog niet bestaat

        # =======================================================================
        # SNAPSHOT — vóór set_platform_choice
        #
        # Twee zaken moeten worden vastgelegd terwijl _platform_choices nog de
        # VORIGE toestand beschrijft:
        #
        # (A) _mip_first_seg: remaining[0] per niet-gestarte trein.
        #     Gebruikt als opzoeksleutel in seg_schedule (die geïndexeerd is op de
        #     segmenten zoals remaining_path() ze teruggaf bij MIP-build).
        #
        # (B) _planned_for_mip_seg: origineel gepland segment per MIP-sleutel.
        #     Nodig voor record_mip_schedule na set_platform_choice.
        #
        # Waarom (B) vóór set_platform_choice?
        #   Scenario platform 1 → 2 → 3:
        #   - Vóór update: _platform_choices = {platform_1: platform_2}
        #     → get_planned_seg_for(platform_2) = platform_1  ✓
        #   - Na update:   _platform_choices = {platform_1: platform_3}
        #     → get_planned_seg_for(platform_2) zoekt waarde platform_2
        #       maar vindt alleen platform_3 → retourneert platform_2 zelf
        #     → get_chosen_seg(platform_2) = platform_2 (niet een sleutel)
        #     → record_mip_schedule slaat op onder platform_2 ipv platform_3 ← BUG
        #   Door de snapshot vóór de update te nemen beschikt stap (B) over de
        #   juiste planned_seg, en stap 2 (get_chosen_seg na update) geeft dan
        #   platform_3 terug.
        # =======================================================================
        _mip_first_seg: dict[int, str] = {}     # voor niet gestarte treinen
        _planned_for_mip_seg: dict[int, dict[str, str]] = {}

        for train_id, seg_schedule in train_data.items():
            if train_id not in self._trains:
                continue
            # (A)
            if self._state.current_segment(train_id) is None and not self._state.is_finished(train_id):
                rem = self._state.remaining_path(train_id)
                if rem:
                    _mip_first_seg[train_id] = rem[0]
            # (B)
            _planned_for_mip_seg[train_id] = {
                seg_id: self._state.get_planned_seg_for(train_id, seg_id)
                for seg_id in seg_schedule
            }

        # --- Retracking: platform-overrides registreren vóór event-push ---
        #
        # Bescherming: overschrijf nooit een platform-keuze voor een segment
        # dat de trein al betreden heeft. De trein zit er al op (of heeft het
        # al verlaten) — wisselen is fysiek onmogelijk en leidt tot deadlock.
        for (train_id, key_seg), chosen_seg in solution.platform_choices.items():
            if train_id not in self._trains:
                continue
            # key_seg is het segment zoals het in de MIP-sleutel stond
            # (chosen_seg van de vorige ronde of het originele planned_seg).
            # Bescherming: controleer of de HUIDIGE keuze al betreden is.
            current_chosen = self._state.get_chosen_seg(
                train_id,
                self._state.get_planned_seg_for(train_id, key_seg),
            )
            try:
                self._state.actual_entry(train_id, current_chosen)
                continue  # al betreden → niet overschrijven
            except KeyError:
                pass
            self._state.set_platform_choice(train_id, key_seg, chosen_seg)

        for train_id, seg_schedule in train_data.items():
            if train_id not in self._trains:
                continue

            # Bug-fix (2): sla MIP-plan op onder het HUIDIGE GEKOZEN segment.
            #
            # Twee-staps vertaling met snapshot (zie uitleg boven):
            #   1. planned_for_seg  = snapshot genomen vóór set_platform_choice
            #   2. chosen_for_mip   = get_chosen_seg(planned) ná set_platform_choice
            #      → geeft het nieuwe gekozen segment (bijv. platform_3).
            self._state.clear_mip_schedule(train_id)
            seg_to_planned = _planned_for_mip_seg.get(train_id, {})
            for seg_id, (mip_e, mip_d) in seg_schedule.items():
                planned_for_seg = seg_to_planned.get(seg_id, seg_id)
                chosen_for_mip  = self._state.get_chosen_seg(train_id, planned_for_seg)
                self._state.record_mip_schedule(train_id, chosen_for_mip, mip_e, mip_d)

            # Trein wordt herplanned — haal hem eerst uit eventuele waiting-lists.
            self._dispatcher.remove_from_queues(train_id)

            current_seg = self._state.current_segment(train_id)

            # ---- Niet gestart ----
            if current_seg is None:
                if self._state.is_finished(train_id):
                    continue
                remaining = self._state.remaining_path(train_id)
                if not remaining:
                    continue
                # remaining[0] is het GEKOZEN segment NA de huidige platform-updates.
                first_seg_chosen  = remaining[0]
                first_seg_planned = self._state.get_planned_seg_for(
                    train_id, first_seg_chosen
                )
                # Bug-fix (1): gebruik de snapshot om de MIP-sleutel op te zoeken.
                #
                # seg_schedule is geïndexeerd op het segment zoals remaining_path()
                # het teruggaf bij MIP-build — dat is _mip_first_seg[train_id].
                # Bij een eerste retrack is dat het geplande segment (= first_seg_planned);
                # bij een tweede retrack is dat het gekozen segment van de vorige ronde,
                # dat noch first_seg_chosen noch first_seg_planned is.
                mip_key   = _mip_first_seg.get(train_id, first_seg_planned)
                mip_entry = seg_schedule.get(mip_key, (None, None))[0]
                if mip_entry is None:
                    continue

                # scheduled_entry als ondergrens — opzoeken via gepland segment
                # (timetable is geïndexeerd op geplande segmenten)
                sched_entry = self._timetable.scheduled_entry(
                    train_id, first_seg_planned
                )
                safe_time = max(mip_entry, current_time, sched_entry)
                # Ruim alle bestaande TrainEntered-events voor deze trein op,
                # ook voor het vorige (voor retracking: andere) segment.
                self._queue.cancel_all_train_entered(train_id)
                self._queue.push(TrainEntered(
                    time=safe_time, train_id=train_id, segment_id=first_seg_chosen,
                ))

            # ---- Actief ----
            else:
                # current_seg is het GEKOZEN segment (opgeslagen door record_entry).
                # seg_schedule is geïndexeerd op GEKOZEN segmenten — gebruik
                # current_seg direct, niet de geplande versie.
                # (get_planned_seg_for is alleen nodig voor timetable-lookups.)
                mip_dep_curr = seg_schedule.get(current_seg, (None, None))[1]
                if mip_dep_curr is None:
                    continue

                try:
                    ae         = self._state.actual_entry(train_id, current_seg)
                    sd         = self._state.sampled_duration(train_id, current_seg)
                    phys_ready = ae + sd
                except KeyError:
                    phys_ready = current_time

                safe_time = max(mip_dep_curr, current_time, phys_ready)
                self._queue.cancel_ready_to_exit(train_id, current_seg)
                self._queue.push(TrainReadyToExit(
                    time=safe_time, train_id=train_id, segment_id=current_seg,
                ))

    # ------------------------------------------------------------------
    # Hulpfuncties
    # ------------------------------------------------------------------

    def _next_segment(self, train_id: int, segment_id: str) -> str | None:
        # Vertaal actual seg → planned seg voor path-lookup (retracking)
        planned = self._state.get_planned_seg_for(train_id, segment_id)
        path    = self._trains[train_id].path
        try:
            idx = path.index(planned)
        except ValueError:
            return None
        if idx + 1 >= len(path):
            return None
        next_planned = path[idx + 1]
        # Vertaal planned → gekozen platform voor het volgende segment
        return self._state.get_chosen_seg(train_id, next_planned)

    def _compute_ready_time(self, train_id: int, segment_id: str, entry_time: float) -> float:
        # Bij between-station retracking is segment_id het GEKOZEN segment,
        # maar train.dynamics en timetable zijn geïndexeerd op het GEPLANDE segment.
        planned_segment_id = self._state.get_planned_seg_for(train_id, segment_id)
        duration = sample_duration(
            train=self._trains[train_id],
            segment=self._segments[segment_id],
            timetable=self._timetable,
            entry_time=entry_time,
            rng=self._rng,
            planned_segment_id=planned_segment_id,
        )
        self._state.record_sampled_duration(train_id, segment_id, duration)

        phys_ready = entry_time + duration
        min_exit   = self._dispatcher.min_exit_time(
            train_id=train_id,
            segment_id=segment_id,
            entry_time=entry_time,
            state=self._state,
            planned_segment_id=planned_segment_id,
        )
        return max(phys_ready, min_exit)


# =============================================================================
# Module-niveau hulpfuncties
# =============================================================================

def _is_passing(train, segment, planned_segment_id: str | None = None) -> bool:
    """
    True als de trein dit stationssegment passeert zonder te stoppen.

    Passing-segmenten worden gemodelleerd als instantane bezetting
    (1s) — voldoende voor headway-conflictdetectie zonder de aanliggende
    rijtijden te vervormen.

    planned_segment_id:
        Bij retracking is segment.id het gekozen platform; halt_indicators
        zijn geïndexeerd op het geplande segment. Gebruik planned_segment_id
        voor de halts_at-check zodat een geretrackte stop correct als stop
        (niet als passing) behandeld wordt.
    """
    plan_id = planned_segment_id if planned_segment_id is not None else segment.id
    return segment.is_station and not train.halts_at(plan_id)


def sample_duration(
    train,
    segment,
    timetable,
    entry_time: float,
    rng,
    planned_segment_id: str | None = None,
) -> float:
    """
    Sample de fysieke bezettingsduur van een segment.

    Drie gevallen:
      1. Stationspassing  → 1s
      2. Stationsstop     → 60s (dal) of 120s (ochtend-/avondspits)
                            Aanname: vaste minimale dwell-tijd per periode.
                            De C2-constraint (min_exit_time = scheduled_exit)
                            zorgt dat de trein nooit eerder vertrekt dan gepland,
                            dus de effectieve verblijftijd is altijd
                            max(60/120s, geplande dwell).
      3. Lijnsegment      → stochastisch via reality.sampling,
                            fallback op gepland tijdsverschil uit timetable.

    De ondergrens van 1s voorkomt dat events op exact dezelfde
    tijdstempel landen, wat de event-queue ordening verstoort.

    planned_segment_id:
        Optioneel. Bij between-station retracking wijkt segment.id (het
        gekozen fysieke segment) af van het geplande segment. train.dynamics
        en timetable zijn altijd geïndexeerd op geplande segmenten, dus
        gebruik plan_id voor die lookups en segment.id voor de
        distributie-lookup in sample_running_time.
    """
    # plan_id = geplande segment voor train.dynamics / timetable lookups;
    # segment.id = gekozen fysiek segment voor de running-time distributie.
    plan_id = planned_segment_id if planned_segment_id is not None else segment.id

    # 1. Stationspassing
    if _is_passing(train, segment, planned_segment_id=plan_id):
        return 1.0

    # 2. Stationsstop
    if segment.is_station:
        period = seconds_to_period(entry_time)
        return 120.0 if period in ("MORNING PEAK", "EVENING PEAK") else 60.0

    # 3. Lijnsegment
    # Gebruik plan_id voor alle lookups: distributie, dynamics én timetable-fallback.
    # Bij between-station retracking heeft het gekozen segment mogelijk geen
    # distribuitiedata, en de rijtijd van de omgekeerde richting is sowieso
    # niet representatief. Het geplande segment is de juiste referentie.
    sampled = sample_running_time(
        section    = plan_id,                     # gepland segment voor distributie
        train_type = train.train_subtype.value,
        dynamics   = train.dynamics_at(plan_id),  # gepland segment voor dynamics
        period     = seconds_to_period(entry_time),
        rng        = rng,
    )
    if sampled is not None:
        return max(1.0, float(sampled))

    # Fallback: gepland segment voor timetable-lookup
    row = timetable.get(train.id, plan_id)
    return max(1.0, float(row.exit_seconds - row.entry_seconds))
