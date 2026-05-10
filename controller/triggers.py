"""
Trigger logic voor de Controller: wanneer wordt de solver aangeroepen?

Drie strategieën (Vande Sompele 2024, Larsen & Pranzo 2019):
  1. Periodic     — vaste interval
  2. Event-Driven — Monte Carlo rollouts zonder rescheduling; solver vuurt als
                    genoeg rollouts de drempel overschrijden
  3. Hybrid       — combinatie: event-driven met periodieke harde deadline

Monte Carlo rollout
-------------------
Elke rollout simuleert het resterende pad van alle actieve treinen zonder
rescheduling. FCFS op operationele tijden: wie fysiek het eerst aankomt bij
een segment, gaat eerst. Delay propagation via seg_free_at (wanneer het
segment vrij komt). Rijtijden worden gesampeld via sample_running_time()
uit reality/sampling.py.

Performance threshold
---------------------
De drempel wordt bij elke evaluatie dynamisch bepaald als:

    performance_threshold = current_metric + n_active * MC_DELAY_PER_TRAIN

waarbij current_metric de huidige totale systeemvertraging is en n_active
het aantal actieve treinen op het moment van evaluatie. De threshold schaalt
mee met de netwerkbelasting: bij meer actieve treinen is een grotere absolute
toename nodig om de solver te activeren.

De threshold wordt gereset na elke reschedule zodat hij herberekend wordt
op basis van de nieuwe systeemtoestand.

Verantwoordelijkheid caller (controller.py)
-------------------------------------------
Na een effectieve reschedule moet notify_rescheduled() aangeroepen worden.
notify_evaluated() wordt intern beheerd door de trigger zelf — de caller
hoeft dit niet aan te roepen.
"""

from __future__ import annotations

import logging

import numpy as np
from domain import SegmentType

from simulation.simulator import sample_duration

from config.settings import (
    MC_DELAY_PER_TRAIN,
    MC_ITERATIONS,
    THRESHOLD_CONFIDENCE,
    SIMULATION_SEED,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base class
# =============================================================================

class BaseTrigger:
    """
    Abstracte basisklasse voor alle triggerstrategie.

    Verantwoordelijkheid caller
    ---------------------------
    - notify_rescheduled(current_time) aanroepen na elke effectieve reschedule.
    - notify_evaluated() wordt intern beheerd door elke subklasse in
      should_reschedule() — de caller hoeft dit niet aan te roepen.
    """

    def __init__(self):
        self._last_reschedule_time = None
        self._last_evaluation_time = None

    def notify_rescheduled(self, current_time: float):
        self._last_reschedule_time = current_time
        self._last_evaluation_time = current_time

    def notify_evaluated(self, current_time: float):
        self._last_evaluation_time = current_time

    def should_reschedule(self, state, current_time: float) -> bool:
        raise NotImplementedError

    def _time_since_reschedule(self, current_time: float) -> float:
        if self._last_reschedule_time is None:
            return float("inf")
        return current_time - self._last_reschedule_time

    def _time_since_evaluation(self, current_time: float) -> float:
        if self._last_evaluation_time is None:
            return float("inf")
        return current_time - self._last_evaluation_time


# =============================================================================
# 1. Periodic Trigger
# =============================================================================

class PeriodicTrigger(BaseTrigger):
    def __init__(self, periodic_freq: float):
        super().__init__()
        self.periodic_freq = periodic_freq
        self._last_reschedule_time = 0.0

    def should_reschedule(self, state, current_time: float) -> bool:
        result = self._time_since_reschedule(current_time) >= self.periodic_freq
        self.notify_evaluated(current_time)
        return result

    def __repr__(self):
        return f"PeriodicTrigger(periodic_freq={self.periodic_freq}s)"


# =============================================================================
# 2. Event-Driven Trigger
# =============================================================================

class EventDrivenTrigger(BaseTrigger):
    """
    Parameters
    ----------
    trains               : dict[int, Train]
    segments             : dict[str, Segment]
    timetable            : Timetable
    event_driven_freq    : float       — min seconden tussen twee solver-aanroepen
    controller_freq      : float       — min seconden tussen twee evaluaties (<= event_driven_freq)
    threshold_confidence : float       — fractie rollouts boven drempel om solver te vuren
    mc_iterations        : int         — aantal rollouts per evaluatie
    mc_delay_per_train   : float       — extra vertraging per actieve trein voor drempelberekening (s)
    performance_threshold: float|None  — drempelwaarde; None = lazy init bij eerste evaluatie,
                                         wordt gereset naar None na elke reschedule
    rng                  : np.random.Generator | None

    Noot: performance_threshold wordt gereset na notify_rescheduled() zodat de
    drempel herberekend wordt op basis van de toestand ná reschedule. Dit voorkomt
    dat een verouderde drempel de triggergevoeligheid structureel vertekent.

    Noot: elke rollout gebruikt een child-RNG afgeleid van self._rng voor
    reproduceerbaarheid — hernummering van rollouts verandert andere resultaten niet.
    """

    def __init__(
        self,
        trains,
        segments,
        timetable,
        event_driven_freq:     float,
        controller_freq:       float,
        threshold_confidence:  float = THRESHOLD_CONFIDENCE,
        mc_iterations:         int   = MC_ITERATIONS,
        mc_delay_per_train:    float = MC_DELAY_PER_TRAIN,
        performance_threshold: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        assert controller_freq <= event_driven_freq
        self._trains    = trains
        self._segments  = segments
        self._timetable = timetable

        self.event_driven_freq     = event_driven_freq
        self.controller_freq       = controller_freq
        self.threshold_confidence  = threshold_confidence
        self.mc_iterations         = mc_iterations
        self.mc_delay_per_train    = mc_delay_per_train
        self.performance_threshold = performance_threshold
        self._rng = rng if rng is not None else np.random.default_rng(SIMULATION_SEED)

    def notify_rescheduled(self, current_time: float):
        super().notify_rescheduled(current_time)
        # Reset drempel na reschedule zodat hij herberekend wordt op
        # de nieuwe toestand — verouderde drempel zou triggergevoeligheid
        # structureel kunnen vertekenen na een succesvolle of mislukte reschedule.
        self.performance_threshold = None

    def should_reschedule(self, state, current_time: float) -> bool:
        if self._time_since_reschedule(current_time) < self.event_driven_freq:
            return False
        if self._time_since_evaluation(current_time) < self.controller_freq:
            return False
        # Monte Carlo draait pas hier — notify_evaluated pas resetten na de MC
        result = self._estimate_confidence(state, current_time) >= self.threshold_confidence
        self.notify_evaluated(current_time)
        return result

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def _estimate_confidence(self, state, current_time: float) -> float:
        current_metric = self._total_delay(state)
        n_active       = len(state.active_train_ids())

        if self.performance_threshold is None:
            # Drempel = huidige systeemvertraging + n_active * mc_delay_per_train
            # Schaalt mee met de netwerkbelasting op het moment van evaluatie.
            self.performance_threshold = (
                current_metric + n_active * self.mc_delay_per_train
            )
            logger.debug(
                f"MC threshold initialiseerd op: {self.performance_threshold:.1f}s "
                f"(current={current_metric:.1f}s, "
                f"n_active={n_active}, "
                f"per_train={self.mc_delay_per_train}s)"
            )

        worse = sum(
            1 for i in range(self.mc_iterations)
            # child-RNG per rollout — reproduceerbaarheid gegarandeerd
            # ongeacht hernummering of volgorde van rollouts.
            if self._run_mc_rollout(
                state,
                current_time,
                rng=np.random.default_rng(self._rng.integers(2**32)),
            ) > self.performance_threshold
        )
        confidence = worse / self.mc_iterations
        logger.debug(
            f"MC: {worse}/{self.mc_iterations} boven drempel "
            f"({self.performance_threshold:.1f}s) -> confidence={confidence:.2f}"
        )
        return confidence

    def _run_mc_rollout(
        self,
        state,
        current_time: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Simuleert het resterende pad van alle actieve treinen zonder rescheduling.
        FCFS op operationele tijden: wie het eerst fysiek aankomt, gaat eerst.
        Delay propagation via seg_free_at.

        Parameters
        ----------
        state        : SystemState
        current_time : float
        rng          : np.random.Generator — per-rollout RNG voor reproduceerbaarheid

        Returns
        -------
        float — geprojecteerde totaalvertraging (seconden)
        """
        # seg_free_at[seg_id] = tijdstip waarop het segment vrij komt
        seg_free_at: dict[str, float] = {}

        # Startpunt per trein: wanneer is hij vrij voor zijn eerste resterende segment?
        start: dict[int, float] = {}

        # Bepaal per trein het startpunt én registreer seg_free_at voor het
        # huidig actieve segment. Het actieve segment wordt in de hoofdlus
        # overgeslagen om dubbele sampling te vermijden.
        active_seg: dict[int, str | None] = {}

        for train_id in state.active_train_ids():
            train          = self._trains.get(train_id)
            remaining_path = state.remaining_path(train_id)
            if train is None or not remaining_path:
                continue

            current_seg = state.current_segment(train_id)
            active_seg[train_id] = current_seg

            if current_seg is not None and current_seg in remaining_path:
                # Mid-segment: sample resterende duur
                try:
                    entry_time = state.actual_entry(train_id, current_seg)
                except KeyError:
                    entry_time = current_time
                elapsed  = current_time - entry_time
                full_dur = sample_duration(
                    train      = self._trains[train_id],
                    segment    = self._segments[current_seg],
                    timetable  = self._timetable,
                    entry_time = entry_time,
                    rng        = rng,
                )
                proj_exit = current_time + max(1.0, full_dur - elapsed)
                seg_free_at[current_seg] = max(seg_free_at.get(current_seg, 0.0), proj_exit)
                start[train_id] = proj_exit
            else:
                # Tussen segmenten: start vanaf actual_exit van laatste afgerond segment
                start[train_id] = self._last_exit(state, self._trains[train_id], current_time)

        # Simuleer elke trein over zijn resterende pad
        total_delay = 0.0

        for train_id in state.active_train_ids():
            train          = self._trains.get(train_id)
            remaining_path = state.remaining_path(train_id)
            if train is None or not remaining_path or train_id not in start:
                continue

            t = start[train_id]

            # Sla het huidig actieve segment over in de hoofdlus — het is al
            # gesimuleerd in de startfase hierboven. Zonder deze skip wordt het
            # segment dubbel gesimuleerd met een nieuwe sample_duration.
            current_seg = active_seg.get(train_id)
            path_to_simulate = (
                remaining_path[1:]
                if current_seg is not None
                and remaining_path
                and remaining_path[0] == current_seg
                else remaining_path
            )

            for seg_id in path_to_simulate:
                # FCFS: wacht tot segment vrij is
                t = max(t, seg_free_at.get(seg_id, 0.0))

                segment  = self._segments.get(seg_id)
                duration = sample_duration(
                    train      = train,
                    segment    = segment,
                    timetable  = self._timetable,
                    entry_time = t,
                    rng        = rng,
                )

                if segment is not None and segment.seg_type == SegmentType.STATION:
                    # C2: niet vroeger vertrekken dan gepland
                    try:
                        t = max(
                            t + duration,
                            self._timetable.scheduled_departure(train_id, seg_id),
                        )
                    except KeyError:
                        t += duration
                else:
                    t += duration

                seg_free_at[seg_id] = t

            # Vertraging t.o.v. geplande exit van het laatste segment
            last_seg = remaining_path[-1]
            try:
                planned     = self._timetable.scheduled_departure(train_id, last_seg)
                total_delay += max(0.0, t - planned)
            except KeyError:
                pass

        return total_delay

    def _last_exit(self, state, train, current_time: float) -> float:
        """
        actual_exit van het laatste afgeronde segment, of current_time als fallback.

        Noot: actual_exit gooit KeyError zowel als het segment nooit betreden is
        als als de exit nog None is (trein is er nog in). Beide gevallen worden
        correct overgeslagen door de try/except.
        """
        for seg_id in reversed(train.path):
            try:
                return state.actual_exit(train.id, seg_id)
            except KeyError:
                continue
        return current_time

    def _total_delay(self, state) -> float:
        try:
            return sum(state.current_delay(t_id) for t_id in state.active_train_ids())
        except AttributeError:
            return 0.0

    def __repr__(self):
        return (
            f"EventDrivenTrigger(event_driven_freq={self.event_driven_freq}s, "
            f"controller_freq={self.controller_freq}s, "
            f"threshold_confidence={self.threshold_confidence}, "
            f"mc_iterations={self.mc_iterations}, "
            f"mc_delay_per_train={self.mc_delay_per_train}s)"
        )


# =============================================================================
# 3. Hybrid Trigger
# =============================================================================

class HybridTrigger(BaseTrigger):
    """
    Event-driven met periodieke harde deadline.
    periodic_freq = maximale tijd tussen twee solver-aanroepen (moet > event_driven_freq).
    """

    def __init__(
        self,
        trains,
        segments,
        timetable,
        event_driven_freq:     float,
        controller_freq:       float,
        periodic_freq:         float,
        threshold_confidence:  float = THRESHOLD_CONFIDENCE,
        mc_iterations:         int   = MC_ITERATIONS,
        mc_delay_per_train:    float = MC_DELAY_PER_TRAIN,
        performance_threshold: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        assert event_driven_freq < periodic_freq
        assert controller_freq <= event_driven_freq

        self.periodic_freq  = periodic_freq
        self._event_trigger = EventDrivenTrigger(
            trains                = trains,
            segments              = segments,
            timetable             = timetable,
            event_driven_freq     = event_driven_freq,
            controller_freq       = controller_freq,
            threshold_confidence  = threshold_confidence,
            mc_iterations         = mc_iterations,
            mc_delay_per_train    = mc_delay_per_train,
            performance_threshold = performance_threshold,
            rng                   = rng,
        )

    def notify_rescheduled(self, current_time: float):
        super().notify_rescheduled(current_time)
        self._event_trigger.notify_rescheduled(current_time)

    def notify_evaluated(self, current_time: float):
        super().notify_evaluated(current_time)
        self._event_trigger.notify_evaluated(current_time)

    def should_reschedule(self, state, current_time: float) -> bool:
        if self._time_since_reschedule(current_time) >= self.periodic_freq:
            return True
        return self._event_trigger.should_reschedule(state, current_time)

    def __repr__(self):
        ed = self._event_trigger
        return (
            f"HybridTrigger(event_driven_freq={ed.event_driven_freq}s, "
            f"controller_freq={ed.controller_freq}s, "
            f"periodic_freq={self.periodic_freq}s, "
            f"threshold_confidence={ed.threshold_confidence}, "
            f"mc_iterations={ed.mc_iterations}, "
            f"mc_delay_per_train={ed.mc_delay_per_train}s)"
        )


# =============================================================================
# Factory
# =============================================================================

def make_trigger(strategy: str, **kwargs) -> BaseTrigger:
    """
    Geeft de juiste trigger terug op basis van strategie-string.

    'periodic'     : periodic_freq
    'event_driven' : trains, segments, timetable, event_driven_freq, controller_freq
    'hybrid'       : idem + periodic_freq
    """
    strategies = {
        "periodic":     PeriodicTrigger,
        "event_driven": EventDrivenTrigger,
        "hybrid":       HybridTrigger,
    }
    if strategy not in strategies:
        raise ValueError(
            f"Onbekende strategie '{strategy}'. Kies uit: {list(strategies)}"
        )
    return strategies[strategy](**kwargs)