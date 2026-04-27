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
"""

from __future__ import annotations

import logging

import numpy as np

from simulation.simulator import _seconds_to_period, sample_duration

from config.settings import (
    THRESHOLD_MULTIPLIER,
    MC_ITERATIONS,
    THRESHOLD_CONFIDENCE,
    SIMULATION_SEED,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Base class
# =============================================================================

class BaseTrigger:
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

    def should_reschedule(self, state, current_time: float) -> bool:
        return self._time_since_reschedule(current_time) >= self.periodic_freq

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
    performance_threshold: float|None  — drempelwaarde; None = lazy init
    rng                  : np.random.Generator | None
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
        self.performance_threshold = performance_threshold
        self._rng = rng if rng is not None else np.random.default_rng(SIMULATION_SEED)

    def should_reschedule(self, state, current_time: float) -> bool:
        if self._time_since_reschedule(current_time) < self.event_driven_freq:
            return False
        if self._time_since_evaluation(current_time) < self.controller_freq:
            return False
        return self._estimate_confidence(state, current_time) >= self.threshold_confidence

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def _estimate_confidence(self, state, current_time: float) -> float:
        current_metric = self._total_delay(state)

        if self.performance_threshold is None:
            self.performance_threshold = max(current_metric * THRESHOLD_MULTIPLIER, 1.0)
            logger.debug(f"MC threshold: {self.performance_threshold:.1f}s")

        worse = sum(
            1 for _ in range(self.mc_iterations)
            if self._run_mc_rollout(state, current_time) > self.performance_threshold
        )
        confidence = worse / self.mc_iterations
        logger.debug(f"MC: {worse}/{self.mc_iterations} boven drempel -> confidence={confidence:.2f}")
        return confidence

    def _run_mc_rollout(self, state, current_time: float) -> float:
        """
        Simuleert het resterende pad van alle actieve treinen zonder rescheduling.
        FCFS op operationele tijden: wie het eerst fysiek aankomt, gaat eerst.
        Delay propagation via seg_free_at.

        Returns
        -------
        float -- geprojecteerde totaalvertraging (seconden)
        """
        # seg_free_at[seg_id] = tijdstip waarop het segment vrij komt
        seg_free_at: dict[str, float] = {}

        # Startpunt per trein: wanneer is hij vrij voor zijn eerste resterende segment?
        start: dict[int, float] = {}

        for train_id in state.active_train_ids():
            train          = self._trains.get(train_id)
            remaining_path = state.remaining_path(train_id)
            if train is None or not remaining_path:
                continue

            current_seg = state.current_segment(train_id)
            if current_seg is not None and current_seg in remaining_path:
                # Mid-segment: sample resterende duur
                try:
                    entry_time = state.actual_entry(train_id, current_seg)
                except KeyError:
                    entry_time = current_time
                elapsed   = current_time - entry_time
                full_dur  = sample_duration(
                    train      = train,
                    segment    = self._segments[current_seg],
                    timetable  = self._timetable,
                    entry_time = entry_time,
                    rng        = self._rng,
                )
                proj_exit = current_time + max(1.0, full_dur - elapsed)
                seg_free_at[current_seg] = max(seg_free_at.get(current_seg, 0.0), proj_exit)
                start[train_id] = proj_exit
            else:
                # Tussen segmenten: start vanaf actual_exit van laatste afgerond segment
                start[train_id] = self._last_exit(state, train, current_time)

        # Simuleer elke trein over zijn resterende pad
        total_delay = 0.0

        for train_id in state.active_train_ids():
            train          = self._trains.get(train_id)
            remaining_path = state.remaining_path(train_id)
            if train is None or not remaining_path or train_id not in start:
                continue

            t = start[train_id]

            for seg_id in remaining_path:
                # FCFS: wacht tot segment vrij is
                t = max(t, seg_free_at.get(seg_id, 0.0))

                segment  = self._segments.get(seg_id)
                segment  = self._segments.get(seg_id)
                duration = sample_duration(
                    train      = train,
                    segment    = segment,
                    timetable  = self._timetable,
                    entry_time = t,
                    rng        = self._rng,
                )

                if segment is not None and segment.seg_type == SegmentType.STATION:
                    # C2: niet vroeger vertrekken dan gepland
                    try:
                        t = max(t + duration, self._timetable.scheduled_departure(train_id, seg_id))
                    except KeyError:
                        t += duration
                else:
                    t += duration

                seg_free_at[seg_id] = t

            # Vertraging t.o.v. geplande exit van het laatste segment
            try:
                planned = self._timetable.scheduled_departure(train_id, remaining_path[-1])
                total_delay += max(0.0, t - planned)
            except KeyError:
                pass

        return total_delay

    def _last_exit(self, state, train, current_time: float) -> float:
        """actual_exit van het laatste afgeronde segment, of current_time."""
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
            f"mc_iterations={self.mc_iterations})"
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
            f"mc_iterations={ed.mc_iterations})"
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
        raise ValueError(f"Onbekende strategie '{strategy}'. Kies uit: {list(strategies)}")
    return strategies[strategy](**kwargs)