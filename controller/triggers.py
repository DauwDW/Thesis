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
segment vrij komt). Rijtijden worden gesampeld via sample_duration()
uit simulation/simulator.py — inclusief timetable-fallback voor segmenten
zonder empirische data.

Performance threshold (geheugenloos)
------------------------------------
Op elk evaluatiemoment wordt de drempel dynamisch berekend uit de huidige
toestand:

    threshold = avg_delay_per_active_train(state) + MC_DELAY_PER_TRAIN

De rollout retourneert de gemiddelde geprojecteerde eindvertraging per
actieve trein. De vergelijking

    rollout > threshold

betekent dus equivalent:

    rollout - avg_delay_now > MC_DELAY_PER_TRAIN

ofwel: "verwachten we meer dan MC_DELAY_PER_TRAIN seconden EXTRA vertraging
per trein bovenop wat al gerealiseerd is?"

Eigenschappen
- Geheugenloos: geen state-tracking over reschedules heen; de drempel
  past zich automatisch aan de huidige toestand aan.
- Geen ratchet: een succesvolle reschedule verlaagt de geprojecteerde
  eindvertraging → volgende evaluaties zien een kleinere (rollout - avg_delay)
  en triggeren niet onnodig.
- Populatie-robuust: zowel rollout als drempel middelen per trein, dus
  in- en uitstromende treinen verstoren de vergelijking niet.

Verantwoordelijkheid caller (controller.py)
-------------------------------------------
Na een effectieve reschedule moet notify_rescheduled(current_time, state)
aangeroepen worden om de frequentieklokken bij te werken. notify_evaluated()
wordt intern beheerd door elke subklasse in should_reschedule() — de caller
hoeft dit niet aan te roepen.
"""

from __future__ import annotations

import logging

import numpy as np
from domain import SegmentType

from simulation.simulator import sample_duration, _seconds_to_period
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
    Abstracte basisklasse voor alle triggerstrategieën.

    Verantwoordelijkheid caller
    ---------------------------
    - notify_rescheduled(current_time, state) aanroepen na elke effectieve reschedule.
    - notify_evaluated() wordt intern beheerd door elke subklasse in
      should_reschedule() — de caller hoeft dit niet aan te roepen.
    """

    def __init__(self):
        self._last_reschedule_time = None
        self._last_evaluation_time = None
        self._n_evaluated = 0

    @property
    def n_evaluated(self) -> int:
        """Aantal keren dat de MC effectief gedraaid heeft (beide frequentiechecks gepasseerd)."""
        return self._n_evaluated

    def notify_rescheduled(self, current_time: float, state):
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
        result = (self._time_since_reschedule(current_time) >= self.periodic_freq)
        self.notify_evaluated(current_time)
        return result

    def __repr__(self):
        return f"PeriodicTrigger(periodic_freq={self.periodic_freq}s)"


# =============================================================================
# 2. Event-Driven Trigger
# =============================================================================

class EventDrivenTrigger(BaseTrigger):
    """
    Event-driven trigger met dynamisch berekende drempel.

    Op elk evaluatiemoment:
        threshold = avg_delay_per_active_train(state) + mc_delay_per_train
        confidence = fractie rollouts met projected_avg_final_delay > threshold

    De solver wordt opgeroepen als confidence >= threshold_confidence.

    Parameters
    ----------
    trains               : dict[int, Train]
    segments             : dict[str, Segment]
    timetable            : Timetable
    event_driven_freq    : float  — min seconden tussen twee solver-aanroepen
    controller_freq      : float  — min seconden tussen twee evaluaties (<= event_driven_freq)
    threshold_confidence : float  — fractie rollouts boven drempel om solver te vuren
    mc_iterations        : int    — aantal rollouts per evaluatie
    mc_delay_per_train   : float  — toegestane extra gemiddelde vertraging per trein (s)
    rng                  : np.random.Generator | None
    """

    def __init__(
        self,
        trains,
        segments,
        timetable,
        event_driven_freq:    float,
        controller_freq:      float,
        threshold_confidence: float = THRESHOLD_CONFIDENCE,
        mc_iterations:        int   = MC_ITERATIONS,
        mc_delay_per_train:   float = MC_DELAY_PER_TRAIN,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        assert controller_freq <= event_driven_freq
        self._trains    = trains
        self._segments  = segments
        self._timetable = timetable

        self.event_driven_freq    = event_driven_freq
        self.controller_freq      = controller_freq
        self.threshold_confidence = threshold_confidence
        self.mc_iterations        = mc_iterations
        self.mc_delay_per_train   = mc_delay_per_train
        self._rng = rng if rng is not None else np.random.default_rng(SIMULATION_SEED)

    def should_reschedule(self, state, current_time: float) -> bool:
        if self._time_since_reschedule(current_time) < self.event_driven_freq:
            return False
        if self._time_since_evaluation(current_time) < self.controller_freq:
            return False

        self._n_evaluated += 1  # beide frequentiechecks gepasseerd: MC draait effectief

        result = self._estimate_confidence(state, current_time) >= self.threshold_confidence
        self.notify_evaluated(current_time)
        return result

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def _estimate_confidence(self, state, current_time: float) -> float:
        """
        Schat de fractie rollouts die de dynamische drempel overschrijden.

        De drempel wordt elke evaluatie opnieuw berekend uit de huidige
        toestand. Vergelijking is equivalent met:

            rollout - avg_delay_now > mc_delay_per_train
        """
        avg_delay = self._avg_delay_per_train(state)
        threshold = avg_delay + self.mc_delay_per_train

        worse = sum(
            1 for _ in range(self.mc_iterations)
            if self._run_mc_rollout(
                state,
                current_time,
                rng=np.random.default_rng(self._rng.integers(2**32)),
            ) > threshold
        )
        confidence = worse / self.mc_iterations

        n_active = len(state.active_train_ids())
        logger.debug(
            f"[t={current_time:.0f}] threshold={threshold:.1f}s/trein "
            f"(avg_delay={avg_delay:.1f}s, margin={self.mc_delay_per_train}s) "
            f"| fraction_above={confidence:.2f} / required={self.threshold_confidence} "
            f"| n_active={n_active}"
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
        FCFS op operationele tijden. Delay propagation via seg_free_at.

        Rijtijden worden gesampeld via sample_duration() — dezelfde functie
        als de simulator, inclusief timetable-fallback voor segmenten zonder
        empirische data (STATION-segmenten, ontbrekende dynamics, etc.).

        Returns
        -------
        float — geprojecteerde gemiddelde eindvertraging per actieve trein (s).
        """
        seg_free_at: dict[str, float] = {}
        start:       dict[int, float] = {}
        active_seg:  dict[int, str | None] = {}

        active_ids = list(state.active_train_ids())

        # --- Startfase: bepaal startpunt per trein ---
        for train_id in active_ids:
            train          = self._trains.get(train_id)
            remaining_path = state.remaining_path(train_id)
            if train is None or not remaining_path:
                continue

            current_seg = state.current_segment(train_id)
            active_seg[train_id] = current_seg

            if current_seg is not None and current_seg in remaining_path:
                try:
                    entry_time = state.actual_entry(train_id, current_seg)
                except KeyError:
                    entry_time = current_time

                elapsed = current_time - entry_time

                full_dur = sample_duration(
                    train      = train,
                    segment    = self._segments[current_seg],
                    timetable  = self._timetable,
                    entry_time = entry_time,
                    rng        = rng,
                )
                # moet het hier niet full_dur = sampled_duration zijn?

                proj_exit = current_time + max(1.0, full_dur - elapsed)
                seg_free_at[current_seg] = max(
                    seg_free_at.get(current_seg, 0.0), proj_exit        # hier zou je eig toch geen max() moeten gebruiken want het kan niet dat 2 treinen op hetzelfde moment dezelfde current_seg hebben
                )
                start[train_id] = proj_exit

            else:
                start[train_id] = self._last_exit(state, train, current_time)

        # --- Sorteer op geprojecteerde starttijd (FCFS-volgorde) ---
        active_ids = sorted(
            active_ids,
            key=lambda tid: start.get(tid, float('inf'))
        )

        # --- Hoofdlus: simuleer elk volgend segment ---
        total_delay = 0.0
        n_simulated = 0

        for train_id in active_ids:
            train          = self._trains.get(train_id)
            remaining_path = state.remaining_path(train_id)
            if train is None or not remaining_path or train_id not in start:
                continue

            t = start[train_id]

            current_seg = active_seg.get(train_id)
            path_to_simulate = (
                remaining_path[1:]
                if current_seg is not None
                and remaining_path
                and remaining_path[0] == current_seg
                else remaining_path
            )

            for seg_id in path_to_simulate:
                segment = self._segments.get(seg_id)
                if segment is None:
                    continue

                # FCFS: wacht tot segment vrij is
                t = max(t, seg_free_at.get(seg_id, 0.0))

                duration = sample_duration(
                    train      = train,
                    segment    = segment,
                    timetable  = self._timetable,
                    entry_time = t,
                    rng        = rng,
                )

                if segment.seg_type == SegmentType.STATION:         #within-station-passing check zou hier moeten !!!
                    # C2: niet vroeger vertrekken dan gepland #!!! dit is een hijkel punt blijkbaar
                    try:
                        t = max(
                            t + duration,
                            self._timetable.scheduled_exit(train_id, seg_id),
                        )
                    except KeyError:
                        t += duration
                else:
                    t += duration

                seg_free_at[seg_id] = t

            # Vertraging t.o.v. geplande exit van het laatste segment
            last_seg = remaining_path[-1]
            try:
                planned     = self._timetable.scheduled_exit(train_id, last_seg)
                total_delay += max(0.0, t - planned)
                n_simulated += 1
            except KeyError:
                pass

        return total_delay / max(1, n_simulated)

    def _last_exit(self, state, train, current_time: float) -> float:
        """actual_exit van het laatste afgeronde segment, of current_time als fallback."""
        for seg_id in reversed(train.path):
            try:
                return state.actual_exit(train.id, seg_id)
            except KeyError:
                continue
        return current_time

    def _avg_delay_per_train(self, state) -> float:
        """Gemiddelde huidige vertraging per actieve trein (seconden)."""
        try:
            active_ids = list(state.active_train_ids())
            if not active_ids:
                return 0.0
            total = sum(state.current_delay(t_id) for t_id in active_ids)
            return total / len(active_ids)
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

    n_evaluated delegeert naar de inner EventDrivenTrigger — periodieke
    deadline-aanroepen (waarbij de MC overgeslagen wordt) tellen niet mee
    in de reschedule rate.
    """

    def __init__(
        self,
        trains,
        segments,
        timetable,
        event_driven_freq:    float,
        controller_freq:      float,
        periodic_freq:        float,
        threshold_confidence: float = THRESHOLD_CONFIDENCE,
        mc_iterations:        int   = MC_ITERATIONS,
        mc_delay_per_train:   float = MC_DELAY_PER_TRAIN,
        rng: np.random.Generator | None = None,
    ):
        super().__init__()
        assert event_driven_freq < periodic_freq
        assert controller_freq <= event_driven_freq

        self.periodic_freq  = periodic_freq
        self._event_trigger = EventDrivenTrigger(
            trains               = trains,
            segments             = segments,
            timetable            = timetable,
            event_driven_freq    = event_driven_freq,
            controller_freq      = controller_freq,
            threshold_confidence = threshold_confidence,
            mc_iterations        = mc_iterations,
            mc_delay_per_train   = mc_delay_per_train,
            rng                  = rng,
        )

    @property
    def n_evaluated(self) -> int:
        """Delegeert naar inner trigger — periodieke deadline-aanroepen tellen niet mee."""
        return self._event_trigger._n_evaluated

    def notify_rescheduled(self, current_time: float, state):
        super().notify_rescheduled(current_time, state)
        self._event_trigger.notify_rescheduled(current_time, state)

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