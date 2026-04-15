"""
UITLEG triggers.py

Defines the trigger logic used by the Controller to decide when to invoke the Solver.

Three rescheduling strategies are supported, following Mariska Vande Sompele (2024)
and Larsen & Pranzo (2019):

  1. Periodic   — solver is called every fixed interval (periodic_freq seconds)
  2. Event-Driven — controller evaluates system state periodically; solver is called
                    only if Monte Carlo simulation predicts performance below threshold
  3. Hybrid     — combines both: event-driven evaluation with a periodic hard deadline

All time values are in seconds (simulation clock).

Usage
-----
    from controller.triggers import PeriodicTrigger, EventDrivenTrigger, HybridTrigger

    trigger = HybridTrigger(
        event_driven_freq=1800,
        controller_freq=900,
        periodic_freq=3600,
        threshold_confidence=0.8,
        mc_iterations=5,
    )

    if trigger.should_reschedule(state, current_time):
        solution = solver.solve(instance)
        trigger.notify_rescheduled(current_time)
    else:
        trigger.notify_evaluated(current_time)
"""

import random

from config.settings import (
    THRESHOLD_MULTIPLIER,
    MC_DELAY_PROBABILITY,
    MC_DELAY_MAX_SECONDS,
    MC_ITERATIONS,
    THRESHOLD_CONFIDENCE,
)

# =============================================================================
# Base class
# =============================================================================

class BaseTrigger:
    """
    Abstract base for all trigger types.
    Subclasses implement should_reschedule().
    """

    def __init__(self):
        self._last_reschedule_time = None   # last time solver was actually called
        self._last_evaluation_time = None   # last time controller evaluated state

    def notify_rescheduled(self, current_time: float):
        """Call this after the solver has been invoked."""
        self._last_reschedule_time = current_time
        self._last_evaluation_time = current_time

    def notify_evaluated(self, current_time: float):
        """Call this when the controller evaluated but did NOT invoke the solver."""
        self._last_evaluation_time = current_time

    def should_reschedule(self, state, current_time: float) -> bool:
        raise NotImplementedError

    def _time_since_reschedule(self, current_time: float) -> float:
        """Seconds since the last solver invocation. Returns inf if never called."""
        if self._last_reschedule_time is None:
            return float("inf")
        return current_time - self._last_reschedule_time

    def _time_since_evaluation(self, current_time: float) -> float:
        """Seconds since the last controller evaluation. Returns inf if never evaluated."""
        if self._last_evaluation_time is None:
            return float("inf")
        return current_time - self._last_evaluation_time


# =============================================================================
# 1. Periodic Trigger
# =============================================================================

class PeriodicTrigger(BaseTrigger):
    """
    Invokes the solver at a fixed time interval, regardless of system state.

    Parameters
    ----------
    periodic_freq : float
        Minimum seconds between two solver invocations.
    """

    def __init__(self, periodic_freq: float):
        super().__init__()
        self.periodic_freq = periodic_freq

    def should_reschedule(self, state, current_time: float) -> bool:
        """
        Returns True if enough time has passed since the last reschedule.
        """
        return self._time_since_reschedule(current_time) >= self.periodic_freq

    def __repr__(self):
        return f"PeriodicTrigger(periodic_freq={self.periodic_freq}s)"


# =============================================================================
# 2. Event-Driven Trigger
# =============================================================================

class EventDrivenTrigger(BaseTrigger):
    """
    Invokes the solver only when the controller estimates the system is likely
    to perform below a threshold, based on Monte Carlo simulation.

    Parameters
    ----------
    event_driven_freq : float
        Minimum seconds between two solver invocations.
    controller_freq : float
        Minimum seconds between two controller evaluations.
        Must be <= event_driven_freq.
    threshold_confidence : float
        Probability [0, 1] required to trigger the solver.
        E.g. 0.8 means: invoke solver if P(metric > threshold) >= 0.8.
    mc_iterations : int
        Number of Monte Carlo roll-outs used to estimate P(metric > threshold).
    performance_threshold : float or None
        The reference performance value. If None, it is estimated from state
        on first call (average delay across all trains).
    """

    def __init__(
        self,
        event_driven_freq: float,
        controller_freq: float,
        threshold_confidence: float = THRESHOLD_CONFIDENCE,
        mc_iterations: int = MC_ITERATIONS,
        performance_threshold: float = None,
    ):
        super().__init__()
        assert controller_freq <= event_driven_freq, (
            "controller_freq must be <= event_driven_freq"
        )
        self.event_driven_freq       = event_driven_freq
        self.controller_freq         = controller_freq
        self.threshold_confidence    = threshold_confidence
        self.mc_iterations           = mc_iterations
        self.performance_threshold   = performance_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def should_reschedule(self, state, current_time: float) -> bool:
        """
        Returns True if:
          (a) enough time has passed since the last reschedule, AND
          (b) enough time has passed since the last evaluation, AND
          (c) Monte Carlo estimate says P(metric > threshold) >= threshold_confidence
        """
        # Gate 1: minimum interval since last reschedule
        if self._time_since_reschedule(current_time) < self.event_driven_freq:
            return False

        # Gate 2: minimum interval since last evaluation
        if self._time_since_evaluation(current_time) < self.controller_freq:
            return False

        # Gate 3: stochastic evaluation
        confidence = self._estimate_confidence(state, current_time)
        return confidence >= self.threshold_confidence

    # ------------------------------------------------------------------
    # Monte Carlo evaluation
    # ------------------------------------------------------------------

    def _estimate_confidence(self, state, current_time: float) -> float:
        """
        Runs mc_iterations light roll-outs of the system without rescheduling,
        counts how many result in a metric worse than performance_threshold,
        and returns that fraction as the estimated confidence.

        The metric used is: total sum of current delays across all active trains.
        If performance_threshold is not set, it is initialised here as 1.5x
        the current total delay (i.e. we trigger if the system is trending worse).
        """
        current_metric = self._total_delay(state)

        # Initialise threshold lazily on first call
        if self.performance_threshold is None:
            # Default: set threshold at 1.5x the current state's total delay.
            # In a real deployment this would be calibrated from historical baselines.
            #TO DO: de 1.5 vervangen average delay wanneer er niet rescheduled wordt. anders best gewoon niet reschedulen
            self.performance_threshold = max(current_metric * THRESHOLD_MULTIPLIER, 1.0)

        worse_count = 0
        for _ in range(self.mc_iterations):
            projected = self._simulate_no_reschedule(state, current_metric)
            if projected > self.performance_threshold:
                worse_count += 1

        return worse_count / self.mc_iterations

    def _total_delay(self, state) -> float:
        """
        Returns the sum of current delays across all active (non-finished) trains.
        Assumes state exposes current_delay(train_id) and active_train_ids().
        """
        try:
            return sum(
                state.current_delay(t_id)
                for t_id in state.active_train_ids()
            )
        except AttributeError:
            # Retourneert 0 als SystemState geen current_delay of active_train_ids heeft
            return 0.0

    def _simulate_no_reschedule(self, state, current_metric: float) -> float:
        """
        Light stochastic projection: adds random perturbations to the current
        total delay to simulate a future without rescheduling.

        This is intentionally simple (no full event loop) to remain fast.
        A more accurate version could run a truncated event simulation.

        The noise model: each active train independently has a 30% chance of
        accumulating additional delay drawn from U(0, 5) minutes.
        """
        try:
            n_active = len(list(state.active_train_ids()))
        except AttributeError:
            n_active = 1

        additional = sum(
            random.uniform(0, MC_DELAY_MAX_SECONDS) for _ in range(n_active) if random.random() < MC_DELAY_PROBABILITY)
        return current_metric + additional

    def __repr__(self):
        return (
            f"EventDrivenTrigger("
            f"event_driven_freq={self.event_driven_freq}s, "
            f"controller_freq={self.controller_freq}s, "
            f"threshold_confidence={self.threshold_confidence}, "
            f"mc_iterations={self.mc_iterations})"
        )


# =============================================================================
# 3. Hybrid Trigger
# =============================================================================

class HybridTrigger(BaseTrigger):
    """
    Combines a periodic hard deadline with event-driven evaluation.

    The solver is invoked when EITHER:
      (a) The periodic deadline is reached (periodic_freq seconds since last reschedule), OR
      (b) The event-driven controller fires (as in EventDrivenTrigger)

    This means:
      - event_driven_freq  = minimum time between reschedules
      - periodic_freq      = maximum time between reschedules (hard deadline)
      - event_driven_freq must be < periodic_freq

    Parameters
    ----------
    event_driven_freq    : float  — min seconds between reschedules
    controller_freq      : float  — min seconds between controller evaluations
    periodic_freq        : float  — max seconds between reschedules (hard deadline)
    threshold_confidence : float  — P threshold for event-driven arm
    mc_iterations        : int    — Monte Carlo samples for event-driven arm
    performance_threshold: float  — reference metric value (None = auto)
    """

    def __init__(
        self,
        event_driven_freq: float,
        controller_freq: float,
        periodic_freq: float,
        threshold_confidence: float = THRESHOLD_CONFIDENCE,
        mc_iterations: int = MC_ITERATIONS,
        performance_threshold: float = None,
    ):
        super().__init__()
        assert event_driven_freq < periodic_freq, (
            "event_driven_freq must be strictly less than periodic_freq"
        )
        assert controller_freq <= event_driven_freq, (
            "controller_freq must be <= event_driven_freq"
        )

        self.periodic_freq = periodic_freq

        # Reuse EventDrivenTrigger for the stochastic arm
        self._event_trigger = EventDrivenTrigger(
            event_driven_freq=event_driven_freq,
            controller_freq=controller_freq,
            threshold_confidence=threshold_confidence,
            mc_iterations=mc_iterations,
            performance_threshold=performance_threshold,
        )

    # Delegate state tracking to both sub-triggers:

    def notify_rescheduled(self, current_time: float):
        super().notify_rescheduled(current_time)
        self._event_trigger.notify_rescheduled(current_time)

    def notify_evaluated(self, current_time: float):
        super().notify_evaluated(current_time)
        self._event_trigger.notify_evaluated(current_time)

    # Main entry point

    def should_reschedule(self, state, current_time: float) -> bool:
        """
        Returns True if the periodic deadline is there OR the event-driven part fires.
        """
        # Periodic part: hard deadline
        if self._time_since_reschedule(current_time) >= self.periodic_freq:
            return True

        # Event-driven part
        return self._event_trigger.should_reschedule(state, current_time)

    def __repr__(self):
        ed = self._event_trigger
        return (
            f"HybridTrigger("
            f"event_driven_freq={ed.event_driven_freq}s, "
            f"controller_freq={ed.controller_freq}s, "
            f"periodic_freq={self.periodic_freq}s, "
            f"threshold_confidence={ed.threshold_confidence}, "
            f"mc_iterations={ed.mc_iterations})"
        )


# =============================================================================
# Trigger maker
# =============================================================================

def make_trigger(strategy: str, **kwargs) -> BaseTrigger:
    """
    Parameters
    ----------
    strategy : str
        One of 'periodic', 'event_driven', 'hybrid'
    **kwargs :
        Passed directly to the corresponding trigger class.

    Examples
    --------
    >>> t = make_trigger('periodic', periodic_freq=900)
    >>> t = make_trigger('event_driven', event_driven_freq=1800, controller_freq=900,
    ...                   threshold_confidence=0.6)
    >>> t = make_trigger('hybrid', event_driven_freq=1800, controller_freq=900,
    ...                   periodic_freq=3600, threshold_confidence=0.8)
    """
    strategies = { #woordenboek van strategieën naar klassen 
        "periodic":     PeriodicTrigger,
        "event_driven": EventDrivenTrigger,
        "hybrid":       HybridTrigger,
    }

    #als strategy valid, retourneert bijpassende trigger met zijn parameters. anders error
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategies)}")
    return strategies[strategy](**kwargs)