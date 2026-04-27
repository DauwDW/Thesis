"""
controller.py

The Controller is the decision-making layer of the rescheduling framework.
It sits between the simulation and the solver, and is responsible for:

  1. Evaluating the trigger to decide if rescheduling is needed
  2. Building a MILP instance from the current SystemState
  3. Calling the solver
  4. Returning the solution to the simulator (which applies it to SystemState)
  5. Falling back to FCFS if the solver finds no feasible solution

The controller does NOT modify SystemState directly — it returns the solution
and lets the simulator handle state updates.

Logging
-------
Important events are logged to the console and can be extended to a file
via utils/logger.py once that module is built.

Usage
-----
    from controller.controller import Controller
    from controller.triggers import make_trigger

    trigger = make_trigger('hybrid', event_driven_freq=1800, controller_freq=900,
                            periodic_freq=3600, threshold_confidence=0.8)
    ctrl = Controller(trigger=trigger, trains=trains, segments=segments, timetable=timetable)

    result = ctrl.step(state, current_time)
    # result is a ControllerResult with .solution, .action, and .fcfs_order
"""

import time
from model.instance import build_instance
from model.solver   import run_solver


# =============================================================================
# FCFS fallback
# =============================================================================

def compute_fcfs_order(state, segments: dict) -> dict:
    active_ids = state.active_train_ids()
    remaining  = {t_id: set(state.remaining_path(t_id)) for t_id in active_ids}

    fcfs = {}
    for seg_id in segments:
        trains_on_seg = [
            t_id for t_id in active_ids
            if seg_id in remaining[t_id]
        ]
        if not trains_on_seg:
            continue

        # Sorteer enkel op actual_entry — treinen zonder entry horen hier niet thuis
        entered = []
        for t_id in trains_on_seg:
            try:
                entry = state.actual_entry(t_id, seg_id)
                entered.append((entry, t_id))
            except KeyError:
                pass  # nog niet op dit segment — niet opnemen in FCFS

        fcfs[seg_id] = [t_id for _, t_id in sorted(entered)]

    return fcfs


# =============================================================================
# Controller result
# =============================================================================

class ControllerResult:
    """
    Returned by Controller.step() after each evaluation.

    Attributes
    ----------
    action      : str              — one of 'rescheduled', 'fcfs_fallback', 'skipped'
    solution    : Solution or None — MILP solution if solver succeeded
    fcfs_order  : dict or None     — FCFS train ordering if solver failed
    runtime     : float            — time spent in this controller step (seconds)
    """

    def __init__(self, action, solution=None, fcfs_order=None, runtime=0.0):
        self.action     = action
        self.solution   = solution
        self.fcfs_order = fcfs_order
        self.runtime    = runtime

    def __repr__(self):
        return f"ControllerResult(action={self.action}, runtime={self.runtime:.3f}s)"


# =============================================================================
# Controller
# =============================================================================

class Controller:
    """
    The rescheduling controller.

    Parameters
    ----------
    trigger   : BaseTrigger        — decides when to invoke the solver
    trains    : dict[int, Train]   — all Train objects
    segments  : dict[str, Segment] — all Segment objects
    timetable : Timetable          — original scheduled times
    """

    def __init__(self, trigger, trains, segments, timetable):
        self.trigger   = trigger
        self.trains    = trains
        self.segments  = segments
        self.timetable = timetable

        self._n_rescheduled   = 0
        self._n_fcfs_fallback = 0
        self._n_skipped       = 0

    # ------------------------------------------------------------------
    # Main entry point — called by the simulator after each TrainExited
    # ------------------------------------------------------------------

    def step(self, state, current_time: float) -> ControllerResult:
        """
        Evaluates the trigger and acts accordingly.

        Parameters
        ----------
        state        : SystemState — current simulation state
        current_time : float       — current simulation time in seconds

        Returns
        -------
        ControllerResult with action, solution, fcfs_order, and runtime
        """
        step_start = time.time()

        # Step 1 — Ask the trigger
        if not self.trigger.should_reschedule(state, current_time):
            self.trigger.notify_evaluated(current_time)
            self._n_skipped += 1
            self._log(current_time, "SKIPPED", "trigger did not fire")
            return ControllerResult(
                action  = "skipped",
                runtime = time.time() - step_start,
            )

        # Step 2 — Trigger fired: build MILP instance
        self._log(current_time, "TRIGGERED", "building instance...")

        instance = build_instance(
            state        = state,
            timetable    = self.timetable,
            trains       = self.trains,
            segments     = self.segments,
            current_time = current_time,
        )
        # Step 3 — Call the solver
        solution = run_solver(instance)

        # Step 4a — Solver succeeded
        if solution.is_feasible():
            self.trigger.notify_rescheduled(current_time)
            self._n_rescheduled += 1
            self._log(
                current_time,
                "RESCHEDULED",
                f"status={solution.status}, "
                f"objective={solution.objective:.2f}, "
                f"solver_runtime={solution.runtime:.2f}s",
            )
            return ControllerResult(
                action  = "rescheduled",
                solution= solution,
                runtime = time.time() - step_start,
            )

        # Step 4b — Solver failed: fall back to FCFS
        self.trigger.notify_evaluated(current_time)
        self._n_fcfs_fallback += 1
        fcfs_order = compute_fcfs_order(state, self.segments)
        self._log(
            current_time,
            "FCFS_FALLBACK",
            f"solver status={solution.status}, falling back to FCFS",
        )
        return ControllerResult(
            action     = "fcfs_fallback",
            fcfs_order = fcfs_order,
            runtime    = time.time() - step_start,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, current_time: float, event: str, message: str):
        print(f"[t={current_time:.0f}s] {event} — {message}")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "n_rescheduled"   : self._n_rescheduled,
            "n_fcfs_fallback" : self._n_fcfs_fallback,
            "n_skipped"       : self._n_skipped,
            "total_steps"     : self._n_rescheduled + self._n_fcfs_fallback + self._n_skipped,
        }

    def __repr__(self):
        return (
            f"Controller("
            f"trigger={self.trigger}, "
            f"rescheduled={self._n_rescheduled}, "
            f"fcfs={self._n_fcfs_fallback}, "
            f"skipped={self._n_skipped})"
        )