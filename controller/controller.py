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
from model.solver   import solve


# =============================================================================
# FCFS fallback
# =============================================================================

def compute_fcfs_order(state, segments):
    """
    Computes First-Come-First-Served ordering per segment.

    For each segment, trains are ordered by their actual arrival time —
    the train that arrived first gets priority.

    Parameters
    ----------
    state    : SystemState   — current simulation state (partner's code)
    segments : list[Segment] — all Segment objects

    Returns
    -------
    dict {segment_id: [train_ids sorted by actual arrival time]}
    """
    fcfs = {}

    for seg in segments:
        # state.active_train_ids() — VAN DAUWS SystemState
        # state.remaining_path(train_id) — VAN DAUWS SystemState
        trains_on_seg = [
            t_id for t_id in state.active_train_ids()
            if seg.id in state.remaining_path(t_id)
        ]

        # state.actual_arrival(train_id, seg.id) — VAN DAUWS SystemState
        # confirm exact method name!!!!
        fcfs[seg.id] = sorted(
            trains_on_seg,
            key=lambda t_id: state.actual_arrival(t_id, seg.id)
        )

    return fcfs

# =============================================================================
# Controller result
# =============================================================================

class ControllerResult:
    """
    Returned by Controller.step() after each evaluation.

    Attributes
    ----------
    action      : str   — one of 'rescheduled', 'fcfs_fallback', 'skipped'
    solution    : Solution or None — MILP solution if solver was called and succeeded
    fcfs_order  : list or None     — FCFS train ordering if solver failed or was skipped
    runtime     : float            — time spent in this controller step (seconds)
    """

    def __init__(self, action, solution=None, fcfs_order=None, runtime=0.0):
        self.action     = action
        self.solution   = solution
        self.fcfs_order = fcfs_order
        self.runtime    = runtime

    def __repr__(self):
        return (
            f"ControllerResult(action={self.action}, "
            f"runtime={self.runtime:.3f}s)"
        )


# =============================================================================
# Controller
# =============================================================================

class Controller:
    """
    The rescheduling controller.

    Parameters
    ----------
    trigger   : BaseTrigger  — decides when to invoke the solver
    trains    : list[Train]  — all Train objects (passed to instance builder)
    segments  : list[Segment]— all Segment objects (passed to instance builder)
    timetable : Timetable    — original scheduled times (passed to instance builder)
    """

    def __init__(self, trigger, trains, segments, timetable):
        self.trigger   = trigger #periodic, event-driven of hybrid
        self.trains    = trains
        self.segments  = segments
        self.timetable = timetable

        # Simple counters for logging/reporting
        self._n_rescheduled   = 0   # number of times solver was called and succeeded
        self._n_fcfs_fallback = 0   # number of times FCFS fallback was used
        self._n_skipped       = 0   # number of times trigger said no

    # ------------------------------------------------------------------
    # Main entry point — called by the simulator at each trigger check
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
                runtime = time.time() - step_start
            )

        # Step 2 — Trigger fired: build MILP instance
        self._log(current_time, "TRIGGERED", "building instance...")

        instance = build_instance(
            state     = state,
            timetable = self.timetable,
            trains    = self.trains,
            segments  = self.segments,
        )

        # Step 3 — Call the solver
        solution = solve(instance)

        # Step 4a — Solver succeeded: return solution
        if solution.is_feasible(): # True wanneer status = optimal of timeout (van in solution.py)
            self.trigger.notify_rescheduled(current_time)
            self._n_rescheduled += 1
            self._log(
                current_time,
                "RESCHEDULED",
                f"status={solution.status}, "
                f"objective={solution.objective:.2f}, "
                f"solver_runtime={solution.runtime:.2f}s"
            )
            return ControllerResult(
                action  = "rescheduled",
                solution= solution,
                runtime = time.time() - step_start
            )
        # Step 4b — Solver failed: fall back to FCFS
        self.trigger.notify_evaluated(current_time)
        self._n_fcfs_fallback += 1
        fcfs_order = compute_fcfs_order(state, self.segments)
        self._log(
            current_time,
            "FCFS_FALLBACK",
            f"solver status={solution.status}, falling back to FCFS"
        )
        return ControllerResult(
            action     = "fcfs_fallback",
            fcfs_order = fcfs_order,
            runtime    = time.time() - step_start
        )

        # Step 5 — Apply solution or FCFS ordering to SystemState
        state.apply_solution(solution)

        state.apply_fcfs(fcfs_order) #TO DO: naam gelijkmaken met DAUWs SystemState

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, current_time: float, event: str, message: str):
        """
        Logs an important controller event to the console.
        Format: [t=Xs] EVENT — message
        Will be extended to write to utils/logger.py later.
        """
        print(f"[t={current_time:.0f}s] {event} — {message}")

    # ------------------------------------------------------------------
    # Summary stats (useful for experiments)
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Returns a summary of controller activity.
        Useful for saving experiment results.
        """
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