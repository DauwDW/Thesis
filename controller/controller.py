from __future__ import annotations

import time
from dataclasses import dataclass

from model.instance import build_instance
from model.solver import solve


# =============================================================================
# FCFS fallback
# =============================================================================

def compute_fcfs_order(state, segments: dict) -> dict[str, list[int]]:
    """
    Simpele FCFS fallback.

    Per segment:
      - neem alle actieve treinen die het segment nog moeten doen
      - sorteer op actuele aanwezigheid / voortgang

    Belangrijk:
      Deze functie bepaalt enkel een ORDE.
      De simulator/dispatcher bepalen de effectieve timing.
    """
    active_ids = state.active_train_ids()

    remaining_paths = {
        train_id: set(state.remaining_path(train_id))
        for train_id in active_ids
    }

    result: dict[str, list[int]] = {}

    for segment_id in segments:

        candidates = [
            train_id
            for train_id in active_ids
            if segment_id in remaining_paths[train_id]
        ]

        if not candidates:
            continue

        ordering = []

        for train_id in candidates:

            # trein al op segment geweest?
            try:
                entry = state.actual_entry(train_id, segment_id)
                ordering.append((0, entry, train_id))
                continue
            except KeyError:
                pass

            # anders: huidige voortgang gebruiken
            delay = state.current_delay(train_id)

            ordering.append((1, delay, train_id))

        ordering.sort()

        result[segment_id] = [
            train_id
            for _, _, train_id in ordering
        ]

    return result


# =============================================================================
# Controller result
# =============================================================================

@dataclass(slots=True)
class ControllerResult:
    """
    Output van Controller.step().
    """

    action: str
    solution: object | None = None
    fcfs_order: dict | None = None
    runtime: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ControllerResult("
            f"action={self.action}, "
            f"runtime={self.runtime:.3f}s)"
        )


# =============================================================================
# Controller
# =============================================================================

class Controller:
    """
    Beslissingslaag tussen simulator en solver.

    Verantwoordelijkheden:
      1. trigger evalueren
      2. MILP-instance bouwen
      3. solver oproepen
      4. oplossing teruggeven aan simulator
      5. fallback naar FCFS indien nodig

    Belangrijk:
      De controller muteert NOOIT de simulator state direct.
    """

    def __init__(
        self,
        trigger,
        trains,
        segments,
        timetable,
        objective_strategy,
        weight_passenger,
        weight_freight,
        upgrade_weight,
        gamma,
    ) -> None:

        self.trigger = trigger
        self.trains = trains
        self.segments = segments
        self.timetable = timetable

        self.objective_strategy = objective_strategy

        self.weight_passenger = weight_passenger
        self.weight_freight = weight_freight
        self.upgrade_weight = upgrade_weight
        self.gamma = gamma

        self._solver_runtimes: list[float] = []

        self._n_rescheduled = 0
        self._n_fcfs_fallback = 0
        self._n_skipped = 0

    # =========================================================================
    # Main entry point
    # =========================================================================

    def step(self, state, current_time: float) -> ControllerResult:
        """
        Eén controller-evaluatie.
        """

        start = time.time()

        # ---------------------------------------------------------------------
        # 1. Trigger check
        # ---------------------------------------------------------------------

        if not self.trigger.should_reschedule(state, current_time):

            self._n_skipped += 1

            return ControllerResult(
                action="skipped",
                runtime=time.time() - start,
            )

        # ---------------------------------------------------------------------
        # 2. Build MILP instance
        # ---------------------------------------------------------------------

        instance = build_instance(
            state=state,
            timetable=self.timetable,
            trains=self.trains,
            segments=self.segments,
            current_time=current_time,
            priority_strategy=self.objective_strategy,
            weight_passenger=self.weight_passenger,
            weight_freight=self.weight_freight,
            upgrade_weight=self.upgrade_weight,
            gamma=self.gamma,
        )

        # ---------------------------------------------------------------------
        # 3. Solve
        # ---------------------------------------------------------------------

        solution = solve(
            instance,
            priority_strategy=self.objective_strategy,
        )

        # ---------------------------------------------------------------------
        # 4. Feasible solution
        # ---------------------------------------------------------------------

        if solution.is_feasible():

            self.trigger.notify_rescheduled(current_time, state)

            self._n_rescheduled += 1
            self._solver_runtimes.append(solution.runtime)

            self._log(
                current_time,
                "RESCHEDULED",
                (
                    f"status={solution.status}, "
                    f"objective={solution.objective:.2f}, "
                    f"solver_runtime={solution.runtime:.2f}s"
                ),
            )

            return ControllerResult(
                action="rescheduled",
                solution=solution,
                runtime=time.time() - start,
            )

        # ---------------------------------------------------------------------
        # 5. FCFS fallback
        # ---------------------------------------------------------------------

        self._n_fcfs_fallback += 1

        fcfs_order = compute_fcfs_order(
            state=state,
            segments=self.segments,
        )

        self._log(
            current_time,
            "FCFS_FALLBACK",
            f"solver_status={solution.status}",
        )

        return ControllerResult(
            action="fcfs_fallback",
            fcfs_order=fcfs_order,
            runtime=time.time() - start,
        )

    # =========================================================================
    # Logging
    # =========================================================================

    def _log(
        self,
        current_time: float,
        event: str,
        message: str,
    ) -> None:
        print(
            f"[t={current_time:.0f}s] "
            f"{event} — {message}"
        )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def summary(self) -> dict:
        return {
            "n_rescheduled":          self._n_rescheduled,
            "n_fcfs_fallback":        self._n_fcfs_fallback,
            "n_skipped":              self._n_skipped,
            "n_evaluated":            self.trigger.n_evaluated,
            "total_steps":            self._n_rescheduled + self._n_fcfs_fallback + self._n_skipped,
            "total_solver_runtime_s": sum(self._solver_runtimes),
        }

    def __repr__(self) -> str:

        return (
            f"Controller("
            f"strategy={self.objective_strategy}, "
            f"rescheduled={self._n_rescheduled}, "
            f"fcfs={self._n_fcfs_fallback}, "
            f"skipped={self._n_skipped})"
        )