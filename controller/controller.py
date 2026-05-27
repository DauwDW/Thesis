from __future__ import annotations

import time
from dataclasses import dataclass

from model.instance import build_instance
from model.solver import solve
from model.fcfs import compute_fcfs_objective



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
        duration_statistic:"scheduled",
        subtype_weights=None,
        min_objective_improvement: float = -10000.0,
        platform_alternatives: dict | None = None,
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
        self.duration_statistic = duration_statistic

        self.subtype_weights = subtype_weights

        # Drempel: MIP-oplossing wordt enkel toegepast als de objective-
        # verbetering t.o.v. de FCFS-baseline minstens deze waarde is.
        # 0.0 (default) = uitgeschakeld: elke feasible oplossing wordt toegepast.
        self.min_objective_improvement = min_objective_improvement

        # Platform-alternatieven voor retracking
        self.platform_alternatives = platform_alternatives or {}

        self._solver_runtimes: list[float] = []
        # MIP-objectiefwaarden (gewogen totale projectievertraging) per toegepaste reschedule.
        # Gebruikt voor de empirische min_objective_threshold kalibratie (sectie 6).
        self._solution_objectives: list[float] = []

        self._n_rescheduled = 0
        self._n_fcfs_fallback = 0
        self._n_skipped = 0
        self._n_skipped_no_improvement = 0
        self._n_platform_switches = 0
        self._consecutive_failures = 0

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
            subtype_weights=self.subtype_weights,
            gamma=self.gamma,
            duration_statistic=self.duration_statistic,
            platform_alternatives=self.platform_alternatives,
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
            self._consecutive_failures = 0

            # ---------------------------------------------------------
            # FCFS-vergelijking: skip toepassing als verbetering te klein
            # ---------------------------------------------------------
            if self.min_objective_improvement > 0.0:
                fcfs_objective, fcfs_converged = compute_fcfs_objective(instance)
                improvement = fcfs_objective - solution.objective

                # Verbetering te klein én FCFS is betrouwbaar → skip.
                # Als FCFS niet convergeert is de schatting onbetrouwbaar
                # (complexe conflictsituatie, mogelijk deadlock-risico) →
                # pas de MIP-oplossing sowieso toe.
                if improvement < self.min_objective_improvement and fcfs_converged:
                    self._n_skipped_no_improvement += 1
                    # Reset trigger-klok: een 'no improvement' kost evenveel tijd
                    # als een toegepaste reschedule, dus tel beide gelijk.
                    self.trigger.notify_rescheduled(current_time, state)

                    self._log(
                        current_time,
                        "SKIPPED_NO_IMPROVEMENT",
                        (
                            f"MIP={solution.objective:.0f}s, "
                            f"FCFS={fcfs_objective:.0f}s, "
                            f"improvement={improvement:.0f}s "
                            f"< threshold={self.min_objective_improvement:.0f}s"
                        ),
                    )

                    return ControllerResult(
                        action="skipped_no_improvement",
                        runtime=time.time() - start,
                    )

                if not fcfs_converged:
                    self._log(
                        current_time,
                        "FCFS_NO_CONVERGENCE",
                        (
                            f"MIP={solution.objective:.0f}s, "
                            f"FCFS={fcfs_objective:.0f}s (onbetrouwbaar) — "
                            f"MIP-oplossing toch toegepast"
                        ),
                    )

            self.trigger.notify_rescheduled(current_time, state)

            self._n_rescheduled += 1
            self._n_platform_switches += solution.n_platform_switches
            self._solver_runtimes.append(solution.runtime)
            if solution.objective is not None:
                self._solution_objectives.append(solution.objective)

            self._log(
                current_time,
                "RESCHEDULED",
                (
                    f"status={solution.status}, "
                    f"objective={solution.objective:.2f}, "
                    f"solver_runtime={solution.runtime:.2f}s, "
                    f"platform_switches={solution.n_platform_switches}"
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
        self._consecutive_failures += 1
        
        # Reset frequency clock: een mislukte solver-call kost evenveel tijd
        # als een succesvolle, dus tel ze gelijk voor de trigger.
        self.trigger.notify_rescheduled(current_time, state)



        return ControllerResult(
            action="no_solution",
            runtime=time.time() - start,
        )
    
    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

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
            "n_rescheduled":            self._n_rescheduled,
            "n_fcfs_fallback":          self._n_fcfs_fallback,
            "n_skipped":                self._n_skipped,
            "n_skipped_no_improvement": self._n_skipped_no_improvement,
            "n_platform_switches":      self._n_platform_switches,
            "n_evaluated":              self.trigger.n_evaluated,
            "total_steps": (
                self._n_rescheduled
                + self._n_fcfs_fallback
                + self._n_skipped
                + self._n_skipped_no_improvement
            ),
            "total_solver_runtime_s":   sum(self._solver_runtimes),
            "solution_objectives":       list(self._solution_objectives),
            "max_consecutive_failures": self._consecutive_failures,
        }

    def __repr__(self) -> str:

        return (
            f"Controller("
            f"strategy={self.objective_strategy}, "
            f"rescheduled={self._n_rescheduled}, "
            f"fcfs={self._n_fcfs_fallback}, "
            f"skipped={self._n_skipped})"
        )