"""
solver.py

Thin wrapper rond het MIP-model. Eén pad voor alle priority strategies —
de upgrade-logica zit in instance.py STEP 6 (exogene weights). De solver
hoeft enkel een feasibility check op de instance te doen en de gebouwde
Gurobi-output naar een Solution-object te vertalen.
"""

from model.mip_model import build_and_solve_model
from model.solution  import parse_solution, Solution
from config.settings import SOLVER_TIMEOUT_SECONDS, SOLVER_MIP_GAP

_EMPTY_SOLUTION = Solution(
    status    = "unknown",
    objective = None,
    runtime   = 0.0,
    entry     = {},
    exit      = {},
    delay     = {},
)

_VALID_STRATEGIES = ("static", "dynamic")


def solve(instance, priority_strategy="static", time_limit=SOLVER_TIMEOUT_SECONDS, verbose=True):

    if priority_strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown priority strategy: '{priority_strategy}'. "
            f"Use one of {_VALID_STRATEGIES}."
        )

    if not instance["T"]:
        return Solution(
            status    = "optimal",
            objective = 0.0,
            runtime   = 0.0,
            entry     = {},
            exit      = {},
            delay     = {},
        )

    try:
        model, entry, dep, delay, y, _ = build_and_solve_model(
            T            = instance["T"],
            S            = instance["S"],
            Ss           = instance["Ss"],
            Sl           = instance["Sl"],
            path         = instance["path"],
            sched_entry  = instance["sched_entry"],
            sched_exit   = instance["sched_exit"],
            runtime      = instance["runtime"],
            dwell        = instance["dwell"],
            conflicts    = instance["conflicts"],
            occupied     = instance["occupied"],
            fixed_entry  = instance["fixed_entry"],
            expected_exit = instance["expected_exit"],
            halts        = instance["halts"],
            weights      = instance["weights"],
            L            = instance["L"],
            current_time = instance["current_time"],
            time_limit   = time_limit,
            verbose      = verbose,
        )
        return parse_solution(model, entry, dep, delay)

    except Exception as e:
        print(f"[solver] Gurobi error: {e}")
        return _EMPTY_SOLUTION