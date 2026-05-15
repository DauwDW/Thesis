from model.mip_base    import build_and_solve_model as solve_static
from model.mip_dynamic import build_and_solve_model as solve_dynamic
from model.solution    import parse_solution, Solution

_EMPTY_SOLUTION = Solution(
    status    = "unknown",
    objective = None,
    runtime   = 0.0,
    arrival   = {},
    departure = {},
    delay     = {},
)

_VALID_STRATEGIES = ("static", "dynamic")


def solve(instance, priority_strategy="static", time_limit=60, verbose=False):

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
            arrival   = {},
            departure = {},
            delay     = {},
        )

    if priority_strategy == "static":
        return _solve_static(instance, time_limit, verbose)
    else:
        return _solve_dynamic(instance, time_limit, verbose)


# ------------------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------------------

def _solve_static(instance, time_limit, verbose):
    try:
        model, entry, dep, delay, y, final_segment = solve_static(
            T              = instance["T"],
            S              = instance["S"],
            Ss             = instance["Ss"],
            Sl             = instance["Sl"],
            path           = instance["path"],
            sched_entry    = instance["sched_entry"],
            sched_exit     = instance["sched_exit"],
            runtime        = instance["runtime"],
            dwell          = instance["dwell"],
            conflicts      = instance["conflicts"],
            occupied       = instance["occupied"],
            fixed_entry    = instance["fixed_entry"],
            actual_entries = instance["actual_entries"],
            weights        = instance["weights"],
            L              = instance["L"],
            current_time   = instance["current_time"],
            time_limit     = time_limit,
            verbose        = verbose,
        )
        return parse_solution(model, entry, dep, delay, y, instance["conflicts"])

    except Exception as e:
        print(f"[solver] Gurobi error (static): {e}")
        return _EMPTY_SOLUTION


def _solve_dynamic(instance, time_limit, verbose):
    try:
        # Reconstrueer H uit conflicts — altijd 0 (geen headway)
        H = {}
        for s, pairs in instance["conflicts"].items():
            for (i, j) in pairs:
                H[(i, j, s)] = 0
                H[(j, i, s)] = 0

        model, a, d, delta, y, pdl, q, C, _ = solve_dynamic(
            T            = instance["T"],
            Tp           = instance["Tp"],
            Tf           = instance["Tf"],
            S            = instance["S"],
            Ss           = instance["Ss"],
            Sl           = instance["Sl"],
            path         = instance["path"],
            sched_entry  = instance["sched_entry"],
            sched_dep    = instance["sched_exit"],
            RT           = instance["runtime"],
            DW           = instance["dwell"],
            H            = H,
            h_stop       = instance.get("h_stop", {}),
            psl          = instance.get("psl", {}),
            gamma        = instance.get("gamma", 0),
            epsilon      = instance.get("epsilon", 0),
            delta_max    = instance.get("delta_max", 0),
            L            = instance["L"],
            C            = instance["conflicts"],
            in_execution = instance["occupied"],
            fix_arrival  = instance["fixed_entry"],
            weights      = instance["weights"],   # exogeen berekend in instance.py
            time_limit   = time_limit,
            current_time = instance["current_time"],
            verbose      = verbose,
        )
        # pdl=None, q=None — parse_solution handles this gracefully
        return parse_solution(model, a, d, delta, y, C, pdl=pdl, q=q)

    except Exception as e:
        print(f"[solver] Gurobi error (dynamic): {e}")
        return _EMPTY_SOLUTION