from model.mip_base    import build_and_solve_model as solve_static
from model.mip_dynamic import build_and_solve_model as solve_dynamic
from model.solution    import parse_solution, Solution

_EMPTY_SOLUTION = Solution(
    status   = "unknown",
    objective= None,
    runtime  = 0.0,
    arrival  = {},
    departure= {},
    delay    = {},
    ordering = {},
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
            status   = "optimal",
            objective= 0.0,
            runtime  = 0.0,
            arrival  = {},
            departure= {},
            delay    = {},
            ordering = {},
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
        model, a, d, delta, y, C, _ = solve_static(
            T            = instance["T"],
            Tp           = instance["Tp"],
            Tf           = instance["Tf"],
            S            = instance["S"],
            Ss           = instance["Ss"],
            Sl           = instance["Sl"],
            path         = instance["path"],
            sched_entry  = instance["sched_entry"],
            sched_dep    = instance["sched_dep"],
            RT           = instance["RT"],
            DW           = instance["DW"],
            H            = instance["H"],
            h_stop       = instance["h_stop"],
            w            = instance["w"],
            L            = instance["L"],
            C            = instance.get("C", None),
            in_execution = instance.get("in_execution", {}),
            fix_arrival  = instance.get("fix_arrival", {}),
            time_limit   = time_limit,
            current_time = instance.get("current_time"),
            verbose      = verbose,
        )
        return parse_solution(model, a, d, delta, y, C)

    except Exception as e:
        print(f"[solver] Gurobi error (static): {e}")
        return _EMPTY_SOLUTION


def _solve_dynamic(instance, time_limit, verbose):
    try:
        model, a, d, delta, y, pdl, q, C, _ = solve_dynamic(
            T            = instance["T"],
            Tp           = instance["Tp"],
            Tf           = instance["Tf"],
            S            = instance["S"],
            Ss           = instance["Ss"],
            Sl           = instance["Sl"],
            path         = instance["path"],
            sched_entry  = instance["sched_entry"],
            sched_dep    = instance["sched_dep"],
            RT           = instance["RT"],
            DW           = instance["DW"],
            H            = instance["H"],
            h_stop       = instance["h_stop"],
            psl          = instance.get("psl", {}),
            gamma        = instance.get("gamma", 0),
            epsilon      = instance.get("epsilon", 0),
            delta_max    = instance.get("delta_max", 0),
            L            = instance["L"],
            C            = instance.get("C", None),
            in_execution = instance.get("in_execution", {}),
            fix_arrival  = instance.get("fix_arrival", {}),
            time_limit   = time_limit,
            current_time = instance.get("current_time"),
            verbose      = verbose,
        )
        return parse_solution(model, a, d, delta, y, C, pdl=pdl, q=q)

    except Exception as e:
        print(f"[solver] Gurobi error (dynamic): {e}")
        return _EMPTY_SOLUTION