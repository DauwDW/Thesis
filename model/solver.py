# """
# solver.py

# Wraps the MILP models and Gurobi solver.
# Takes instance parameters, calls the right model (static or dynamic priority),
# and returns a clean Solution object.
# """

# from model.mip_base    import build_and_solve_model as solve_static
# from model.mip_dynamic import build_and_solve_model as solve_dynamic
# from model.solution    import parse_solution, Solution

# _EMPTY_SOLUTION = Solution(
#     status   = "unknown",
#     objective= None,
#     runtime  = 0.0,
#     arrival  = {},
#     departure= {},
#     delay    = {},
#     ordering = {},
# )

# _VALID_STRATEGIES = ("static", "dynamic")


# def solve(instance, priority_strategy="static", time_limit=60, verbose=False):
#     """
#     Runs the MILP solver on a given instance.

#     Parameters
#     ----------
#     instance           : dict — output of instance.py build_instance()
#     priority_strategy  : str  — 'static' or 'dynamic'
#     time_limit         : int  — solver time limit in seconds (default 60)
#     verbose            : bool — print Gurobi output (default False)

#     Returns
#     -------
#     Solution object — always returns, never raises.
#     status='unknown' signals a Gurobi-level failure (no feasible solution found).
#     """

#     # Guard: unknown strategy — raise immediately (programmer error, not a solver error)
#     if priority_strategy not in _VALID_STRATEGIES:
#         raise ValueError(
#             f"Unknown priority strategy: '{priority_strategy}'. "
#             f"Use one of {_VALID_STRATEGIES}."
#         )

#     # Guard: empty train set — no rescheduling needed, return trivial solution
#     if not instance["T"]:
#         return Solution(
#             status   = "optimal",
#             objective= 0.0,
#             runtime  = 0.0,
#             arrival  = {},
#             departure= {},
#             delay    = {},
#             ordering = {},
#         )

#     if priority_strategy == "static":
#         return _solve_static(instance, time_limit, verbose)
#     else:
#         return _solve_dynamic(instance, time_limit, verbose)


# # ------------------------------------------------------------------------------
# # Private helpers
# # ------------------------------------------------------------------------------

# def _solve_static(instance, time_limit, verbose):
#     try:
#         model, a, d, delta, y, C, _ = solve_static(
#             T            = instance["T"],
#             Tp           = instance["Tp"],
#             Tf           = instance["Tf"],
#             S            = instance["S"],
#             Ss           = instance["Ss"],
#             Sl           = instance["Sl"],
#             path         = instance["path"],
#             sched_entry  = instance["sched_entry"],
#             sched_dep    = instance["sched_dep"],
#             RT           = instance["RT"],
#             DW           = instance["DW"],
#             H            = instance["H"],
#             h_stop       = instance["h_stop"],
#             w            = instance["w"],
#             L            = instance["L"],
#             in_execution = instance.get("in_execution", {}),
#             fix_arrival  = instance.get("fix_arrival", {}),
#             time_limit   = time_limit,
#             verbose      = verbose,
#         )
#         return parse_solution(model, a, d, delta, y, C)

#     except Exception as e:
#         print(f"[solver] Gurobi error (static): {e}")
#         return _EMPTY_SOLUTION


# def _solve_dynamic(instance, time_limit, verbose):
#     try:
#         model, a, d, delta, y, pdl, q, C, _ = solve_dynamic(
#             T            = instance["T"],
#             Tp           = instance["Tp"],
#             Tf           = instance["Tf"],
#             S            = instance["S"],
#             Ss           = instance["Ss"],
#             Sl           = instance["Sl"],
#             path         = instance["path"],
#             sched_entry  = instance["sched_entry"],
#             sched_dep    = instance["sched_dep"],
#             RT           = instance["RT"],
#             DW           = instance["DW"],
#             H            = instance["H"],
#             h_stop       = instance["h_stop"],
#             psl          = instance.get("psl",       {}),
#             gamma        = instance.get("gamma",      0),
#             epsilon      = instance.get("epsilon",    0),
#             delta_max    = instance.get("delta_max",  0),
#             L            = instance["L"],
#             in_execution = instance.get("in_execution", {}),
#             fix_arrival  = instance.get("fix_arrival",  {}),
#             time_limit   = time_limit,
#             verbose      = verbose,
#         )
#         return parse_solution(model, a, d, delta, y, C, pdl=pdl, q=q)

#     except Exception as e:
#         print(f"[solver] Gurobi error (dynamic): {e}")
#         return _EMPTY_SOLUTION
"""
solver.py (DEBUG VERSION)

Wraps the MILP models and Gurobi solver.
"""

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

    # =========================
    # 🔍 DEBUG: INSTANCE PRINT
    # =========================
    print("\n=== DEBUG INSTANCE ===")
    print("Aantal treinen:", len(instance["T"]))
    print("Aantal segmenten:", len(instance["S"]))

    fix_arr = instance.get("fix_arrival", {})
    in_exec = instance.get("in_execution", {})

    print("fix_arrival (eerste 5):", list(fix_arr.items())[:5])
    print("in_execution (eerste 5):", list(in_exec.items())[:5])

    # 🔥 check probleemgevallen
    for (t, s), a in fix_arr.items():
        sched = instance["sched_dep"][(t, s)]
        if a > sched:
            print(f"❌ PROBLEM: train {t}, seg {s} → fix_arrival={a} > sched_dep={sched}")

    print("=======================\n")

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
            in_execution = instance.get("in_execution", {}),
            fix_arrival  = instance.get("fix_arrival", {}),
            time_limit   = time_limit,
            current_time = instance.get("current_time"),
            verbose      = True,   # 👈 force Gurobi logs
        )

        # 🔍 DEBUG: status
        print(f"[DEBUG] Gurobi status code: {model.Status}")
        print(f"[DEBUG] Solutions found: {model.SolCount}")
        print(f"[DEBUG] Runtime: {model.Runtime:.2f}s")

        # 🔥 indien geen oplossing → probeer IIS
        if model.SolCount == 0:
            print("⚠️ No feasible solution — computing IIS...")
            try:
                model.computeIIS()
                model.write("debug_model.ilp")
                print("📁 IIS written to debug_model.ilp")
            except Exception as e:
                print(f"[DEBUG] IIS failed: {e}")

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
            in_execution = instance.get("in_execution", {}),
            fix_arrival  = instance.get("fix_arrival", {}),
            time_limit   = time_limit,
            current_time = instance.get("current_time"),
            verbose      = True,
        )

        print(f"[DEBUG] Gurobi status code: {model.Status}")
        print(f"[DEBUG] Solutions found: {model.SolCount}")
        print(f"[DEBUG] Runtime: {model.Runtime:.2f}s")

        if model.SolCount == 0:
            print("⚠️ No feasible solution — computing IIS...")
            try:
                model.computeIIS()
                model.write("debug_model.ilp")
                print("📁 IIS written to debug_model.ilp")
            except Exception as e:
                print(f"[DEBUG] IIS failed: {e}")

        return parse_solution(model, a, d, delta, y, C, pdl=pdl, q=q)

    except Exception as e:
        print(f"[solver] Gurobi error (dynamic): {e}")
        return _EMPTY_SOLUTION