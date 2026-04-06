"""
solver.py

Wraps the MILP models and Gurobi solver.
Takes instance parameters, calls the right model (static or dynamic priority),
and returns a clean Solution object.
"""

from model.mip_base    import build_and_solve_model as solve_static # NAAM AAN TE PASSEN VAN MILP DOC
from model.mip_dynamic import build_and_solve_model as solve_dynamic # NOG TE MAKEN
from model.solution    import parse_solution


def run_solver(instance, priority_strategy="static", time_limit=60, verbose=False):
    """
    Runs the MILP solver on a given instance.

    Parameters
    ----------
    instance           : dict   — output of instance.py build_instance()
    priority_strategy  : str    — 'static' or 'dynamic'
    time_limit         : int    — solver time limit in seconds (default 60)
    verbose            : bool   — print Gurobi output (default False)

    Returns
    -------
    Solution object
    """

    # -------------------------------------------------------------------------
    # Choose the right model based on priority strategy
    # -------------------------------------------------------------------------
    if priority_strategy == "static":
        solve = solve_static
    elif priority_strategy == "dynamic":
        solve = solve_dynamic
    else:
        raise ValueError(f"Unknown priority strategy: '{priority_strategy}'. Use 'static' or 'dynamic'.")

    # -------------------------------------------------------------------------
    # Run the solver
    # -------------------------------------------------------------------------
    model, a, d, delta, y, C, final_seg = solve(
        T           = instance["T"],
        Tp          = instance["Tp"],
        Tf          = instance["Tf"],
        S           = instance["S"],
        Ss          = instance["Ss"],
        Sl          = instance["Sl"],
        path        = instance["path"],
        sched_entry = instance["sched_entry"],
        sched_dep   = instance["sched_dep"],
        RT          = instance["RT"],
        DW          = instance["DW"],
        H           = instance["H"],
        h_stop      = instance["h_stop"],
        w           = instance["w"],
        L           = 1440, #aantal minuten in een dag
        time_limit  = time_limit,
        verbose     = verbose
    )

    # -------------------------------------------------------------------------
    # Parse and return the solution
    # -------------------------------------------------------------------------
    return parse_solution(model, a, d, delta, y, C)