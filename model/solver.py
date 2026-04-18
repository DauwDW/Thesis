"""
solver.py

Wraps the MILP models and Gurobi solver.
Takes instance parameters, calls the right model (static or dynamic priority),
and returns a clean Solution object.
"""

from model.mip_base    import build_and_solve_model as solve_static
from model.mip_dynamic import build_and_solve_model as solve_dynamic
from model.solution    import parse_solution


def run_solver(instance, priority_strategy="static", time_limit=60, verbose=False):
    """
    Runs the MILP solver on a given instance.

    Parameters:
    instance           : dict   — output of instance.py build_instance()
    priority_strategy  : str    — 'static' or 'dynamic'
    time_limit         : int    — solver time limit in seconds (default 60)
    verbose            : bool   — print Gurobi output (default False)
    """

    # Choose the right model based on priority strategy
    if priority_strategy == "static":
        model, a, d, delta, y, C, final_seg = solve_static(
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
            L           = instance["L"],
            in_execution = instance.get("in_execution", {}),
            fix_arrival  = instance.get("fix_arrival", {}),
            time_limit  = time_limit,
            verbose     = verbose)
        return parse_solution(model, a, d, delta, y, C)
        
    elif priority_strategy == "dynamic":
        model, a, d, delta, y, pdl, q, C, final_seg = solve_dynamic(
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
            psl          = instance["psl"],
            gamma        = instance["gamma"],
            epsilon      = instance["epsilon"],
            delta_max    = instance["delta_max"],
            L            = instance["L"],
            in_execution = instance.get("in_execution", {}),
            fix_arrival  = instance.get("fix_arrival", {}),
            time_limit   = time_limit,
            verbose      = verbose
        )
        return parse_solution(model, a, d, delta, y, C, pdl=pdl, q=q)
    else:
        raise ValueError(f"Unknown priority strategy: '{priority_strategy}'. Use 'static' or 'dynamic'.")
