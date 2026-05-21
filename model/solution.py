"""
solution.py

Parses the Gurobi model output into a clean Solution object.
The Solution object is used by the controller to apply the
rescheduling decisions back to the SystemState.

Variabelenamen consistent met mip_base/mip_dynamic:
  entry  (was: a)
  exit   (was: d)
  delay  (was: delta)
"""


from gurobipy import GRB


class Solution:
    """
    Stores the result of a single MILP solve.

    Attributes
    ----------
    status    : str         — 'optimal', 'timeout', 'infeasible', 'unknown'
    objective : float|None  — objective value (total weighted delay)
    runtime   : float       — solver runtime in seconds
    entry     : dict        — {(train_id, segment_id): entry time}
    exit      : dict        — {(train_id, segment_id): exit time}
    delay     : dict        — {(train_id, segment_id): delay in seconds}
    """

    def __init__(self, status, objective, runtime, entry, exit, delay):
        self.status    = status
        self.objective = objective
        self.runtime   = runtime
        self.entry     = entry
        self.exit      = exit
        self.delay     = delay

    def is_feasible(self) -> bool:
        """True als de solver een haalbare oplossing heeft gevonden."""
        return self.status in ("optimal")   # timeout weglaten, anders , "timeout" toevoegen

    def entry_time(self, train_id, segment_id) -> float | None:
        return self.entry.get((train_id, segment_id))

    def exit_time(self, train_id, segment_id) -> float | None:
        return self.exit.get((train_id, segment_id))

    def delay_at(self, train_id, segment_id) -> float | None:
        return self.delay.get((train_id, segment_id))

    def __repr__(self) -> str:
        obj_str = f"{self.objective:.2f}" if self.objective is not None else "None"
        return (
            f"Solution(status={self.status}, "
            f"objective={obj_str}, "
            f"runtime={self.runtime:.2f}s)"
        )


def parse_solution(model, entry, dep, delay) -> Solution:
    """
    Parses Gurobi model output into a Solution object.

    Parameters
    ----------
    model : gurobipy.Model
    entry : dict {(train_id, seg_id): Var}  — entry-tijden
    dep   : dict {(train_id, seg_id): Var}  — exit-tijden
    delay : dict {(train_id, seg_id): Var}  — vertragingen
    """

    # --- Status ---
    status_code = model.Status

    if status_code == GRB.OPTIMAL:
        status = "optimal"
    elif status_code == GRB.TIME_LIMIT and model.SolCount > 0:
        status = "timeout"
    elif status_code == GRB.INFEASIBLE:
        status = "infeasible"
    else:
        status = "unknown"

    # --- Geen haalbare oplossing ---
    if status in ("infeasible", "unknown"):
        return Solution(
            status=status,
            objective=None,
            runtime=model.Runtime,
            entry={},
            exit={},
            delay={},
        )

    # --- Haalbare oplossing: extraheer variabelwaarden ---
    return Solution(
        status    = status,
        objective = model.ObjVal,
        runtime   = model.Runtime,
        entry     = {key: var.X for key, var in entry.items()},
        exit      = {key: var.X for key, var in dep.items()},
        delay     = {key: var.X for key, var in delay.items()},
    )