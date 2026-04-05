"""
solution.py

Parses the Gurobi model output into a clean Solution object.
The Solution object is used by the controller to apply the
rescheduling decisions back to the SystemState.
"""

from gurobipy import GRB


class Solution:
    """
    Stores the result of a single MILP solve.

    Attributes
    ----------
    status          : str   — 'optimal', 'timeout', 'infeasible', or 'unknown'
    objective       : float — objective value (total weighted delay)
    runtime         : float — solver runtime in seconds
    arrival         : dict  — {(train_id, segment_id): arrival time}
    departure       : dict  — {(train_id, segment_id): departure time}
    delay           : dict  — {(train_id, segment_id): delay in minutes}
    ordering        : dict  — {(i, j, segment_id): 1 if i before j, 0 otherwise}
    """

    def __init__(self, status, objective, runtime, arrival, departure, delay, ordering):
        self.status    = status
        self.objective = objective
        self.runtime   = runtime
        self.arrival   = arrival
        self.departure = departure
        self.delay     = delay
        self.ordering  = ordering

    def arrival_time(self, train_id, segment_id):
        """Returns the rescheduled arrival time for a given train and segment."""
        return self.arrival.get((train_id, segment_id))

    def departure_time(self, train_id, segment_id):
        """Returns the rescheduled departure time for a given train and segment."""
        return self.departure.get((train_id, segment_id))

    def delay_at(self, train_id, segment_id):
        """Returns the delay for a given train at a given segment. als de trein dat bepaalde segment nog niet heeft gepasseerd,geeft het None terug."""
        return self.delay.get((train_id, segment_id))

    def train_goes_first(self, train_i, train_j, segment_id):
        """
        Returns True if train_i is scheduled before train_j on the given segment.
        """
        return self.ordering.get((train_i, train_j, segment_id))

    def is_feasible(self):
        """Returns True if the solver found a feasible solution (both optimal or not)."""
        return self.status in ("optimal", "timeout")

    def __repr__(self): # toont hoe Solution object eruitziet als je het print
        obj_str = f"{self.objective:.2f}" if self.objective is not None else "None"
        return (
            f"Solution(status={self.status}, "
            f"objective={self.objective:.2f}, "
            f"runtime={self.runtime:.2f}s)"
        )


# =============================================================================
# Parser function
# =============================================================================

def parse_solution(model, a, d, delta, y, C):
    """
    Parses Gurobi model output into a Solution object.

    Parameters
    ----------
    model : gurobipy.Model  — solved Gurobi model
    a     : gurobipy.Vars   — arrival time variables
    d     : gurobipy.Vars   — departure time variables
    delta : gurobipy.Vars   — delay variables
    y     : gurobipy.Vars   — ordering variables
    C     : dict            — conflict sets per segment {segment: [(i,j), ...]}

    Returns
    -------
    Solution object
    """

    # -------------------------------------------------------------------------
    # Determine solver status
    # -------------------------------------------------------------------------
    status_code = model.Status

    if status_code == GRB.OPTIMAL:
        status = "optimal"
    elif status_code == GRB.TIME_LIMIT and model.SolCount > 0:
        status = "timeout"       # timed out but found a feasible solution, otherwise unknown
    elif status_code == GRB.INFEASIBLE:
        status = "infeasible"
    else:
        status = "unknown"

    # -------------------------------------------------------------------------
    # If no feasible solution found, skip extraction and return empty Solution
    # -------------------------------------------------------------------------
    if status in ("infeasible", "unknown"):
        return Solution(
            status=status,
            objective=None,
            runtime=model.Runtime,
            arrival={},
            departure={},
            delay={},
            ordering={}
        )

    # -------------------------------------------------------------------------
    # Extract objective value and runtime, only when there is a feasible solution
    # -------------------------------------------------------------------------
    objective = model.ObjVal
    runtime   = model.Runtime

    # -------------------------------------------------------------------------
    # Extract arrival, departure and delay values
    # -------------------------------------------------------------------------
    arrival   = {key: var.X for key, var in a.items()}
    departure = {key: var.X for key, var in d.items()}
    delay     = {key: var.X for key, var in delta.items()}

    # -------------------------------------------------------------------------
    # Extract ordering decisions
    # -------------------------------------------------------------------------
    ordering = {}
    for s, pairs in C.items():
        for (i, j) in pairs:
            ordering[i, j, s] = round(y[i, j, s].X)  # round to 0 or 1

    # -------------------------------------------------------------------------
    # Return Solution object
    # -------------------------------------------------------------------------
    return Solution(
        status=status,
        objective=objective,
        runtime=runtime,
        arrival=arrival,
        departure=departure,
        delay=delay,
        ordering=ordering
    )