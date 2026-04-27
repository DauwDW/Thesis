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
    delay           : dict  — {(train_id, segment_id): delay in seconds}
    ordering        : dict  — {(i, j, segment_id): 1 if i before j, 0 otherwise}
    """

    def __init__(self, status, objective, runtime, arrival, departure, delay, ordering, priority_upgrade=None, upgrade_contribution=None):
        self.status    = status
        self.objective = objective
        self.runtime   = runtime
        self.arrival   = arrival
        self.departure = departure
        self.delay     = delay
        self.ordering  = ordering
        self.priority_upgrade     = priority_upgrade or {}
        self.upgrade_contribution = upgrade_contribution or {}

    def arrival_time(self, train_id, segment_id): #Returns the rescheduled arrival time for a given train and segment
        return self.arrival.get((train_id, segment_id))

    def departure_time(self, train_id, segment_id): #Returns the rescheduled departure time for a given train and segment
        return self.departure.get((train_id, segment_id))

    def delay_at(self, train_id, segment_id): #Returns the delay for a given train at a given segment. als de trein dat bepaalde segment nog niet heeft gepasseerd,geeft het None terug
        return self.delay.get((train_id, segment_id))

    def train_goes_first(self, train_i, train_j, segment_id): #Returns True if train_i is scheduled before train_j on the given segment
        return self.ordering.get((train_i, train_j, segment_id))
    
    def is_upgraded(self, train_id): #Returns True if the dynamic priority upgrade was triggered for this train
        return bool(self.priority_upgrade.get(train_id, 0))

    def is_feasible(self): #Returns True if the solver found a feasible solution (both optimal or not)
        return self.status in ("optimal", "timeout")

    def __repr__(self):
        obj_str = f"{self.objective:.2f}" if self.objective is not None else "None"
        return (
            f"Solution(status={self.status}, "
            f"objective={obj_str}, "
            f"runtime={self.runtime:.2f}s)")


def parse_solution(model, a, d, delta, y, C, pdl=None, q=None): #Parses Gurobi model output into a Solution object. Returns Solution Object
    
    # Determine solver status
    status_code = model.Status

    if status_code == GRB.OPTIMAL:
        status = "optimal"
    elif status_code == GRB.TIME_LIMIT and model.SolCount > 0:
        status = "timeout"       # timed out but found a feasible solution, otherwise unknown
    elif status_code == GRB.INFEASIBLE:
        status = "infeasible"
    else:
        status = "unknown"

    # If no feasible solution found, skip extraction and return empty Solution
    if status in ("infeasible", "unknown"):
        return Solution(
            status=status,
            objective=None,
            runtime=model.Runtime,
            arrival={},
            departure={},
            delay={},
            ordering={})

    # Extract objective value and runtime, only when there is a feasible solution
    objective = model.ObjVal
    runtime   = model.Runtime

    # Extract arrival, departure and delay values
    arrival   = {key: var.X for key, var in a.items()}
    departure = {key: var.X for key, var in d.items()}
    delay     = {key: var.X for key, var in delta.items()}

    # Extract ordering decisions
    ordering = {}
    for s, pairs in C.items():
        for (i, j) in pairs:
            ordering[i, j, s] = round(y[i, j, s].X)  # round to 0 or 1

    # Extract dynamic priority variables (only set for dynamic model)
    priority_upgrade     = {t: round(pdl[t].X) for t in pdl} if pdl is not None else {}
    upgrade_contribution = {t: q[t].X          for t in q}   if q   is not None else {}


    # Return Solution object
    return Solution(
        status=status,
        objective=objective,
        runtime=runtime,
        arrival=arrival,
        departure=departure,
        delay=delay,
        ordering=ordering,
        priority_upgrade=priority_upgrade,
        upgrade_contribution=upgrade_contribution)