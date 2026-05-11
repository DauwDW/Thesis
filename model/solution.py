"""
solution.py

Parses the Gurobi model output into a clean Solution object.
The Solution object is used by the controller to apply the
rescheduling decisions back to the SystemState.

Variabelenamen consistent met mip_base/mip_dynamic:
  entry  (was: a)
  dep    (was: d)
  delay  (was: delta)

De ordering dict (y-variabelen) is verwijderd: de simulator gebruikt
uitsluitend de entry-tijden uit solution.arrival om volgorde te bepalen.
"""

from gurobipy import GRB


class Solution:
    """
    Stores the result of a single MILP solve.

    Attributes
    ----------
    status               : str          — 'optimal', 'timeout', 'infeasible', 'unknown'
    objective            : float|None   — objective value (total weighted delay)
    runtime              : float        — solver runtime in seconds
    arrival              : dict         — {(train_id, segment_id): entry time}
    departure            : dict         — {(train_id, segment_id): departure time}
    delay                : dict         — {(train_id, segment_id): delay in seconds}
    priority_upgrade     : dict         — {train_id: 0|1}  (dynamic model only)
    upgrade_contribution : dict         — {train_id: float} (dynamic model only)
    """

    def __init__(
        self,
        status,
        objective,
        runtime,
        arrival,
        departure,
        delay,
        priority_upgrade=None,
        upgrade_contribution=None,
    ):
        self.status               = status
        self.objective            = objective
        self.runtime              = runtime
        self.arrival              = arrival
        self.departure            = departure
        self.delay                = delay
        self.priority_upgrade     = priority_upgrade     or {}
        self.upgrade_contribution = upgrade_contribution or {}

    def is_feasible(self) -> bool:
        """True als de solver een haalbare oplossing heeft gevonden."""
        return self.status in ("optimal", "timeout")

    def arrival_time(self, train_id, segment_id) -> float | None:
        return self.arrival.get((train_id, segment_id))

    def departure_time(self, train_id, segment_id) -> float | None:
        return self.departure.get((train_id, segment_id))

    def delay_at(self, train_id, segment_id) -> float | None:
        return self.delay.get((train_id, segment_id))

    def is_upgraded(self, train_id) -> bool:
        """True als de dynamische prioriteitsupgrade getriggerd werd voor train_id."""
        return bool(self.priority_upgrade.get(train_id, 0))

    def __repr__(self) -> str:
        obj_str = f"{self.objective:.2f}" if self.objective is not None else "None"
        return (
            f"Solution(status={self.status}, "
            f"objective={obj_str}, "
            f"runtime={self.runtime:.2f}s)"
        )


def parse_solution(model, entry, dep, delay, y, C, pdl=None, q=None) -> Solution:
    """
    Parses Gurobi model output into a Solution object.

    Parameters
    ----------
    model : gurobipy.Model
    entry : dict {(train_id, seg_id): Var}  — entry-tijden
    dep   : dict {(train_id, seg_id): Var}  — departure-tijden
    delay : dict {(train_id, seg_id): Var}  — vertragingen
    y     : gurobipy.Vars                   — binaire volgordevariabelen (ongebruikt)
    C     : dict {seg_id: [(i,j)]}          — conflictparen (ongebruikt)
    pdl   : dict | None                     — dynamic priority upgrade variabelen
    q     : dict | None                     — linearisatie variabelen
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
            arrival={},
            departure={},
            delay={},
        )

    # --- Haalbare oplossing: extraheer variabelwaarden ---
    return Solution(
        status    = status,
        objective = model.ObjVal,
        runtime   = model.Runtime,
        arrival   = {key: var.X for key, var in entry.items()},
        departure = {key: var.X for key, var in dep.items()},
        delay     = {key: var.X for key, var in delay.items()},
        priority_upgrade     = {t: round(pdl[t].X) for t in pdl} if pdl is not None else {},
        upgrade_contribution = {t: q[t].X          for t in q}   if q   is not None else {},
    )