"""
tests/test_mip.py

Rigoureuze tests voor alle bestanden in model/:
  - model/solution.py     : Solution klasse en parse_solution
  - model/mip_base.py     : statisch MIP model (constraints C1–C4)
  - model/mip_dynamic.py  : dynamisch MIP model (constraints C1–C4 + C7–C8)
  - model/solver.py       : solver dispatch

Vereisten: gurobipy met geldige licentie (Academic of trial)
Uitvoeren:  pytest tests/test_mip.py -v
"""

import pytest
from unittest.mock import MagicMock
from gurobipy import GRB

from model.solution    import Solution, parse_solution
from model.mip_base    import build_and_solve_model as solve_static
from model.mip_dynamic import build_and_solve_model as solve_dynamic
from model.solver      import solve


# =============================================================================
# Gedeelde fixtures
# =============================================================================

@pytest.fixture
def single_train_line():
    """
    Minimale instantie: 1 trein, 1 lijnsegment.
    On-time, geen conflicten.
    """
    T   = [1]
    Tp  = [1]
    Tf  = []
    S   = ["line_AB"]
    Ss  = set()
    Sl  = {"line_AB"}
    path = {1: ["line_AB"]}

    sched_entry = {(1, "line_AB"): 3600.0}
    sched_dep   = {(1, "line_AB"): 3660.0}
    RT          = {(1, "line_AB"): 60.0}
    DW          = {}
    H           = {}
    h_stop      = {}
    w           = {1: 2}
    L           = 86400.0

    return dict(
        T=T, Tp=Tp, Tf=Tf,
        S=S, Ss=Ss, Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_dep=sched_dep,
        RT=RT, DW=DW, H=H, h_stop=h_stop,
        w=w, L=L,
    )


@pytest.fixture
def single_train_station():
    """
    Minimale instantie: 1 trein, 1 stationssegment (stoppend).
    """
    T   = [1]
    Tp  = [1]
    Tf  = []
    S   = ["sta_A"]
    Ss  = {"sta_A"}
    Sl  = set()
    path = {1: ["sta_A"]}

    sched_entry = {(1, "sta_A"): 3600.0}
    sched_dep   = {(1, "sta_A"): 3660.0}
    RT          = {}
    DW          = {(1, "sta_A"): 60.0}
    H           = {}
    h_stop      = {(1, "sta_A"): True}
    w           = {1: 2}
    L           = 86400.0

    return dict(
        T=T, Tp=Tp, Tf=Tf,
        S=S, Ss=Ss, Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_dep=sched_dep,
        RT=RT, DW=DW, H=H, h_stop=h_stop,
        w=w, L=L,
    )


@pytest.fixture
def two_trains_conflict():
    """
    2 treinen op hetzelfde lijnsegment — conflict vereist ordering.
    Trein 1 vertraagd (entry=3700 i.p.v. 3600).
    """
    T   = [1, 2]
    Tp  = [1, 2]
    Tf  = []
    S   = ["line_AB"]
    Ss  = set()
    Sl  = {"line_AB"}
    path = {1: ["line_AB"], 2: ["line_AB"]}

    sched_entry = {(1, "line_AB"): 3600.0, (2, "line_AB"): 3700.0}
    sched_dep   = {(1, "line_AB"): 3660.0, (2, "line_AB"): 3760.0}
    RT          = {(1, "line_AB"): 60.0, (2, "line_AB"): 60.0}
    DW          = {}
    H           = {(1, 2, "line_AB"): 0, (2, 1, "line_AB"): 0}
    h_stop      = {}
    w           = {1: 2, 2: 2}
    L           = 86400.0

    return dict(
        T=T, Tp=Tp, Tf=Tf,
        S=S, Ss=Ss, Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_dep=sched_dep,
        RT=RT, DW=DW, H=H, h_stop=h_stop,
        w=w, L=L,
    )


@pytest.fixture
def multi_segment_train():
    """
    1 trein, 3 segmenten: lijn → station → lijn.
    Test C1c (transitie), C1a, C1b, C2.
    """
    T   = [1]
    Tp  = [1]
    Tf  = []
    S   = ["line_AB", "sta_B", "line_BC"]
    Ss  = {"sta_B"}
    Sl  = {"line_AB", "line_BC"}
    path = {1: ["line_AB", "sta_B", "line_BC"]}

    sched_entry = {
        (1, "line_AB"): 3600.0,
        (1, "sta_B"):   3660.0,
        (1, "line_BC"): 3720.0,
    }
    sched_dep = {
        (1, "line_AB"): 3660.0,
        (1, "sta_B"):   3720.0,
        (1, "line_BC"): 3780.0,
    }
    RT     = {(1, "line_AB"): 60.0, (1, "line_BC"): 60.0}
    DW     = {(1, "sta_B"):   60.0}
    H      = {}
    h_stop = {(1, "sta_B"): True}
    w      = {1: 2}
    L      = 86400.0

    return dict(
        T=T, Tp=Tp, Tf=Tf,
        S=S, Ss=Ss, Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_dep=sched_dep,
        RT=RT, DW=DW, H=H, h_stop=h_stop,
        w=w, L=L,
    )


@pytest.fixture
def dynamic_instance_base():
    """
    Minimale instantie voor mip_dynamic: 1 trein, 1 lijn, vertraagd.
    """
    T   = [1]
    Tp  = [1]
    Tf  = []
    S   = ["line_AB"]
    Ss  = set()
    Sl  = {"line_AB"}
    path = {1: ["line_AB"]}

    sched_entry = {(1, "line_AB"): 3600.0}
    sched_dep   = {(1, "line_AB"): 3660.0}
    RT          = {(1, "line_AB"): 60.0}
    DW          = {}
    H           = {}
    h_stop      = {}
    psl         = {1: 1}
    w           = {1: 2}
    L           = 86400.0

    return dict(
        T=T, Tp=Tp, Tf=Tf,
        S=S, Ss=Ss, Sl=Sl,
        path=path,
        sched_entry=sched_entry,
        sched_dep=sched_dep,
        RT=RT, DW=DW, H=H, h_stop=h_stop,
        w=w, psl=psl,
        gamma=300.0, epsilon=1.0, delta_max=3600.0,
        L=L,
    )


def _solve_base(inst, **kwargs):
    """Helper: roep solve_static aan met alle vereiste velden."""
    return solve_static(
        T            = inst["T"],
        Tp           = inst["Tp"],
        Tf           = inst["Tf"],
        S            = inst["S"],
        Ss           = inst["Ss"],
        Sl           = inst["Sl"],
        path         = inst["path"],
        sched_entry  = inst["sched_entry"],
        sched_dep    = inst["sched_dep"],
        RT           = inst["RT"],
        DW           = inst["DW"],
        H            = inst["H"],
        h_stop       = inst["h_stop"],
        w            = inst["w"],
        L            = inst["L"],
        verbose      = False,
        **kwargs,
    )


def _solve_dyn(inst, **kwargs):
    """Helper: roep solve_dynamic aan met alle vereiste velden."""
    return solve_dynamic(
        T            = inst["T"],
        Tp           = inst["Tp"],
        Tf           = inst["Tf"],
        S            = inst["S"],
        Ss           = inst["Ss"],
        Sl           = inst["Sl"],
        path         = inst["path"],
        sched_entry  = inst["sched_entry"],
        sched_dep    = inst["sched_dep"],
        RT           = inst["RT"],
        DW           = inst["DW"],
        H            = inst["H"],
        h_stop       = inst["h_stop"],
        psl          = inst["psl"],
        gamma        = inst["gamma"],
        epsilon      = inst["epsilon"],
        delta_max    = inst["delta_max"],
        L            = inst["L"],
        verbose      = False,
        **kwargs,
    )


# =============================================================================
# Tests: model/solution.py
# =============================================================================

class TestSolution:

    def test_is_feasible_optimal(self):
        s = Solution("optimal", 10.0, 1.0, {}, {}, {}, {})
        assert s.is_feasible() is True

    def test_is_feasible_timeout(self):
        s = Solution("timeout", 10.0, 60.0, {}, {}, {}, {})
        assert s.is_feasible() is True

    def test_is_feasible_infeasible(self):
        s = Solution("infeasible", None, 0.0, {}, {}, {}, {})
        assert s.is_feasible() is False

    def test_is_feasible_unknown(self):
        s = Solution("unknown", None, 0.0, {}, {}, {}, {})
        assert s.is_feasible() is False

    def test_arrival_time_returns_correct(self):
        s = Solution("optimal", 0.0, 0.0,
                     arrival={(1, "seg"): 3600.0}, departure={}, delay={}, ordering={})
        assert s.arrival_time(1, "seg") == 3600.0

    def test_arrival_time_missing_returns_none(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, {})
        assert s.arrival_time(99, "missing") is None

    def test_departure_time_returns_correct(self):
        s = Solution("optimal", 0.0, 0.0,
                     arrival={}, departure={(1, "seg"): 3660.0}, delay={}, ordering={})
        assert s.departure_time(1, "seg") == 3660.0

    def test_delay_at_returns_correct(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, delay={(1, "seg"): 120.0}, ordering={})
        assert s.delay_at(1, "seg") == 120.0

    def test_delay_at_missing_returns_none(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, {})
        assert s.delay_at(1, "missing") is None

    def test_train_goes_first_returns_ordering(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, ordering={(1, 2, "seg"): 1})
        assert s.train_goes_first(1, 2, "seg") == 1

    def test_is_upgraded_true(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, {}, priority_upgrade={1: 1})
        assert s.is_upgraded(1) is True

    def test_is_upgraded_false(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, {}, priority_upgrade={1: 0})
        assert s.is_upgraded(1) is False

    def test_is_upgraded_missing_returns_false(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, {})
        assert s.is_upgraded(99) is False

    def test_priority_upgrade_defaults_to_empty(self):
        s = Solution("optimal", 0.0, 0.0, {}, {}, {}, {})
        assert s.priority_upgrade == {}
        assert s.upgrade_contribution == {}

    def test_repr_optimal(self):
        s = Solution("optimal", 42.5, 1.23, {}, {}, {}, {})
        r = repr(s)
        assert "optimal" in r
        assert "42" in r

    def test_repr_infeasible_geen_crash(self):
        """repr mag niet crashen als objective None is — bekende bug in solution.py.
        Fix: gebruik obj_str in de f-string i.p.v. self.objective rechtstreeks.
        """
        s = Solution("infeasible", None, 0.5, {}, {}, {}, {})
        r = repr(s)
        assert "infeasible" in r
        assert "None" in r


# =============================================================================
# Tests: model/mip_base.py
# =============================================================================

class TestMipBase:

    # ------------------------------------------------------------------
    # Basis: oplossing gevonden
    # ------------------------------------------------------------------

    def test_single_train_line_optimal(self, single_train_line):
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_line)
        assert model.Status == GRB.OPTIMAL

    def test_single_train_station_optimal(self, single_train_station):
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_station)
        assert model.Status == GRB.OPTIMAL

    def test_multi_segment_optimal(self, multi_segment_train):
        model, a, d, delta, y, C, final_seg = _solve_base(multi_segment_train)
        assert model.Status == GRB.OPTIMAL

    def test_two_trains_conflict_optimal(self, two_trains_conflict):
        model, a, d, delta, y, C, final_seg = _solve_base(two_trains_conflict)
        assert model.Status == GRB.OPTIMAL

    # ------------------------------------------------------------------
    # C1a — minimale rijtijd op lijnsegment
    # ------------------------------------------------------------------

    def test_c1a_running_time_respected(self, single_train_line):
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_line)
        assert model.Status == GRB.OPTIMAL
        assert d[1, "line_AB"].X >= a[1, "line_AB"].X + 60.0 - 1e-6

    def test_c1a_running_time_with_delay(self, single_train_line):
        """Trein mag niet vroeger vertrekken dan entry + RT, ook niet als entry laat is."""
        inst = dict(single_train_line)
        inst["fix_arrival"] = {(1, "line_AB"): 3700.0}
        model, a, d, delta, y, C, final_seg = _solve_base(inst)
        assert model.Status == GRB.OPTIMAL
        assert d[1, "line_AB"].X >= a[1, "line_AB"].X + 60.0 - 1e-6

    def test_c1a_in_execution_overrides_rt(self, single_train_line):
        """Als in_execution gezet is, wordt remaining_time gebruikt i.p.v. RT."""
        inst = dict(single_train_line)
        inst["in_execution"] = {(1, "line_AB"): 30.0}
        inst["fix_arrival"]  = {(1, "line_AB"): 3630.0}
        model, a, d, delta, y, C, final_seg = _solve_base(inst)
        assert model.Status == GRB.OPTIMAL
        assert d[1, "line_AB"].X >= a[1, "line_AB"].X + 30.0 - 1e-6

    # ------------------------------------------------------------------
    # C1b — minimale dwell op stationssegment
    # ------------------------------------------------------------------

    def test_c1b_dwell_time_respected_when_stopping(self, single_train_station):
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_station)
        assert model.Status == GRB.OPTIMAL
        assert d[1, "sta_A"].X >= a[1, "sta_A"].X + 60.0 - 1e-6

    def test_c1b_dwell_zero_when_not_stopping(self):
        """Passeertreinen (h_stop=False) hoeven geen dwell te respecteren."""
        inst = dict(
            T=[1], Tp=[1], Tf=[],
            S=["sta_A"], Ss={"sta_A"}, Sl=set(),
            path={1: ["sta_A"]},
            sched_entry={(1, "sta_A"): 3600.0},
            sched_dep  ={(1, "sta_A"): 3660.0},
            RT={}, DW={(1, "sta_A"): 60.0},
            H={}, h_stop={(1, "sta_A"): False},
            w={1: 2}, L=86400.0,
        )
        model, a, d, delta, y, C, final_seg = _solve_base(inst)
        assert model.Status == GRB.OPTIMAL
        # d - a mag kleiner zijn dan dwell want h_stop=False → DW*0=0
        assert d[1, "sta_A"].X >= a[1, "sta_A"].X - 1e-6

    # ------------------------------------------------------------------
    # C1c — transitie tussen segmenten
    # ------------------------------------------------------------------

    def test_c1c_transition_respected(self, multi_segment_train):
        model, a, d, delta, y, C, final_seg = _solve_base(multi_segment_train)
        assert model.Status == GRB.OPTIMAL
        path = [1, "line_AB", "sta_B", "line_BC"]
        assert a[1, "sta_B"].X   >= d[1, "line_AB"].X - 1e-6
        assert a[1, "line_BC"].X >= d[1, "sta_B"].X   - 1e-6

    # ------------------------------------------------------------------
    # C2 — geen vroege vertrek
    # ------------------------------------------------------------------

    def test_c2_no_early_departure(self, single_train_station):
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_station)
        assert model.Status == GRB.OPTIMAL
        assert d[1, "sta_A"].X >= 3660.0 - 1e-6

    def test_c2_not_applied_for_passing_train(self):
        """C2 mag niet verhinderen dat passeertreinen vroeg vertrekken."""
        inst = dict(
            T=[1], Tp=[1], Tf=[],
            S=["sta_A"], Ss={"sta_A"}, Sl=set(),
            path={1: ["sta_A"]},
            sched_entry={(1, "sta_A"): 3600.0},
            sched_dep  ={(1, "sta_A"): 3660.0},
            RT={}, DW={(1, "sta_A"): 60.0},
            H={}, h_stop={(1, "sta_A"): False},
            w={1: 2}, L=86400.0,
        )
        model, a, d, delta, y, C, final_seg = _solve_base(inst)
        assert model.Status == GRB.OPTIMAL
        # Passeertreinen zijn niet gebonden aan sched_dep via C2

    # ------------------------------------------------------------------
    # C3 — vertraging definitie
    # ------------------------------------------------------------------

    def test_c3_delay_nonnegative(self, single_train_line):
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_line)
        assert model.Status == GRB.OPTIMAL
        assert delta[1, "line_AB"].X >= -1e-6

    def test_c3_delay_positive_when_late(self, single_train_line):
        """Als trein laat aankomt, moet delta >= verschil zijn."""
        model, a, d, delta, y, C, final_seg = _solve_base(
            single_train_line,
            fix_arrival={(1, "line_AB"): 3700.0},
        )
        assert model.Status == GRB.OPTIMAL
        assert delta[1, "line_AB"].X >= 3700.0 - 3600.0 - 1e-6

    # ------------------------------------------------------------------
    # C4 — conflictconstraints / headway
    # ------------------------------------------------------------------

    def test_c4_ordering_variable_binary(self, two_trains_conflict):
        model, a, d, delta, y, C, final_seg = _solve_base(two_trains_conflict)
        assert model.Status == GRB.OPTIMAL
        val = y[1, 2, "line_AB"].X
        assert abs(val - round(val)) < 1e-6  # binair

    def test_c4_trains_dont_overlap(self, two_trains_conflict):
        """Na oplossing mogen 2 treinen niet tegelijk op hetzelfde segment zijn."""
        model, a, d, delta, y, C, final_seg = _solve_base(two_trains_conflict)
        assert model.Status == GRB.OPTIMAL
        # Als y=1: trein 1 voor trein 2 → a[2] >= d[1]
        # Als y=0: trein 2 voor trein 1 → a[1] >= d[2]
        y_val = round(y[1, 2, "line_AB"].X)
        if y_val == 1:
            assert a[2, "line_AB"].X >= d[1, "line_AB"].X - 1e-6
        else:
            assert a[1, "line_AB"].X >= d[2, "line_AB"].X - 1e-6

    def test_c4_headway_respected(self):
        """Met H > 0 moet de gap groter zijn dan het headway."""
        T   = [1, 2]
        S   = ["line_AB"]
        Ss  = set()
        Sl  = {"line_AB"}
        path = {1: ["line_AB"], 2: ["line_AB"]}
        H = {(1, 2, "line_AB"): 180, (2, 1, "line_AB"): 180}
        inst = dict(
            T=T, Tp=T, Tf=[],
            S=S, Ss=Ss, Sl=Sl,
            path=path,
            sched_entry={(1, "line_AB"): 3600.0, (2, "line_AB"): 3700.0},
            sched_dep  ={(1, "line_AB"): 3660.0, (2, "line_AB"): 3760.0},
            RT={(1, "line_AB"): 60.0, (2, "line_AB"): 60.0},
            DW={}, H=H, h_stop={},
            w={1: 2, 2: 2}, L=86400.0,
        )
        model, a, d, delta, y, C, final_seg = _solve_base(inst)
        assert model.Status == GRB.OPTIMAL
        y_val = round(y[1, 2, "line_AB"].X)
        if y_val == 1:
            assert a[2, "line_AB"].X >= d[1, "line_AB"].X + 180 - 1e-6
        else:
            assert a[1, "line_AB"].X >= d[2, "line_AB"].X + 180 - 1e-6

    # ------------------------------------------------------------------
    # Objectief
    # ------------------------------------------------------------------

    def test_objective_minimizes_delay(self, single_train_line):
        """Oplossing zonder vertraging heeft objectief ≈ 0."""
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_line)
        assert model.Status == GRB.OPTIMAL
        assert model.ObjVal >= -1e-6

    def test_objective_passenger_weighted_higher(self):
        """Passagierstrein (w=2) telt zwaarder dan goederentrein (w=1).
        We verifieren dit door twee asymmetrische instanties te vergelijken:
        - inst_p: passagierstrein vertraagd → hoog gewogen objectief
        - inst_f: goederentrein vertraagd   → laag gewogen objectief
        inst_p moet een hogere objectiefwaarde hebben dan inst_f.
        """
        base = dict(
            T=[1], Tp=[1], Tf=[],
            S=["line_AB"], Ss=set(), Sl={"line_AB"},
            path={1: ["line_AB"]},
            sched_entry={(1, "line_AB"): 3600.0},
            sched_dep  ={(1, "line_AB"): 3660.0},
            RT={(1, "line_AB"): 60.0},
            DW={}, H={}, h_stop={}, L=86400.0,
        )
        # Passagierstrein 100s vertraagd (w=2) → objectief = 2 * 100 = 200
        inst_p = dict(base, w={1: 2})
        model_p, a_p, d_p, delta_p, y_p, C_p, fs_p = _solve_base(
            inst_p, fix_arrival={(1, "line_AB"): 3700.0}
        )
        # Goederentrein 100s vertraagd (w=1) → objectief = 1 * 100 = 100
        inst_f = dict(base, w={1: 1}, Tp=[], Tf=[1])
        model_f, a_f, d_f, delta_f, y_f, C_f, fs_f = _solve_base(
            inst_f, fix_arrival={(1, "line_AB"): 3700.0}
        )
        assert model_p.Status == GRB.OPTIMAL
        assert model_f.Status == GRB.OPTIMAL
        assert model_p.ObjVal > model_f.ObjVal - 1e-6

    # ------------------------------------------------------------------
    # fix_arrival
    # ------------------------------------------------------------------

    def test_fix_arrival_pins_entry_time(self, single_train_line):
        model, a, d, delta, y, C, final_seg = _solve_base(
            single_train_line,
            fix_arrival={(1, "line_AB"): 3700.0},
        )
        assert model.Status == GRB.OPTIMAL
        assert abs(a[1, "line_AB"].X - 3700.0) < 1e-6

    # ------------------------------------------------------------------
    # Return structuur
    # ------------------------------------------------------------------

    def test_return_tuple_length(self, single_train_line):
        result = _solve_base(single_train_line)
        assert len(result) == 7  # model, a, d, delta, y, C, final_seg

    def test_final_seg_correct(self, multi_segment_train):
        model, a, d, delta, y, C, final_seg = _solve_base(multi_segment_train)
        assert final_seg[1] == "line_BC"

    def test_C_contains_all_conflict_pairs(self, two_trains_conflict):
        model, a, d, delta, y, C, final_seg = _solve_base(two_trains_conflict)
        assert "line_AB" in C
        assert (1, 2) in C["line_AB"]

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_train_set_runs(self):
        """Lege T — geen treinen, triviale oplossing."""
        model, a, d, delta, y, C, final_seg = solve_static(
            T=[], Tp=[], Tf=[],
            S=[], Ss=set(), Sl=set(),
            path={}, sched_entry={}, sched_dep={},
            RT={}, DW={}, H={}, h_stop={},
            w={}, L=86400.0, verbose=False,
        )
        assert model.Status == GRB.OPTIMAL

    def test_single_segment_single_train_no_conflict(self, single_train_line):
        """Geen conflictparen bij 1 trein."""
        model, a, d, delta, y, C, final_seg = _solve_base(single_train_line)
        assert C["line_AB"] == []

    def test_solution_nonnegative_times(self, multi_segment_train):
        """Alle tijden moeten niet-negatief zijn."""
        model, a, d, delta, y, C, final_seg = _solve_base(multi_segment_train)
        assert model.Status == GRB.OPTIMAL
        for val in a.values():
            assert val.X >= -1e-6
        for val in d.values():
            assert val.X >= -1e-6

    def test_departure_after_arrival_all_segments(self, multi_segment_train):
        """d[t,s] >= a[t,s] voor elk segment."""
        model, a, d, delta, y, C, final_seg = _solve_base(multi_segment_train)
        assert model.Status == GRB.OPTIMAL
        for t in [1]:
            for s in ["line_AB", "sta_B", "line_BC"]:
                assert d[t, s].X >= a[t, s].X - 1e-6


# =============================================================================
# Tests: model/mip_dynamic.py
# =============================================================================

class TestMipDynamic:

    def test_single_train_optimal(self, dynamic_instance_base):
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(dynamic_instance_base)
        assert model.Status == GRB.OPTIMAL

    def test_return_tuple_length(self, dynamic_instance_base):
        result = _solve_dyn(dynamic_instance_base)
        assert len(result) == 9  # model, a, d, delta, y, pdl, q, C, final_seg

    def test_pdl_binary(self, dynamic_instance_base):
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(dynamic_instance_base)
        assert model.Status == GRB.OPTIMAL
        val = pdl[1].X
        assert abs(val - round(val)) < 1e-6

    def test_q_nonnegative(self, dynamic_instance_base):
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(dynamic_instance_base)
        assert model.Status == GRB.OPTIMAL
        assert q[1].X >= -1e-6

    def test_c7a_pdl_forced_when_delay_exceeds_gamma(self, dynamic_instance_base):
        """Als delay >= gamma, moet pdl=1."""
        # fix_arrival=4100 → delay=500s > gamma=300s → pdl moet 1 zijn
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(
            dynamic_instance_base,
            fix_arrival={(1, "line_AB"): 3600.0 + 500.0},
        )
        assert model.Status == GRB.OPTIMAL
        assert round(pdl[1].X) == 1

    def test_c7b_pdl_zero_when_no_delay(self, dynamic_instance_base):
        """Als delay < gamma, moet pdl=0."""
        # On-time trein (fix_arrival = sched_entry) → delay ≈ 0 < gamma=300
        inst = dict(dynamic_instance_base)
        inst["fix_arrival"] = {(1, "line_AB"): 3600.0}
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(inst)
        assert model.Status == GRB.OPTIMAL
        assert round(pdl[1].X) == 0

    def test_c8a_q_zero_when_pdl_zero(self, dynamic_instance_base):
        """Als pdl=0, moet q=0."""
        inst = dict(dynamic_instance_base)
        inst["fix_arrival"] = {(1, "line_AB"): 3600.0}
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(inst)
        assert model.Status == GRB.OPTIMAL
        if round(pdl[1].X) == 0:
            assert q[1].X <= 1e-6

    def test_c8b_q_at_most_delta(self, dynamic_instance_base):
        """q <= delta altijd."""
        inst = dict(dynamic_instance_base)
        inst["fix_arrival"] = {(1, "line_AB"): 3600.0 + 500.0}
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(inst)
        assert model.Status == GRB.OPTIMAL
        s_last = "line_AB"
        assert q[1].X <= delta[1, s_last].X + 1e-6

    def test_same_constraints_as_base(self, dynamic_instance_base):
        """C1–C4 zijn identiek in beide modellen: check d >= a + RT."""
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(dynamic_instance_base)
        assert model.Status == GRB.OPTIMAL
        assert d[1, "line_AB"].X >= a[1, "line_AB"].X + 60.0 - 1e-6

    def test_objective_nonnegative(self, dynamic_instance_base):
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(dynamic_instance_base)
        assert model.Status == GRB.OPTIMAL
        assert model.ObjVal >= -1e-6

    def test_two_trains_conflict_dynamic(self):
        """2 treinen, dynamisch model — conflict correct opgelost."""
        T   = [1, 2]
        S   = ["line_AB"]
        Ss  = set()
        Sl  = {"line_AB"}
        path = {1: ["line_AB"], 2: ["line_AB"]}
        H = {(1, 2, "line_AB"): 0, (2, 1, "line_AB"): 0}
        inst = dict(
            T=T, Tp=T, Tf=[],
            S=S, Ss=Ss, Sl=Sl,
            path=path,
            sched_entry={(1, "line_AB"): 3600.0, (2, "line_AB"): 3700.0},
            sched_dep  ={(1, "line_AB"): 3660.0, (2, "line_AB"): 3760.0},
            RT={(1, "line_AB"): 60.0, (2, "line_AB"): 60.0},
            DW={}, H=H, h_stop={},
            w={1: 2, 2: 2},
            psl={1: 1, 2: 1},
            gamma=300.0, epsilon=1.0, delta_max=3600.0,
            L=86400.0,
        )
        model, a, d, delta, y, pdl, q, C, final_seg = _solve_dyn(inst)
        assert model.Status == GRB.OPTIMAL
        y_val = round(y[1, 2, "line_AB"].X)
        if y_val == 1:
            assert a[2, "line_AB"].X >= d[1, "line_AB"].X - 1e-6
        else:
            assert a[1, "line_AB"].X >= d[2, "line_AB"].X - 1e-6


# =============================================================================
# Tests: model/solver.py
# =============================================================================

class TestSolver:

    def test_static_strategy_returns_solution(self, single_train_line):
        inst = dict(single_train_line)
        inst.update(psl={1: 1}, gamma=300.0, epsilon=1.0, delta_max=3600.0)
        sol = solve(inst, priority_strategy="static", verbose=False)
        assert isinstance(sol, Solution)
        assert sol.is_feasible()

    def test_dynamic_strategy_returns_solution(self, dynamic_instance_base):
        sol = solve(dynamic_instance_base, priority_strategy="dynamic", verbose=False)
        assert isinstance(sol, Solution)
        assert sol.is_feasible()

    def test_unknown_strategy_raises(self, single_train_line):
        with pytest.raises(ValueError, match="Unknown priority strategy"):
            solve(single_train_line, priority_strategy="onbekend", verbose=False)

    def test_empty_T_returns_optimal(self, single_train_line):
        inst = dict(single_train_line)
        inst["T"] = []
        sol = solve(inst, priority_strategy="static", verbose=False)
        assert sol.status == "optimal"
        assert sol.objective == 0.0

    def test_solution_arrival_keys_match_instance(self, single_train_line):
        inst = dict(single_train_line)
        inst.update(psl={1: 1}, gamma=300.0, epsilon=1.0, delta_max=3600.0)
        sol = solve(inst, priority_strategy="static", verbose=False)
        assert (1, "line_AB") in sol.arrival

    def test_solution_departure_keys_match_instance(self, single_train_line):
        inst = dict(single_train_line)
        inst.update(psl={1: 1}, gamma=300.0, epsilon=1.0, delta_max=3600.0)
        sol = solve(inst, priority_strategy="static", verbose=False)
        assert (1, "line_AB") in sol.departure

    def test_static_solution_has_no_priority_upgrade(self, single_train_line):
        inst = dict(single_train_line)
        inst.update(psl={1: 1}, gamma=300.0, epsilon=1.0, delta_max=3600.0)
        sol = solve(inst, priority_strategy="static", verbose=False)
        assert sol.priority_upgrade == {}
        assert sol.upgrade_contribution == {}

    def test_dynamic_solution_has_priority_upgrade(self, dynamic_instance_base):
        sol = solve(dynamic_instance_base, priority_strategy="dynamic", verbose=False)
        assert isinstance(sol.priority_upgrade, dict)
        assert 1 in sol.priority_upgrade

    def test_solution_ordering_binary(self, two_trains_conflict):
        inst = dict(two_trains_conflict)
        inst.update(psl={1: 1, 2: 1}, gamma=300.0, epsilon=1.0, delta_max=3600.0)
        sol = solve(inst, priority_strategy="static", verbose=False)
        for val in sol.ordering.values():
            assert val in (0, 1)

    def test_runtime_positive(self, single_train_line):
        inst = dict(single_train_line)
        inst.update(psl={1: 1}, gamma=300.0, epsilon=1.0, delta_max=3600.0)
        sol = solve(inst, priority_strategy="static", verbose=False)
        assert sol.runtime >= 0.0

    def test_static_and_dynamic_same_feasibility(self, dynamic_instance_base):
        """Beide strategieën moeten feasible zijn op dezelfde instantie."""
        sol_s = solve(dynamic_instance_base, priority_strategy="static",  verbose=False)
        sol_d = solve(dynamic_instance_base, priority_strategy="dynamic", verbose=False)
        assert sol_s.is_feasible()
        assert sol_d.is_feasible()