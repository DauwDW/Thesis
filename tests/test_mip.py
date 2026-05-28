"""
tests/test_mip.py

Stress-tests voor de geünificeerde resource-choice formulering (mip_model.py).

Twee test-netwerken
-------------------

1. FULL PATH  A → AB → B  (single-train basisconstraints)
   Ss = {"A","B"}, Sl = {"AB"}, pad = ["A","AB","B"]
   Tijdlijn per trein bij on-time rijden:
     A  : [a_entry,  a_entry + 30]
     AB : [a_entry + 30, a_entry + 90]
     B  : [a_entry + 90, a_entry + 120]

2. LINE ONLY  AB  (multi-train conflict- en retrackingtests)
   Ss = {}, Sl = {"AB"}, pad = ["AB"]
   Eén segment per trein; step 2 genereert geen nevenconflicten op
   station-segmenten → schone isolatie van AB-conflicten.
   Tijdlijn: AB = [ab_entry, ab_entry + 60]

Retracking wordt gesimuleerd via:
   platform_alternatives = {(t, "AB"): ["AB_alt"]}
en geldt uitsluitend voor het tussen-station segment "AB" (∈ Sl).

LETOP OVER GUROBI GC
---------------------
Gurobi-variabelen verliezen hun .X-attribuut zodra de Gurobi Model-instantie
wordt vrijgegeven door Python's GC. Gebruik altijd de volledige uitpak-volgorde
   model, entry, dep, delay, use, fs = solve_ok(inst)
zodat `model` als naam gebonden blijft voor de duur van de test-functie.

Testgroepen
-----------
1.  ReturnStructure    — return-tuple lengte en typen
2.  BasicConstraints   — C1, C2, C2b, C3, C5  (single train, full path)
3.  OccupiedSegments   — in-uitvoering segmenten
4.  UseVariables       — C6a resource-keuze
5.  ConflictResolution — C6b headway, objectief-gewichten  (line-only)
6.  Retracking         — switchen, penalty, RETRACK_CONFLICT_WINDOW  (line-only)
7.  FixedEntry         — gefixeerde aankomsttijden
8.  WarmStart          — warm-start convergentie
9.  Stress             — 5-trein scenario's, lege instantie, determinisme
"""

import pytest
import model.mip_model as mm
from model.mip_model import build_and_solve_model

# ---------------------------------------------------------------------------
# Netwerk-constanten
# ---------------------------------------------------------------------------
L              = 86400
DWELL_A        = 30       # verblijftijd station A (s)
DWELL_B        = 30       # verblijftijd station B (s)
RUNTIME_AB     = 60       # rijtijd op spoor AB (s)
TRAVEL_TOTAL   = DWELL_A + RUNTIME_AB + DWELL_B   # 120 s  (full path)

Ss_full = frozenset({"A", "B"})
Sl_ab   = frozenset({"AB"})
S_full  = Ss_full | Sl_ab


# ---------------------------------------------------------------------------
# Helper A: single-train full path  A → AB → B
# ---------------------------------------------------------------------------

def make_full(
    a_entry          = 0,
    b_exit           = None,
    current_time     = 0,
    halts            = None,
    fixed_entry      = None,
    occupied         = None,
    platform_alternatives = None,
    weights          = None,
):
    """
    Bouwt een instantie voor één trein op het pad A → AB → B.
    b_exit defaults to a_entry + TRAVEL_TOTAL (on-time).
    """
    if b_exit is None:
        b_exit = a_entry + TRAVEL_TOTAL

    a_exit  = a_entry + DWELL_A
    ab_exit = a_exit  + RUNTIME_AB
    b_entry = ab_exit

    t = 1
    return dict(
        T                     = [t],
        S                     = S_full,
        Ss                    = Ss_full,
        Sl                    = Sl_ab,
        path                  = {t: ["A", "AB", "B"]},
        sched_entry           = {(t,"A"): a_entry, (t,"AB"): a_exit, (t,"B"): b_entry},
        sched_exit            = {(t,"A"): a_exit,  (t,"AB"): ab_exit, (t,"B"): b_exit},
        runtime               = {(t,"AB"): RUNTIME_AB},
        dwell                 = {(t,"A"): DWELL_A, (t,"B"): DWELL_B},
        conflicts             = {},
        occupied              = occupied or {},
        fixed_entry           = fixed_entry or {},
        expected_exit         = {(t,"A"): a_exit, (t,"AB"): ab_exit, (t,"B"): b_exit},
        halts                 = halts or {},
        weights               = weights or {t: 1.0},
        L                     = L,
        current_time          = current_time,
        platform_alternatives = platform_alternatives or {},
    )


# ---------------------------------------------------------------------------
# Helper B: multi-train line-only  AB
# ---------------------------------------------------------------------------

def make_line(
    trains,                          # [(tid, ab_entry), ...] — ab_exit = ab_entry + 60
    conflicts             = None,
    current_time          = 0,
    platform_alternatives = None,
    weights               = None,
    fixed_entry           = None,
    occupied              = None,
):
    """
    Bouwt een instantie met pad = ["AB"] voor elk trein.
    ab_exit = ab_entry + RUNTIME_AB (60 s) voor on-time rijden.

    Doordat er geen station-segmenten in het pad zitten, genereert step 2
    in mip_model.py ENKEL conflicten op "AB" en "AB_alt". Geen neveneffecten.
    """
    T    = [tid for tid, _ in trains]
    path = {tid: ["AB"] for tid, _ in trains}

    sched_entry = {}
    sched_exit  = {}
    runtime     = {}
    expected_exit = {}

    for tid, ab_entry in trains:
        ab_exit = ab_entry + RUNTIME_AB
        sched_entry[tid, "AB"]  = ab_entry
        sched_exit [tid, "AB"]  = ab_exit
        runtime    [tid, "AB"]  = RUNTIME_AB
        expected_exit[tid,"AB"] = ab_exit

    return dict(
        T                     = T,
        S                     = Sl_ab,           # S = {"AB"} (geen station-segs)
        Ss                    = frozenset(),
        Sl                    = Sl_ab,
        path                  = path,
        sched_entry           = sched_entry,
        sched_exit            = sched_exit,
        runtime               = runtime,
        dwell                 = {},              # geen stationssegmenten
        conflicts             = conflicts or {},
        occupied              = occupied or {},
        fixed_entry           = fixed_entry or {},
        expected_exit         = expected_exit,
        halts                 = {},
        weights               = weights or {tid: 1.0 for tid, _ in trains},
        L                     = L,
        current_time          = current_time,
        platform_alternatives = platform_alternatives or {},
    )


def solve_ok(inst, time_limit=15):
    """Lost op; eist status OPTIMAL (2) of TIME_LIMIT-met-incumbent (9)."""
    result = build_and_solve_model(**inst, verbose=False, time_limit=time_limit)
    model  = result[0]
    assert model.Status in (2, 9), (
        f"Onverwachte Gurobi-status {model.Status} "
        f"(2=OPTIMAL, 3=INFEASIBLE, 9=TIME_LIMIT)"
    )
    return result


# ===========================================================================
# 1. ReturnStructure
# ===========================================================================

class TestReturnStructure:
    """Return-tuple heeft precies 6 elementen met de juiste typen."""

    def test_tuple_length_is_six(self):
        result = build_and_solve_model(**make_full(), verbose=False)
        assert len(result) == 6, "Verwacht (model, entry, dep, delay, use, final_segment)"

    def test_tuple_types(self):
        import gurobipy as gp
        model, entry, dep, delay, use, fs = build_and_solve_model(
            **make_full(), verbose=False
        )
        assert isinstance(model, gp.Model)
        assert isinstance(entry, dict)
        assert isinstance(dep,   dict)
        assert isinstance(delay, dict)
        assert isinstance(use,   dict)    # tupledict is een dict-subklasse
        assert isinstance(fs,    dict)

    def test_final_segment_full_path(self):
        model, entry, dep, delay, use, fs = build_and_solve_model(
            **make_full(), verbose=False
        )
        assert fs[1] == "B"

    def test_final_segment_line_only(self):
        inst = make_line([(1, 0), (2, 200)])
        model, entry, dep, delay, use, fs = build_and_solve_model(**inst, verbose=False)
        assert fs[1] == "AB"
        assert fs[2] == "AB"

    def test_delay_only_on_final_segment_full_path(self):
        """delay-variabelen bestaan enkel voor het eindsegment (Törnquist z_{n_i})."""
        model, entry, dep, delay, use, fs = build_and_solve_model(
            **make_full(), verbose=False
        )
        # Final segment van het pad ["A","AB","B"] is "B"
        assert set(delay.keys()) == {(1, "B")}

    def test_delay_keys_line_three_trains(self):
        inst = make_line([(1, 0), (2, 200), (3, 400)])
        model, entry, dep, delay, use, fs = build_and_solve_model(**inst, verbose=False)
        assert set(delay.keys()) == {(1, "AB"), (2, "AB"), (3, "AB")}

    def test_entry_dep_cover_all_visits_full_path(self):
        model, entry, dep, delay, use, fs = build_and_solve_model(
            **make_full(), verbose=False
        )
        expected = {(1, s) for s in ["A", "AB", "B"]}
        assert set(entry.keys()) == expected
        assert set(dep.keys())   == expected


# ===========================================================================
# 2. BasicConstraints  —  C1, C2, C2b, C3, C5  (single train, full path)
# ===========================================================================

class TestBasicConstraints:

    def test_on_time_train_zero_delay(self):
        """On-time trein (current_time=0) heeft delay[1,'B'] ≈ 0."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        assert delay[1, "B"].X == pytest.approx(0.0, abs=1e-4)

    def test_late_start_creates_delay(self):
        """current_time=50 → trein vertrekt minstens 50 s te laat → delay ≈ 50."""
        model, entry, dep, delay, use, fs = solve_ok(make_full(current_time=50))
        assert delay[1, "B"].X == pytest.approx(50.0, abs=1e-4)

    def test_C2_path_continuity(self):
        """C2: entry[t, s_next] == dep[t, s] voor elk opeenvolgend segmentpaar."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        assert entry[1, "AB"].X == pytest.approx(dep[1, "A"].X,  abs=1e-4)
        assert entry[1, "B"].X  == pytest.approx(dep[1, "AB"].X, abs=1e-4)

    def test_C1_line_segment_minimum_runtime(self):
        """C1: dep[t, AB] − entry[t, AB] ≥ runtime[t, AB] = 60 s."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        assert dep[1, "AB"].X - entry[1, "AB"].X >= RUNTIME_AB - 1e-4

    def test_C1_station_segment_minimum_dwell(self):
        """C1: dep[t, A] − entry[t, A] ≥ dwell[t, A] = 30 s."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        assert dep[1, "A"].X - entry[1, "A"].X >= DWELL_A - 1e-4

    def test_C2b_no_early_entry(self):
        """C2b: entry[t, eerste segment] ≥ sched_entry (geen vroeg inrijden)."""
        model, entry, dep, delay, use, fs = solve_ok(make_full(a_entry=100, current_time=0))
        assert entry[1, "A"].X >= 100 - 1e-4

    def test_C3_delay_on_time(self):
        """C3: delay = max(0, dep_B − sched_exit_B) = 0 voor on-time trein."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        tardiness = max(0.0, dep[1, "B"].X - TRAVEL_TOTAL)
        assert delay[1, "B"].X == pytest.approx(tardiness, abs=1e-4)

    def test_C3_delay_late(self):
        """C3: delay = dep_B − sched_exit_B voor te late trein."""
        model, entry, dep, delay, use, fs = solve_ok(make_full(current_time=50))
        tardiness = max(0.0, dep[1, "B"].X - TRAVEL_TOTAL)
        assert delay[1, "B"].X == pytest.approx(tardiness, abs=1e-4)

    def test_C5_halt_minimum_departure(self):
        """C5: dep[t, B] ≥ sched_exit[t, B] als (t, B) een geplande stop is."""
        inst = make_full(halts={(1, "B"): True})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert dep[1, "B"].X >= TRAVEL_TOTAL - 1e-4

    def test_C5_halt_holds_even_when_late(self):
        """C5 geldt ook als de trein al te laat binnenkomt."""
        inst = make_full(halts={(1, "B"): True}, current_time=80)
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert dep[1, "B"].X >= TRAVEL_TOTAL - 1e-4

    def test_objective_nonnegative(self):
        """Objectiefwaarde ≥ 0 voor elke feasible instantie."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        assert model.ObjVal >= -1e-4


# ===========================================================================
# 3. OccupiedSegments
# ===========================================================================

class TestOccupiedSegments:
    """C1-variant voor in-uitvoering segmenten: dep[t,s] ≥ current_time + resterende."""

    def test_occupied_sets_minimum_departure(self):
        """dep[t, A] ≥ current_time + occupied[t, A]."""
        inst = make_full(
            current_time = 50,
            fixed_entry  = {(1, "A"): 40},   # trein startte A op t=40
            occupied     = {(1, "A"): 20},    # nog 20 s te gaan
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert dep[1, "A"].X >= 70 - 1e-4        # 50 + 20

    def test_occupied_overrides_duration_formula(self):
        """Als (t, s) in occupied: C1 = current_time + resterende (niet entry + duur)."""
        inst = make_full(
            current_time = 100,
            fixed_entry  = {(1, "A"): 50},
            occupied     = {(1, "A"): 5},    # slechts 5 s resterende
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert dep[1, "A"].X >= 105 - 1e-4      # 100 + 5  (niet 50 + 30 = 80)


# ===========================================================================
# 4. UseVariables  —  C6a resource-keuze
# ===========================================================================

class TestUseVariables:
    """C6a: Σ use[t, s, p] == 1 voor elk bezoek (t, s)."""

    def test_use_sum_one_no_alternatives_full_path(self):
        """Zonder alternatieven: use[t, s, s] == 1 voor alle bezoeken."""
        model, entry, dep, delay, use, fs = solve_ok(make_full())
        for s in ["A", "AB", "B"]:
            assert use[1, s, s].X == pytest.approx(1.0, abs=1e-4), (
                f"use[1,{s},{s}] = {use[1,s,s].X:.6f} ≠ 1"
            )

    def test_use_sum_one_no_alternatives_line(self):
        """Line-only: use[t,'AB','AB'] == 1 voor twee treinen zonder alternatieven."""
        inst = make_line([(1, 0), (2, 200)])
        model, entry, dep, delay, use, fs = solve_ok(inst)
        for t in [1, 2]:
            assert use[t, "AB", "AB"].X == pytest.approx(1.0, abs=1e-4)

    def test_use_sum_one_with_alternative(self):
        """Met één alternatief: use[t,'AB','AB'] + use[t,'AB','AB_alt'] == 1."""
        inst = make_line([(1, 0)], platform_alternatives={(1, "AB"): ["AB_alt"]})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        total = use[1, "AB", "AB"].X + use[1, "AB", "AB_alt"].X
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_use_keys_contain_both_platforms(self):
        """use-dict bevat sleutels voor zowel gepland als alternatief platform."""
        inst = make_line([(1, 0)], platform_alternatives={(1, "AB"): ["AB_alt"]})
        model, entry, dep, delay, use, fs = build_and_solve_model(**inst, verbose=False)
        assert (1, "AB", "AB")     in use
        assert (1, "AB", "AB_alt") in use

    def test_no_switch_without_conflict(self):
        """Zonder conflict: trein kiest gepland platform (warm start + geen incentief)."""
        inst = make_line([(1, 0)], platform_alternatives={(1, "AB"): ["AB_alt"]})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert use[1, "AB", "AB"].X == pytest.approx(1.0, abs=1e-4)

    def test_station_segments_have_no_alternatives(self):
        """Station-segmenten A en B hebben nooit alternatieven in de use-variabelen."""
        inst = make_full(platform_alternatives={(1, "AB"): ["AB_alt"]})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert (1, "A", "A_alt") not in use
        assert (1, "B", "B_alt") not in use
        assert (1, "A", "A") in use
        assert (1, "B", "B") in use


# ===========================================================================
# 5. ConflictResolution  —  C6b headway  (line-only network)
# ===========================================================================

class TestConflictResolution:
    """
    Alle conflicttests gebruiken het line-only netwerk (pad=["AB"]).
    Zo genereert step 2 in mip_model.py enkel conflicten op "AB" en "AB_alt",
    zonder neveneffecten op station-segmenten.

    Timing:
      Train 1: AB = [0, 60]
      Train 2: AB = [20, 80]  → overlap met trein 1 op [20, 60]
    """

    def test_two_trains_far_apart_both_on_time(self):
        """Twee treinen ver uit elkaar: geen conflictconstraint → beide on-time."""
        inst = make_line([(1, 0), (2, 500)])
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert delay[1, "AB"].X == pytest.approx(0.0, abs=1e-4)
        assert delay[2, "AB"].X == pytest.approx(0.0, abs=1e-4)

    def test_conflict_prevents_overlap(self):
        """Twee treinen overlappen op AB; conflict → één moet wachten."""
        inst = make_line([(1, 0), (2, 20)], conflicts={"AB": [(1, 2)]})
        model, entry, dep, delay, use, fs = solve_ok(inst)

        e1, d1 = entry[1, "AB"].X, dep[1, "AB"].X
        e2, d2 = entry[2, "AB"].X, dep[2, "AB"].X
        no_overlap = (d1 <= e2 + 1e-4) or (d2 <= e1 + 1e-4)
        assert no_overlap, (
            f"Overlap op AB: T1=[{e1:.1f},{d1:.1f}], T2=[{e2:.1f},{d2:.1f}]"
        )

    def test_conflict_produces_positive_total_delay(self):
        """Overlappende schema's + conflict → totale vertraging > 0."""
        inst = make_line([(1, 0), (2, 20)], conflicts={"AB": [(1, 2)]})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert delay[1, "AB"].X + delay[2, "AB"].X > 0.5

    def test_high_weight_train_gets_priority(self):
        """Trein met hoog gewicht wordt vooraan gepland → laagste vertraging."""
        inst = make_line([(1, 0), (2, 20)],
                         conflicts={"AB": [(1, 2)]},
                         weights={1: 10.0, 2: 1.0})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        # Trein 1 (gewicht 10) heeft ≤ vertraging dan trein 2 (gewicht 1)
        assert delay[1, "AB"].X <= delay[2, "AB"].X + 1e-4

    def test_symmetric_weights_feasible_solution(self):
        """Gelijke gewichten: solver vindt een feasible oplossing."""
        inst = make_line([(1, 0), (2, 20)], conflicts={"AB": [(1, 2)]})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert model.ObjVal >= 0.0

    def test_three_trains_chain_no_pairwise_overlap(self):
        """Drie treinen; alle paren conflicteren → geen paargewijze overlap."""
        inst = make_line([(1, 0), (2, 10), (3, 20)],
                         conflicts={"AB": [(1, 2), (2, 3), (1, 3)]})
        model, entry, dep, delay, use, fs = solve_ok(inst)

        sorted_ivs = sorted(
            (entry[t, "AB"].X, dep[t, "AB"].X) for t in [1, 2, 3]
        )
        for i in range(len(sorted_ivs) - 1):
            _, d = sorted_ivs[i]
            e, _ = sorted_ivs[i + 1]
            assert d <= e + 1e-4, (
                f"Overlap in gesorteerde AB-intervallen: dep={d:.1f} > entry={e:.1f}"
            )

    def test_naturally_satisfied_conflict_no_delay(self):
        """
        Conflict op AB tussen twee treinen die reeds voldoende gescheiden zijn:
        de constraint is triviaal voldaan (train 1 vertrekt vóór train 2 arriveert)
        → geen vertraging voor beide treinen.
        """
        # Train 1: AB = [0, 60]   Train 2: AB = [200, 260]  → geen overlap
        inst = make_line([(1, 0), (2, 200)], conflicts={"AB": [(1, 2)]})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert delay[1, "AB"].X == pytest.approx(0.0, abs=1e-4)
        assert delay[2, "AB"].X == pytest.approx(0.0, abs=1e-4)


# ===========================================================================
# 6. Retracking  —  het enige between-station segment AB (∈ Sl)
#    Gebruikt het line-only netwerk (geen station-nevenconflicten).
# ===========================================================================

class TestRetracking:
    """
    Timing (line-only):
      Train 1: AB = [0, 60]
      Train 2: AB = [20, 80]  → overlap met train 1 op [20, 60]

    Met SWITCH_PENALTY=0 en conflict op AB: solver stuurt trein 2 naar AB_alt.
    """

    def test_retrack_avoids_conflict_zero_delay(self):
        """
        Trein 1 vast op AB; trein 2 heeft AB_alt.
        Conflict op AB → solver stuurt trein 2 naar AB_alt → 0 vertraging.
        """
        inst = make_line(
            [(1, 0), (2, 20)],
            conflicts={"AB": [(1, 2)]},
            platform_alternatives={(2, "AB"): ["AB_alt"]},
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)

        assert use[2, "AB", "AB_alt"].X == pytest.approx(1.0, abs=1e-4), (
            "Trein 2 had moeten switchen naar AB_alt"
        )
        assert delay[1, "AB"].X == pytest.approx(0.0, abs=1e-4)
        assert delay[2, "AB"].X == pytest.approx(0.0, abs=1e-4)

    def test_high_switch_penalty_prevents_switch(self):
        """
        SWITCH_PENALTY >> verwachte vertraging → solver wacht liever dan te switchen.

        Zonder switch: trein 2 wacht tot trein 1 AB heeft verlaten.
          entry[2,"AB"] ≥ dep[1,"AB"] = 60
          dep[2,"AB"] = 60 + 60 = 120
          delay[2,"AB"] = 120 − 80 = 40 s

        ObjVal zonder switch = 40 s.
        ObjVal met switch = SWITCH_PENALTY = 10 000 s.
        Solver kiest 40 < 10 000 → geen switch.
        """
        original = mm.SWITCH_PENALTY
        try:
            mm.SWITCH_PENALTY = 10_000.0

            inst = make_line(
                [(1, 0), (2, 20)],
                conflicts={"AB": [(1, 2)]},
                platform_alternatives={(2, "AB"): ["AB_alt"]},
            )
            model, entry, dep, delay, use, fs = solve_ok(inst)

            assert use[2, "AB", "AB"].X == pytest.approx(1.0, abs=1e-4), (
                "Trein 2 had NIET mogen switchen bij SWITCH_PENALTY=10 000"
            )
            assert delay[2, "AB"].X == pytest.approx(40.0, abs=1e-4)
        finally:
            mm.SWITCH_PENALTY = original

    def test_zero_penalty_switch_is_optimal(self):
        """SWITCH_PENALTY = 0 → switch gratis; solver switcht als het delay bespaart."""
        original = mm.SWITCH_PENALTY
        try:
            mm.SWITCH_PENALTY = 0.0
            inst = make_line(
                [(1, 0), (2, 20)],
                conflicts={"AB": [(1, 2)]},
                platform_alternatives={(2, "AB"): ["AB_alt"]},
            )
            model, entry, dep, delay, use, fs = solve_ok(inst)

            assert use[2, "AB", "AB_alt"].X == pytest.approx(1.0, abs=1e-4)
            assert delay[2, "AB"].X          == pytest.approx(0.0, abs=1e-4)
        finally:
            mm.SWITCH_PENALTY = original

    def test_both_retrackable_solver_separates_resources(self):
        """
        Beide treinen kunnen AB of AB_alt kiezen.
        Conflict op AB → solver plaatst ze op verschillende resources
        of ordent ze correct als ze hetzelfde kiezen.
        """
        inst = make_line(
            [(1, 0), (2, 20)],
            conflicts={"AB": [(1, 2)]},
            platform_alternatives={(1, "AB"): ["AB_alt"], (2, "AB"): ["AB_alt"]},
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)

        p1 = "AB" if round(use[1, "AB", "AB"].X) else "AB_alt"
        p2 = "AB" if round(use[2, "AB", "AB"].X) else "AB_alt"

        # Exact één platform per trein (C6a)
        assert round(use[1,"AB","AB"].X) + round(use[1,"AB","AB_alt"].X) == 1
        assert round(use[2,"AB","AB"].X) + round(use[2,"AB","AB_alt"].X) == 1

        # Als beide hetzelfde platform kiezen: geen overlap
        if p1 == p2:
            e1, d1 = entry[1, "AB"].X, dep[1, "AB"].X
            e2, d2 = entry[2, "AB"].X, dep[2, "AB"].X
            assert (d1 <= e2 + 1e-4) or (d2 <= e1 + 1e-4), (
                f"Beide treinen op {p1} maar overlappend"
            )

    def test_fixed_on_alt_no_false_conflict_with_planned(self):
        """
        Bug-fix 5 reproductie: trein 2 wordt naar AB_alt gestuurd.
        AB en AB_alt zijn gescheiden resources → geen conflict → beide on-time.
        """
        inst = make_line(
            [(1, 0), (2, 20)],
            conflicts={"AB": [(1, 2)]},
            platform_alternatives={(2, "AB"): ["AB_alt"]},
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)

        if round(use[2, "AB", "AB_alt"].X) == 1:
            # Trein 2 op AB_alt, trein 1 op AB → aparte resources
            assert delay[1, "AB"].X == pytest.approx(0.0, abs=1e-4)
            assert delay[2, "AB"].X == pytest.approx(0.0, abs=1e-4)

    def test_conflict_window_filters_distant_trains(self):
        """
        Twee treinen met AB_alt, expected_exit-verschil > RETRACK_CONFLICT_WINDOW.
        Step 2 voegt geen conflict toe → geen ord-variabele → beide on-time.
        """
        from config.settings import RETRACK_CONFLICT_WINDOW
        gap = RETRACK_CONFLICT_WINDOW + 200

        inst = make_line(
            [(1, 0), (2, gap)],
            platform_alternatives={(1, "AB"): ["AB_alt"], (2, "AB"): ["AB_alt"]},
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert delay[1, "AB"].X == pytest.approx(0.0, abs=1e-4)
        assert delay[2, "AB"].X == pytest.approx(0.0, abs=1e-4)

    def test_conflict_window_triggers_near_trains(self):
        """
        Twee treinen met AB_alt en expected_exit-verschil ≤ RETRACK_CONFLICT_WINDOW.
        Step 2 voegt een retracking-conflict toe → solver scheidt ze.
        """
        from config.settings import RETRACK_CONFLICT_WINDOW
        gap = max(0, RETRACK_CONFLICT_WINDOW - 60)   # net binnen het venster

        inst = make_line(
            [(1, 0), (2, gap)],
            platform_alternatives={(1, "AB"): ["AB_alt"], (2, "AB"): ["AB_alt"]},
        )
        model, entry, dep, delay, use, fs = solve_ok(inst)

        p1 = "AB" if round(use[1, "AB", "AB"].X) else "AB_alt"
        p2 = "AB" if round(use[2, "AB", "AB"].X) else "AB_alt"

        if p1 == p2:
            e1, d1 = entry[1, "AB"].X, dep[1, "AB"].X
            e2, d2 = entry[2, "AB"].X, dep[2, "AB"].X
            assert (d1 <= e2 + 1e-4) or (d2 <= e1 + 1e-4), (
                f"Treinen op {p1} overlappen: T1=[{e1:.1f},{d1:.1f}], "
                f"T2=[{e2:.1f},{d2:.1f}]"
            )

    def test_alternatives_only_for_between_station_segment(self):
        """Full-path: platform_alternatives op 'AB' (∈ Sl) heeft GEEN effect op 'A'/'B'."""
        inst = make_full(platform_alternatives={(1, "AB"): ["AB_alt"]})
        model, entry, dep, delay, use, fs = solve_ok(inst)

        # AB heeft twee keuzes
        assert (1, "AB", "AB")     in use
        assert (1, "AB", "AB_alt") in use
        # Stations nooit
        assert (1, "A", "A_alt") not in use
        assert (1, "B", "B_alt") not in use


# ===========================================================================
# 7. FixedEntry
# ===========================================================================

class TestFixedEntry:

    def test_fixed_entry_pins_first_segment(self):
        """fixed_entry: entry[t, A] wordt gepind op de opgegeven waarde."""
        inst = make_full(fixed_entry={(1, "A"): 7.0})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert entry[1, "A"].X == pytest.approx(7.0, abs=1e-4)

    def test_fixed_entry_pins_middle_segment(self):
        """fixed_entry voor een middelste segment wordt correct gepind."""
        # dep[1,"A"] is vrij (minimaal 30), maar C2 koppelt entry[1,"AB"]=dep[1,"A"].
        # Met lb=ub=45 op entry[1,"AB"] forceert C2 dep[1,"A"]=45.
        inst = make_full(fixed_entry={(1, "AB"): 45.0})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert entry[1, "AB"].X == pytest.approx(45.0, abs=1e-4)

    def test_non_fixed_entry_respects_C2b(self):
        """Niet-vaste entry: trein rijdt nooit vroeger in dan gepland (C2b)."""
        inst = make_full(a_entry=100, current_time=0)
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert entry[1, "A"].X >= 100 - 1e-4

    def test_fixed_entry_line_only(self):
        """fixed_entry werkt ook voor een AB-only pad."""
        inst = make_line([(1, 0)], fixed_entry={(1, "AB"): 15.0})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert entry[1, "AB"].X == pytest.approx(15.0, abs=1e-4)


# ===========================================================================
# 8. WarmStart
# ===========================================================================

class TestWarmStart:
    """Indirecte verificatie van warm-start via convergentie-garantie."""

    def test_single_train_with_alternative_converges(self):
        """Model met warm start op gepland platform convergeert naar OPTIMAL."""
        inst = make_line([(1, 0)], platform_alternatives={(1, "AB"): ["AB_alt"]})
        model, entry, dep, delay, use, fs = build_and_solve_model(**inst, verbose=False)
        assert model.Status == 2

    def test_conflict_pair_converges(self):
        """Model met ord warm-start op timetable-volgorde convergeert naar OPTIMAL."""
        inst = make_line([(1, 0), (2, 20)], conflicts={"AB": [(1, 2)]})
        model, entry, dep, delay, use, fs = build_and_solve_model(**inst, verbose=False)
        assert model.Status == 2

    def test_warm_start_does_not_block_optimal_switch(self):
        """
        Warm start staat op gepland platform (geen switch).
        Als switchen optimaal is (SWITCH_PENALTY=0), bereikt de solver
        toch de optimale switch-oplossing (ObjVal=0).
        """
        original = mm.SWITCH_PENALTY
        try:
            mm.SWITCH_PENALTY = 0.0
            inst = make_line(
                [(1, 0), (2, 20)],
                conflicts={"AB": [(1, 2)]},
                platform_alternatives={(2, "AB"): ["AB_alt"]},
            )
            model, entry, dep, delay, use, fs = solve_ok(inst)
            assert model.ObjVal == pytest.approx(0.0, abs=1e-4)
        finally:
            mm.SWITCH_PENALTY = original


# ===========================================================================
# 9. Stress
# ===========================================================================

class TestStress:

    @staticmethod
    def _all_pairs(ids):
        return [(i, j) for i in ids for j in ids if i < j]

    def test_five_trains_all_conflicting_no_pairwise_overlap(self):
        """5 treinen op AB, alle C(5,2)=10 paren conflicteren → geen overlap."""
        trains = [(i, i * 10) for i in range(1, 6)]
        inst   = make_line(trains, conflicts={"AB": self._all_pairs(range(1, 6))})

        model, entry, dep, delay, use, fs = solve_ok(inst, time_limit=30)

        sorted_ivs = sorted(
            (entry[t, "AB"].X, dep[t, "AB"].X) for t in range(1, 6)
        )
        for i in range(len(sorted_ivs) - 1):
            _, d = sorted_ivs[i]
            e, _ = sorted_ivs[i + 1]
            assert d <= e + 1e-4, (
                f"Overlap na sortering: dep={d:.1f} > entry={e:.1f}"
            )

    def test_five_trains_three_retrackable_no_same_resource_overlap(self):
        """
        5 treinen; treinen 3-5 kunnen naar AB_alt.
        Alle paren conflicteren. Twee treinen op hetzelfde platform mogen
        niet overlappen.
        """
        trains = [(i, i * 10) for i in range(1, 6)]
        inst   = make_line(
            trains,
            conflicts={"AB": self._all_pairs(range(1, 6))},
            platform_alternatives={
                (3, "AB"): ["AB_alt"],
                (4, "AB"): ["AB_alt"],
                (5, "AB"): ["AB_alt"],
            },
        )

        model, entry, dep, delay, use, fs = solve_ok(inst, time_limit=30)

        def chosen_platform(t):
            opts = ["AB"] + (["AB_alt"] if t >= 3 else [])
            for p in opts:
                if round(use[t, "AB", p].X) == 1:
                    return p
            raise AssertionError(f"Trein {t} heeft geen platform gekozen")

        platforms = {t: chosen_platform(t) for t in range(1, 6)}

        for t1 in range(1, 6):
            for t2 in range(t1 + 1, 6):
                if platforms[t1] == platforms[t2]:
                    e1, d1 = entry[t1, "AB"].X, dep[t1, "AB"].X
                    e2, d2 = entry[t2, "AB"].X, dep[t2, "AB"].X
                    assert (d1 <= e2 + 1e-4) or (d2 <= e1 + 1e-4), (
                        f"T{t1} en T{t2} overlappen op {platforms[t1]}: "
                        f"[{e1:.1f},{d1:.1f}] vs [{e2:.1f},{d2:.1f}]"
                    )

    def test_objective_nonnegative_under_conflicts(self):
        """Objectiefwaarde ≥ 0 bij meerdere conflicterende treinen."""
        trains = [(i, i * 10) for i in range(1, 4)]
        inst   = make_line(trains, conflicts={"AB": self._all_pairs(range(1, 4))})
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert model.ObjVal >= -1e-4

    def test_empty_train_set_returns_optimal_zero(self):
        """Lege treinen-set → status OPTIMAL (2), ObjVal=0, lege dicts."""
        inst = make_line([])
        model, entry, dep, delay, use, fs = build_and_solve_model(**inst, verbose=False)
        assert model.Status == 2,            "Verwacht OPTIMAL (2)"
        assert model.ObjVal == pytest.approx(0.0, abs=1e-4)
        assert entry         == {}
        assert dep           == {}
        assert delay         == {}
        assert len(use)      == 0
        assert fs            == {}

    def test_repeated_solves_deterministic_objective(self):
        """Herhaling van hetzelfde model geeft dezelfde objectiefwaarde."""
        inst = make_full()
        objectives = [
            build_and_solve_model(**inst, verbose=False)[0].ObjVal
            for _ in range(3)
        ]
        assert all(abs(v - objectives[0]) < 1e-4 for v in objectives)

    def test_large_current_time_delay_finite_and_nonneg(self):
        """Vertraging is eindig en niet-negatief bij grote current_time."""
        inst = make_line([(1, 0)], current_time=3600)
        model, entry, dep, delay, use, fs = solve_ok(inst)
        assert delay[1, "AB"].X >= 0
        assert delay[1, "AB"].X < 2 * L
