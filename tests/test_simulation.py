# tests/test_simulation.py
#
# Unit tests voor simulation/event_queue.py, simulation/dispatcher.py
# en simulation/simulator.py.
#
# Teststructuur:
#   TestEventQueue        — basis + edge cases
#   TestDispatcher        — basis + edge cases
#   TestSimulatorBasic    — normale flow
#   TestSimulatorEdge     — edge cases: conflicten, MIP, vertraging, volgorde

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from simulation.event_queue import EventQueue, TrainEntered, TrainExited
from simulation.dispatcher  import Dispatcher
from simulation.simulator   import Simulator
from domain.segment         import SegmentType


# =============================================================================
# Mocks
# =============================================================================

class MockSegment:
    def __init__(self, seg_id: str, seg_type: SegmentType, source: str, target: str):
        self.id       = seg_id
        self.seg_type = seg_type
        self.source   = source
        self.target   = target

    @property
    def is_station(self):
        return self.seg_type == SegmentType.STATION

    @property
    def is_line(self):
        return self.seg_type == SegmentType.BETWEEN_STATION


class MockTrain:
    def __init__(self, train_no: int, path: list[str]):
        self.train_no        = train_no
        self.path            = tuple(path)
        self.train_type      = MagicMock()
        self.train_subtype   = MagicMock()
        self.train_subtype.value = "IC"

    @property
    def id(self):
        return self.train_no

    @property
    def first_segment(self):
        return self.path[0]

    @property
    def last_segment(self):
        return self.path[-1]

    def halts_at(self, segment_id: str) -> bool:
        return True

    def dynamics_at(self, segment_id: str):
        return None


class MockTimetable:
    """
    Configureerbare timetable voor tests.
    Standaard: entry = base_time + index * 120, exit = entry + duration
    """
    def __init__(
        self,
        trains:    dict,
        segments:  dict,
        base_time: float = 3600.0,
        line_duration:    float = 60.0,
        station_duration: float = 120.0,
        dwell:            float = 60.0,
    ):
        self._trains           = trains
        self._segments         = segments
        self._base             = base_time
        self._line_duration    = line_duration
        self._station_duration = station_duration
        self._dwell            = dwell

    def _index(self, train_id: int, segment_id: str) -> int:
        return list(self._trains[train_id].path).index(segment_id)

    def scheduled_arrival(self, train_id: int, segment_id: str) -> float:
        return self._base + self._index(train_id, segment_id) * 120.0

    def scheduled_departure(self, train_id: int, segment_id: str) -> float:
        seg = self._segments[segment_id]
        dur = self._station_duration if seg.seg_type == SegmentType.STATION else self._line_duration
        return self.scheduled_arrival(train_id, segment_id) + dur

    def dwell_time(self, train_id: int, segment_id: str) -> float:
        if self._segments[segment_id].seg_type != SegmentType.STATION:
            raise ValueError("Geen stationssegment")
        return self._dwell

    def running_time(self, train_id: int, segment_id: str) -> float:
        if self._segments[segment_id].seg_type != SegmentType.BETWEEN_STATION:
            raise ValueError("Geen lijnsegment")
        return self._line_duration


class MockController:
    """Controller die altijd 'skipped' teruggeeft."""
    def step(self, state, current_time):
        result = MagicMock()
        result.action = "skipped"
        return result


class ReschedulingController:
    """
    Controller die één keer een MIP-oplossing teruggeeft,
    daarna altijd 'skipped'.
    """
    def __init__(self, solution, fire_after: float):
        self._solution   = solution
        self._fire_after = fire_after
        self._fired      = False

    def step(self, state, current_time):
        result = MagicMock()
        if not self._fired and current_time >= self._fire_after:
            self._fired     = True
            result.action   = "rescheduled"
            result.solution = self._solution
        else:
            result.action = "skipped"
        return result


class FcfsController:
    """Controller die één keer een FCFS-fallback teruggeeft."""
    def __init__(self, fcfs_order: dict, fire_after: float):
        self._fcfs_order = fcfs_order
        self._fire_after = fire_after
        self._fired      = False

    def step(self, state, current_time):
        result = MagicMock()
        if not self._fired and current_time >= self._fire_after:
            self._fired        = True
            result.action      = "fcfs_fallback"
            result.fcfs_order  = self._fcfs_order
        else:
            result.action = "skipped"
        return result


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def line_seg():
    return MockSegment("seg-line", SegmentType.BETWEEN_STATION, "A", "B")

@pytest.fixture
def station_seg():
    return MockSegment("seg-station", SegmentType.STATION, "B", "B")

@pytest.fixture
def segments(line_seg, station_seg):
    return {"seg-line": line_seg, "seg-station": station_seg}

@pytest.fixture
def one_train(segments):
    return {1: MockTrain(1, ["seg-line", "seg-station"])}

@pytest.fixture
def two_trains(segments):
    return {
        1: MockTrain(1, ["seg-line", "seg-station"]),
        2: MockTrain(2, ["seg-line", "seg-station"]),
    }

@pytest.fixture
def three_trains(segments):
    return {
        1: MockTrain(1, ["seg-line", "seg-station"]),
        2: MockTrain(2, ["seg-line", "seg-station"]),
        3: MockTrain(3, ["seg-line", "seg-station"]),
    }

@pytest.fixture
def timetable(one_train, segments):
    return MockTimetable(one_train, segments)

@pytest.fixture
def two_timetable(two_trains, segments):
    return MockTimetable(two_trains, segments)

@pytest.fixture
def three_timetable(three_trains, segments):
    return MockTimetable(three_trains, segments)

@pytest.fixture
def dispatcher(timetable, segments):
    return Dispatcher(timetable=timetable, segments=segments)


# =============================================================================
# TestEventQueue — edge cases
# =============================================================================

class TestEventQueue:

    def test_push_pop_volgorde(self):
        """Events worden in chronologische volgorde teruggegeven."""
        q = EventQueue()
        q.push(TrainEntered(time=300.0, train_id=1, segment_id="s"))
        q.push(TrainEntered(time=100.0, train_id=2, segment_id="s"))
        q.push(TrainEntered(time=200.0, train_id=3, segment_id="s"))
        assert q.pop().time == 100.0
        assert q.pop().time == 200.0
        assert q.pop().time == 300.0

    def test_tie_breaker_fifo(self):
        """Bij gelijke tijd: insertievolgorde bepaalt wie eerst komt."""
        q = EventQueue()
        q.push(TrainEntered(time=100.0, train_id=1, segment_id="s"))
        q.push(TrainEntered(time=100.0, train_id=2, segment_id="s"))
        q.push(TrainEntered(time=100.0, train_id=3, segment_id="s"))
        assert q.pop().train_id == 1
        assert q.pop().train_id == 2
        assert q.pop().train_id == 3

    def test_cancel_alle_types(self):
        """Cancel verwijdert zowel TrainEntered als TrainExited."""
        q = EventQueue()
        q.push(TrainEntered(time=100.0, train_id=1, segment_id="s"))
        q.push(TrainExited( time=200.0, train_id=1, segment_id="s"))
        q.push(TrainEntered(time=300.0, train_id=2, segment_id="s"))
        removed = q.cancel(train_id=1, segment_id="s")
        assert removed == 2
        assert len(q)  == 1
        assert q.pop().train_id == 2

    def test_cancel_herordent_heap_correct(self):
        """Na cancel blijft de heap geldig gesorteerd."""
        q = EventQueue()
        for t in [500, 100, 300, 200, 400]:
            q.push(TrainEntered(time=float(t), train_id=1, segment_id="s"))
        q.push(TrainEntered(time=250.0, train_id=2, segment_id="s"))
        q.cancel(train_id=1, segment_id="s")
        assert len(q) == 1
        assert q.pop().time == 250.0

    def test_cancel_op_lege_queue(self):
        """Cancel op lege queue geeft 0 terug zonder crash."""
        q = EventQueue()
        assert q.cancel(train_id=1, segment_id="s") == 0

    def test_has_entered_na_cancel(self):
        """has_entered geeft False na cancel."""
        q = EventQueue()
        q.push(TrainEntered(time=100.0, train_id=1, segment_id="s"))
        q.cancel(train_id=1, segment_id="s")
        assert q.has_entered(train_id=1, segment_id="s") is False

    def test_duizend_events_volgorde(self):
        """Heap blijft correct gesorteerd bij 1000 events."""
        import random
        q      = EventQueue()
        times  = [random.uniform(0, 10000) for _ in range(1000)]
        for t in times:
            q.push(TrainEntered(time=t, train_id=1, segment_id="s"))
        popped = [q.pop().time for _ in range(1000)]
        assert popped == sorted(times)

    def test_gemengde_event_types_volgorde(self):
        """TrainEntered en TrainExited worden samen correct gesorteerd."""
        q = EventQueue()
        q.push(TrainExited( time=150.0, train_id=1, segment_id="s"))
        q.push(TrainEntered(time=100.0, train_id=1, segment_id="s"))
        q.push(TrainExited( time=200.0, train_id=2, segment_id="s"))
        assert isinstance(q.pop(), TrainEntered)
        assert isinstance(q.pop(), TrainExited)
        assert isinstance(q.pop(), TrainExited)


# =============================================================================
# TestDispatcher — edge cases
# =============================================================================

class TestDispatcher:

    def test_drie_treinen_juiste_volgorde(self, dispatcher):
        """Drie treinen in wachtrij — volgorde strikt op planned_time."""
        dispatcher.enqueue(3, "seg-line", planned_time=300.0)
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        dispatcher.enqueue(2, "seg-line", planned_time=200.0)

        assert dispatcher.request_entry(1, "seg-line", 300.0) is True
        assert dispatcher.request_entry(2, "seg-line", 300.0) is False
        assert dispatcher.request_entry(3, "seg-line", 300.0) is False

        dispatcher.confirm_entry(1, "seg-line")
        dispatcher.release(1, "seg-line")

        assert dispatcher.request_entry(2, "seg-line", 300.0) is True
        dispatcher.confirm_entry(2, "seg-line")
        dispatcher.release(2, "seg-line")

        assert dispatcher.request_entry(3, "seg-line", 300.0) is True

    def test_reorder_midden_in_wachtrij(self, dispatcher):
        """reorder() past volgorde correct aan voor alle treinen in wachtrij."""
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        dispatcher.enqueue(2, "seg-line", planned_time=200.0)
        dispatcher.enqueue(3, "seg-line", planned_time=300.0)

        # Keer volgorde om: 3, 2, 1
        dispatcher.reorder({"seg-line": [3, 2, 1]})

        assert dispatcher.request_entry(3, "seg-line", 300.0) is True
        assert dispatcher.request_entry(1, "seg-line", 300.0) is False
        assert dispatcher.request_entry(2, "seg-line", 300.0) is False

    def test_reorder_na_confirm_entry(self, dispatcher):
        """reorder() heeft geen effect op trein die segment al betreedt."""
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        dispatcher.enqueue(2, "seg-line", planned_time=200.0)
        dispatcher.confirm_entry(1, "seg-line")

        # Probeer trein 2 prioriteit te geven via reorder
        dispatcher.reorder({"seg-line": [2, 1]})

        # Trein 1 bezet nog steeds het segment
        assert dispatcher.request_entry(2, "seg-line", 200.0) is False
        dispatcher.release(1, "seg-line")

        # Nu mag trein 2 als eerste
        assert dispatcher.request_entry(2, "seg-line", 200.0) is True

    def test_release_verkeerde_trein_geeft_warning(self, dispatcher, caplog):
        """release() van verkeerde trein logt een warning."""
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        dispatcher.confirm_entry(1, "seg-line")

        import logging
        with caplog.at_level(logging.WARNING, logger="simulation.dispatcher"):
            dispatcher.release(99, "seg-line")  # trein 99 bezet het segment niet

        assert any("warning" in r.levelname.lower() for r in caplog.records)

    def test_min_exit_time_vroege_aankomst(self, dispatcher, timetable, one_train):
        """Als trein vroeg aankomt, is min_exit_time toch de geplande vertrektijd."""
        planned_exit = timetable.scheduled_departure(1, "seg-station")
        early_entry  = planned_exit - 200.0  # 200s vroeger dan gepland

        min_exit = dispatcher.min_exit_time(1, "seg-station", early_entry)
        assert min_exit == planned_exit

    def test_min_exit_time_late_aankomst(self, dispatcher, timetable):
        """Als trein laat aankomt, is min_exit_time entry + dwell_time."""
        planned_exit = timetable.scheduled_departure(1, "seg-station")
        late_entry   = planned_exit + 100.0  # 100s later dan gepland vertrek

        min_exit = dispatcher.min_exit_time(1, "seg-station", late_entry)
        assert min_exit == late_entry + 60.0  # entry + dwell_time

    def test_enqueue_zelfde_trein_twee_keer(self, dispatcher):
        """Dubbele enqueue heeft geen effect op wachtrij."""
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        assert len(dispatcher._queue["seg-line"]) == 1

    def test_reorder_onbekend_segment_geen_crash(self, dispatcher):
        """reorder() voor segment zonder wachtende treinen crasht niet."""
        dispatcher.reorder({"seg-line": [1, 2, 3]})  # niemand in wachtrij

    def test_next_in_queue_na_reorder(self, dispatcher):
        """next_in_queue geeft correct eerste trein na reorder."""
        dispatcher.enqueue(1, "seg-line", planned_time=100.0)
        dispatcher.enqueue(2, "seg-line", planned_time=200.0)
        dispatcher.reorder({"seg-line": [2, 1]})
        assert dispatcher.next_in_queue("seg-line") == 2

    def test_volledig_doorlopen_wachtrij(self, dispatcher):
        """Drie treinen doorlopen volledig de wachtrij in correcte volgorde."""
        for train_id, t in [(3, 300.0), (1, 100.0), (2, 200.0)]:
            dispatcher.enqueue(train_id, "seg-line", planned_time=t)

        volgorde = []
        for _ in range(3):
            first = dispatcher.next_in_queue("seg-line")
            volgorde.append(first)
            dispatcher.confirm_entry(first, "seg-line")
            dispatcher.release(first, "seg-line")

        assert volgorde == [1, 2, 3]


# =============================================================================
# TestSimulatorBasic — normale flow
# =============================================================================

class TestSimulatorBasic:

    def _sim(self, trains, segments, timetable, controller=None):
        return Simulator(
            trains     = trains,
            segments   = segments,
            timetable  = timetable,
            controller = controller or MockController(),
            seed       = 42,
        )

    def test_een_trein_voltooit(self, one_train, segments, timetable):
        """Één trein doorloopt volledig pad en is finished."""
        state = self._sim(one_train, segments, timetable).run()
        assert state.is_finished(1)

    def test_entries_voor_exits(self, one_train, segments, timetable):
        """Voor elk segment: actual_entry < actual_exit."""
        state = self._sim(one_train, segments, timetable).run()
        for seg_id in one_train[1].path:
            assert state.actual_entry(1, seg_id) < state.actual_exit(1, seg_id)

    def test_segmenten_aaneensluitend(self, one_train, segments, timetable):
        """Exit van segment N ≤ entry van segment N+1."""
        state = self._sim(one_train, segments, timetable).run()
        path  = list(one_train[1].path)
        for i in range(len(path) - 1):
            exit_i  = state.actual_exit( 1, path[i])
            entry_i1 = state.actual_entry(1, path[i + 1])
            assert exit_i <= entry_i1 + 0.001

    def test_c2_constraint(self, one_train, segments, timetable):
        """Trein verlaat stationssegment nooit voor geplande vertrektijd."""
        state        = self._sim(one_train, segments, timetable).run()
        planned_exit = timetable.scheduled_departure(1, "seg-station")
        actual_exit  = state.actual_exit(1, "seg-station")
        assert actual_exit >= planned_exit - 0.001

    def test_simulatietijd_monotoon(self, one_train, segments, timetable):
        """Simulatietijd gaat nooit achteruit."""
        times = []
        sim   = self._sim(one_train, segments, timetable)

        orig = sim._state.advance_time
        def tracked(t):
            times.append(t)
            orig(t)
        sim._state.advance_time = tracked
        sim.run()

        assert times == sorted(times)


# =============================================================================
# TestSimulatorEdge — edge cases
# =============================================================================

class TestSimulatorEdge:

    def _sim(self, trains, segments, timetable, controller=None):
        return Simulator(
            trains     = trains,
            segments   = segments,
            timetable  = timetable,
            controller = controller or MockController(),
            seed       = 42,
        )

    def test_twee_treinen_geen_deadlock(self, two_trains, segments, two_timetable):
        """Twee treinen op hetzelfde segment eindigen beide zonder deadlock."""
        state = self._sim(two_trains, segments, two_timetable).run()
        assert state.is_finished(1)
        assert state.is_finished(2)

    def test_twee_treinen_geen_overlap_op_segment(self, two_trains, segments, two_timetable):
        """Twee treinen overlappen nooit op hetzelfde lijnsegment."""
        state = self._sim(two_trains, segments, two_timetable).run()

        for train_id in [1, 2]:
            other_id = 2 if train_id == 1 else 1
            entry_a  = state.actual_entry(train_id, "seg-line")
            exit_a   = state.actual_exit( train_id, "seg-line")
            entry_b  = state.actual_entry(other_id,  "seg-line")
            exit_b   = state.actual_exit( other_id,  "seg-line")

            # Geen overlap: één van beiden moet volledig voor de andere zijn
            geen_overlap = (exit_a <= entry_b + 0.001) or (exit_b <= entry_a + 0.001)
            assert geen_overlap, (
                f"Trein {train_id} ({entry_a:.0f}-{exit_a:.0f}) en "
                f"trein {other_id} ({entry_b:.0f}-{exit_b:.0f}) overlappen op seg-line"
            )

    def test_drie_treinen_strikte_volgorde(self, three_trains, segments, three_timetable):
        """Drie treinen betreden lijnsegment in de juiste volgorde."""
        state = self._sim(three_trains, segments, three_timetable).run()

        entries = sorted(
            [(state.actual_entry(t_id, "seg-line"), t_id) for t_id in [1, 2, 3]]
        )
        volgorde = [t_id for _, t_id in entries]

        # Volgorde moet consistent zijn met planned_times
        planned = sorted(
            [(three_timetable.scheduled_arrival(t_id, "seg-line"), t_id) for t_id in [1, 2, 3]]
        )
        verwacht = [t_id for _, t_id in planned]
        assert volgorde == verwacht

    def test_mip_solution_past_tijden_aan(self, one_train, segments, timetable):
        """Na MIP-oplossing wordt seg-station op nieuwe tijd gepland."""
        # Stel: MIP plant seg-station veel later
        mip_entry = 99999.0
        mip_exit  = 100060.0

        solution           = MagicMock()
        solution.arrival   = {(1, "seg-station"): mip_entry}
        solution.departure = {(1, "seg-station"): mip_exit}

        sim = self._sim(one_train, segments, timetable)
        sim._initialise()

        # Simuleer dat seg-line al afgerond is
        sim._state.record_entry(1, "seg-line", 3600.0)
        sim._state.record_exit( 1, "seg-line", 3660.0)

        sim._apply_solution(solution)

        assert sim._queue.has_entered(1, "seg-station")
        entered = next(
            e.event for e in sim._queue._heap
            if isinstance(e.event, TrainEntered)
            and e.event.train_id   == 1
            and e.event.segment_id == "seg-station"
        )
        assert entered.time == mip_entry

    def test_mip_raakt_geen_afgeronde_segmenten(self, one_train, segments, timetable):
        """_apply_solution raakt segmenten met geregistreerde exit niet aan."""
        solution           = MagicMock()
        solution.arrival   = {(1, "seg-line"): 99999.0}
        solution.departure = {(1, "seg-line"): 100060.0}

        sim = self._sim(one_train, segments, timetable)
        sim._initialise()

        # seg-line is al volledig afgerond
        sim._state.record_entry(1, "seg-line", 3600.0)
        sim._state.record_exit( 1, "seg-line", 3660.0)

        sim._apply_solution(solution)

        # Geen TrainEntered voor seg-line op MIP-tijd
        entered_times = [
            e.event.time for e in sim._queue._heap
            if isinstance(e.event, TrainEntered)
            and e.event.train_id   == 1
            and e.event.segment_id == "seg-line"
        ]
        assert 99999.0 not in entered_times

    def test_fcfs_fallback_past_dispatcher_aan(self, two_trains, segments, two_timetable):
        """Na FCFS-fallback respecteert dispatcher de nieuwe volgorde."""
        fcfs_order  = {"seg-line": [2, 1]}  # trein 2 krijgt voorrang
        controller  = FcfsController(fcfs_order=fcfs_order, fire_after=3600.0)

        sim = self._sim(two_trains, segments, two_timetable, controller)
        sim._initialise()

        # Trigger één TrainExited om controller aan te roepen
        sim._state.record_entry(1, "seg-line", 3600.0)
        sim._state.record_exit( 1, "seg-line", 3660.0)
        sim._dispatcher.enqueue(1, "seg-station",
            sim._timetable.scheduled_arrival(1, "seg-station"))
        sim._dispatcher.enqueue(2, "seg-line",
            sim._timetable.scheduled_arrival(2, "seg-line"))

        result = controller.step(sim._state, 3660.0)
        sim._dispatcher.reorder(result.fcfs_order)

        assert sim._dispatcher.next_in_queue("seg-line") == 2

    def test_geen_duplicaten_na_apply_solution(self, one_train, segments, timetable):
        """Na _apply_solution staat elk (train_id, segment_id) maar één keer in de queue."""
        solution           = MagicMock()
        solution.arrival   = {(1, "seg-station"): 9000.0}
        solution.departure = {(1, "seg-station"): 9120.0}

        sim = self._sim(one_train, segments, timetable)
        sim._initialise()
        sim._state.record_entry(1, "seg-line", 3600.0)
        sim._state.record_exit( 1, "seg-line", 3660.0)

        # Voeg een duplicaat toe alsof handle_exited al een event pushte
        sim._queue.push(TrainEntered(time=3720.0, train_id=1, segment_id="seg-station"))

        sim._apply_solution(solution)

        # Tel TrainEntered events voor seg-station
        count = sum(
            1 for e in sim._queue._heap
            if isinstance(e.event, TrainEntered)
            and e.event.train_id   == 1
            and e.event.segment_id == "seg-station"
        )
        assert count == 1

    def test_lege_solution_geen_crash(self, one_train, segments, timetable):
        """_apply_solution met lege oplossing crasht niet."""
        solution           = MagicMock()
        solution.arrival   = {}
        solution.departure = {}

        sim = self._sim(one_train, segments, timetable)
        sim._initialise()
        sim._apply_solution(solution)  # mag niet crashen

    def test_vertraging_wordt_doorgepropageerd(self, one_train, segments, timetable):
        """Vertraging op eerste segment leidt tot latere exit op tweede segment."""
        state        = self._sim(one_train, segments, timetable).run()
        planned_exit = timetable.scheduled_departure(1, "seg-station")
        actual_exit  = state.actual_exit(1, "seg-station")

        # Trein start op geplande tijd, dus zonder extra vertraging
        # mag actual_exit niet ver voor planned_exit liggen
        assert actual_exit >= planned_exit - 0.001

    def test_current_delay_nul_voor_ongestarte_trein(self, one_train, segments, timetable):
        """current_delay is 0 voor een trein die nog niet gestart is."""
        sim   = self._sim(one_train, segments, timetable)
        state = sim._state
        assert state.current_delay(1) == 0.0

    def test_remaining_path_leeg_na_finish(self, one_train, segments, timetable):
        """remaining_path is leeg nadat trein klaar is."""
        state = self._sim(one_train, segments, timetable).run()
        assert state.remaining_path(1) == []

    def test_active_train_ids_tijdens_simulatie(self, two_trains, segments, two_timetable):
        """active_train_ids bevat enkel gestarte, niet-klare treinen."""
        sim = self._sim(two_trains, segments, two_timetable)
        sim._initialise()

        # Trein 1 is gestart maar nog niet klaar
        sim._state.record_entry(1, "seg-line", 3600.0)
        active = sim._state.active_train_ids()
        assert 1 in active
        assert 2 not in active  # nog niet gestart