# simulation/event_queue.py
#
# EventQueue — prioriteitswachtrij van simulatie-events.
#
# Twee event-types:
#   TrainEntered : trein wil een segment betreden op de geplande/MIP tijd
#   TrainExited  : trein verlaat een segment op de gesamplede werkelijke tijd
#
# De queue bevat uitsluitend GEPLANDE tijden — de werkelijke tijden worden
# bepaald door de Dispatcher (feasibility) en de reality module (sampling).
#
# Verantwoordelijkheden van event_queue.py:
#   - Events toevoegen (push)
#   - Vroegste event opvragen (pop / peek)
#   - Events annuleren voor een (train_id, segment_id) (cancel)
#   - Duplicaatcheck voor TrainEntered (has_entered)
#
# Verantwoordelijkheden van dispatcher.py:
#   - Volgorde bepalen op basis van geplande/MIP tijden
#   - Feasibility checken op basis van werkelijke bezetting
#   - Wachtrij beheren voor treinen die nog niet kunnen betreden
#
# Verantwoordelijkheden van simulator.py:
#   - Event-loop draaien
#   - MIP-oplossing verwerken (_apply_solution)
#   - Reality module aanroepen voor werkelijke rijtijden
#   - SystemState bijwerken via record_entry / record_exit

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# Event types
# =============================================================================

@dataclass(order=False)
class TrainEntered:
    """
    Trein wil een segment betreden.

    time = geplande of MIP entry-tijd — NIET de werkelijke tijd.

    Verwerking door simulator.py:
      → dispatcher.request_entry(train_id, segment_id, planned_time)
         - Toegang verleend: dispatcher.confirm_entry()
                             state.record_entry(werkelijke tijd)
                             TrainExited plannen op gesamplede duur
         - Toegang geweigerd: dispatcher beheert wachtrij intern,
                              simulator plant geen extra polling-events
    """
    time:       float   # geplande entry-tijd
    train_id:   int
    segment_id: str
    kind: Literal["entered"] = field(default="entered", init=False, repr=False)


@dataclass(order=False)
class TrainExited:
    """
    Trein verlaat een segment.

    time = werkelijke exit-tijd (entry + gesamplede duur).

    Verwerking door simulator.py:
      → dispatcher.release()
        state.record_exit()
        TrainEntered plannen voor volgend segment op geplande tijd
        controller.step()
    """
    time:       float   # werkelijke exit-tijd
    train_id:   int
    segment_id: str
    kind: Literal["exited"] = field(default="exited", init=False, repr=False)


Event = TrainEntered | TrainExited


# =============================================================================
# Heap entry
# =============================================================================

@dataclass(order=True)
class _HeapEntry:
    """
    Interne wrapper zodat de heap nooit Event-objecten direct vergelijkt.

    Sorteervolgorde: (time, tie_breaker)
    tie_breaker = insertievolgorde — deterministisch bij gelijke tijd.
    """
    time:        float
    tie_breaker: int
    event:       Event = field(compare=False)


# =============================================================================
# EventQueue
# =============================================================================

class EventQueue:
    """
    Prioriteitswachtrij van simulatie-events, gesorteerd op tijd.

    De queue is een domme datastructuur — geen volgorde-logica,
    geen feasibility-checks. Dat is de verantwoordelijkheid van
    de Dispatcher.

    Gebruik
    -------
    queue = EventQueue()
    queue.push(TrainEntered(time=3600.0, train_id=123, segment_id="25:A-B"))
    event = queue.pop()
    """

    def __init__(self) -> None:
        self._heap:    list[_HeapEntry] = []
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Basis operaties
    # ------------------------------------------------------------------
    def push(self, event: Event) -> None:
        """Voegt een event toe aan de queue."""
        if event.time == 0.0 and hasattr(event, 'train_id') and event.train_id == 1995:
            import traceback
            print(f"[DEBUG] TrainEntered(0.0, 1995) gepusht via:")
            traceback.print_stack()
        heapq.heappush(self._heap, _HeapEntry(
            time        = event.time,
            tie_breaker = self._counter,
            event       = event,
        ))
        self._counter += 1
    # def push(self, event: Event) -> None:
    #     """Voegt een event toe aan de queue."""
    #     heapq.heappush(self._heap, _HeapEntry(
    #         time        = event.time,
    #         tie_breaker = self._counter,
    #         event       = event,
    #     ))
    #     self._counter += 1

    def pop(self) -> Event:
        """Verwijdert en geeft het vroegste event terug."""
        if not self._heap:
            raise IndexError("EventQueue is leeg")
        return heapq.heappop(self._heap).event

    def peek(self) -> Event:
        """Geeft het vroegste event terug zonder het te verwijderen."""
        if not self._heap:
            raise IndexError("EventQueue is leeg")
        return self._heap[0].event

    # ------------------------------------------------------------------
    # Annulering
    # ------------------------------------------------------------------

    def cancel(self, train_id: int, segment_id: str) -> int:
        """
        Verwijdert alle events voor (train_id, segment_id).

        Gebruikt door _apply_solution() in simulator.py om bestaande
        events te verwijderen voordat nieuwe MIP-tijden gepusht worden.

        Returns
        -------
        int — aantal verwijderde events
        """
        before = len(self._heap)
        self._heap = [
            e for e in self._heap
            if not (
                e.event.train_id   == train_id
                and e.event.segment_id == segment_id
            )
        ]
        heapq.heapify(self._heap)
        return before - len(self._heap)

    def has_entered(self, train_id: int, segment_id: str) -> bool:
        """
        True als er al een TrainEntered gepland staat voor (train_id, segment_id).

        Gebruikt door _handle_exited() om duplicaten te vermijden na
        _apply_solution().
        """
        return any(
            isinstance(e.event, TrainEntered)
            and e.event.train_id   == train_id
            and e.event.segment_id == segment_id
            for e in self._heap
        )

    # ------------------------------------------------------------------
    # Diagnostiek
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __repr__(self) -> str:
        return f"EventQueue({len(self._heap)} events)"