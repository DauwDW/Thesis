

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class _PrioritizedItem: #Wrapper om heapq te kunnen vergelijken, heapq --> kleinste element altijd bovenaan
    time: float
    priority: int
    event: object = field(compare=False)


@dataclass(slots=True, order= False)
class TrainEntered:
    time: float
    train_id: int
    segment_id: str
    cancelled: bool = False


@dataclass(slots=True, order= False)
class TrainReadyToExit:
    time: float
    train_id: int
    segment_id: str
    cancelled: bool = False


class EventQueue:

    def __init__(self) -> None:
        self._heap: list[_PrioritizedItem] = []
        self._counter = 0

    def push(self, event) -> None:
        heapq.heappush(
            self._heap,
            _PrioritizedItem(
                time=event.time,
                priority=self._counter,
                event=event,
            ),
        )
        self._counter += 1

    def pop(self):

        while self._heap:
            item = heapq.heappop(self._heap)

            # skip cancelled events
            if item.event.cancelled:       
                continue
            else:
                return item.event

        raise IndexError("pop from empty EventQueue")

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel_train_entered(self, train_id: int, segment_id: str) -> None:
        """
        Markeer TrainEntered-events voor een specifiek segment als cancelled.
        """
        for item in self._heap:
            ev = item.event
            if (
                isinstance(ev, TrainEntered)
                and ev.train_id == train_id
                and ev.segment_id == segment_id
            ):
                ev.cancelled = True

    def cancel_all_train_entered(self, train_id: int) -> None:
        """
        Markeer alle TrainEntered-events voor train_id als cancelled,
        ongeacht segment. Gebruik dit bij reschedule van een niet-gestarte
        trein zodat eventuele verouderde events voor het vorige (gepland of
        gekozen) segment actief worden opgeruimd.
        """
        for item in self._heap:
            ev = item.event
            if isinstance(ev, TrainEntered) and ev.train_id == train_id:
                ev.cancelled = True

    def cancel_ready_to_exit(self, train_id: int, segment_id: str) -> None:
        """
        Markeer TrainReadyToExit-events als cancelled.
        """
        for item in self._heap:
            ev = item.event

            if (
                isinstance(ev, TrainReadyToExit)
                and ev.train_id == train_id
                and ev.segment_id == segment_id
            ):
                ev.cancelled = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_entered(self, train_id: int, segment_id: str) -> bool:
        return any(
            isinstance(item.event, TrainEntered)
            and not item.event.cancelled
            and item.event.train_id == train_id
            and item.event.segment_id == segment_id
            for item in self._heap
        )

    def has_ready_to_exit(self, train_id: int, segment_id: str) -> bool:
        return any(
            isinstance(item.event, TrainReadyToExit)
            and not item.event.cancelled
            and item.event.train_id == train_id
            and item.event.segment_id == segment_id
            for item in self._heap
        )

    def __bool__(self) -> bool:
        return any(
            not getattr(item.event, "cancelled", False)
            for item in self._heap
        )