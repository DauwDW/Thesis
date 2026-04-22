# simulation/event_queue.py
#
# EventQueue — prioriteitswachtrij van simulatie-events.
#
# Twee event-types:
#   TrainEntered : trein betreedt een segment (triggert record_entry in SystemState)
#   TrainExited  : trein verlaat een segment  (triggert record_exit  in SystemState)
#
# De queue is een heap gesorteerd op (time, tie_breaker) zodat events met
# gelijke tijd deterministisch verwerkt worden.
#
# Verantwoordelijkheden van event_queue.py:
#   - Events aanmaken en toevoegen (push)
#   - Volgend event opvragen (pop)
#   - FCFS-herordening toepassen na een controller-beslissing (apply_fcfs)
#
# Verantwoordelijkheden van simulator.py:
#   - MIP-oplossing verwerken en events herbouwen (apply_solution)
#   - record_entry / record_exit aanroepen op SystemState
#   - De event-loop draaien

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
    Trein betreedt een segment.

    Verwerking door simulator.py:
      state.record_entry(train_id, segment_id, time)
      → schedule TrainExited voor hetzelfde (train_id, segment_id)
    """
    time:       float
    train_id:   int
    segment_id: str
    kind: Literal["entered"] = field(default="entered", init=False, repr=False)


@dataclass(order=False)
class TrainExited:
    """
    Trein verlaat een segment.

    Verwerking door simulator.py:
      state.record_exit(train_id, segment_id, time)
      → schedule TrainEntered voor het volgende segment (als dat bestaat)
    """
    time:       float
    train_id:   int
    segment_id: str
    kind: Literal["exited"] = field(default="exited", init=False, repr=False)


Event = TrainEntered | TrainExited


# =============================================================================
# Heap entry — wrapper voor stabiele sortering
# =============================================================================

@dataclass(order=True)
class _HeapEntry:
    """
    Interne wrapper zodat de heap nooit Event-objecten direct vergelijkt.

    Sorteervolgorde: (time, tie_breaker)
    tie_breaker is een monotoon oplopende teller die insertion-volgorde
    bewaart bij gelijke tijden — dit maakt de queue deterministisch.
    """
    time:         float
    tie_breaker:  int
    event:        Event = field(compare=False)


# =============================================================================
# EventQueue
# =============================================================================

class EventQueue:
    """
    Prioriteitswachtrij van simulatie-events, gesorteerd op tijd.

    Intern een min-heap van _HeapEntry objecten. Events met gelijke tijd
    worden verwerkt in de volgorde waarin ze werden toegevoegd (FIFO),
    tenzij apply_fcfs de volgorde aanpast.

    Gebruik
    -------
    queue = EventQueue()
    queue.push(TrainEntered(time=3600.0, train_id=123, segment_id="25:A-B"))
    event = queue.pop()   # geeft het vroegste event terug
    """

    def __init__(self) -> None:
        self._heap:        list[_HeapEntry] = []
        self._counter:     int = 0            # tie-breaker teller

    # ------------------------------------------------------------------
    # Basis operaties
    # ------------------------------------------------------------------

    def push(self, event: Event) -> None:
        """
        Voegt een event toe aan de queue.

        Parameters
        ----------
        event : TrainEntered | TrainExited
        """
        entry = _HeapEntry(
            time        = event.time,
            tie_breaker = self._counter,
            event       = event,
        )
        heapq.heappush(self._heap, entry)
        self._counter += 1

    def pop(self) -> Event:
        """
        Verwijdert en geeft het vroegste event terug.

        Raises
        ------
        IndexError als de queue leeg is
        """
        if not self._heap:
            raise IndexError("EventQueue is leeg")
        return heapq.heappop(self._heap).event

    def peek(self) -> Event:
        """
        Geeft het vroegste event terug zonder het te verwijderen.

        Raises
        ------
        IndexError als de queue leeg is
        """
        if not self._heap:
            raise IndexError("EventQueue is leeg")
        return self._heap[0].event

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    # ------------------------------------------------------------------
    # FCFS herordening — aangeroepen door simulator.py
    # ------------------------------------------------------------------

    def apply_fcfs(self, fcfs_order: dict[str, list[int]]) -> None:
        """
        Past de FCFS-volgorde toe op TrainEntered-events in de queue.

        Voor elk segment in fcfs_order worden de geplande TrainEntered-events
        voor dat segment hergesorteerd zodat de trein met de hoogste FCFS-prioriteit
        als eerste de queue verlaat. De tijden van andere events (TrainExited)
        worden niet aangeraakt.

        Strategie: herverdeel de bestaande tijden van de betrokken TrainEntered-events
        over de FCFS-volgorde. Zo blijft de globale timing intact en wordt enkel
        de onderlinge volgorde bij gelijke (of bijna gelijke) tijden gecorrigeerd.

        Parameters
        ----------
        fcfs_order : dict[str, list[int]]
            Per segment een geordende lijst van train_ids (hoogste prioriteit eerst).
            Output van compute_fcfs_order() in controller/controller.py.
        """
        if not fcfs_order:
            return

        # Verzamel alle relevante TrainEntered entries per segment
        entered_per_segment: dict[str, list[_HeapEntry]] = {
            seg_id: [] for seg_id in fcfs_order
        }
        remaining: list[_HeapEntry] = []

        for entry in self._heap:
            if (
                isinstance(entry.event, TrainEntered)
                and entry.event.segment_id in fcfs_order
            ):
                entered_per_segment[entry.event.segment_id].append(entry)
            else:
                remaining.append(entry)

        # Herorden per segment en voeg terug toe
        new_entries: list[_HeapEntry] = list(remaining)

        for seg_id, ordered_train_ids in fcfs_order.items():
            entries = entered_per_segment[seg_id]
            if not entries:
                continue

            # Bestaande tijden gesorteerd — herverdeel over FCFS-volgorde
            existing_times = sorted(e.time for e in entries)

            # Bouw een mapping train_id → entry voor snelle lookup
            entry_by_train: dict[int, _HeapEntry] = {
                e.event.train_id: e for e in entries
            }

            for time, train_id in zip(existing_times, ordered_train_ids):
                if train_id not in entry_by_train:
                    continue  # trein staat niet in de queue voor dit segment
                old_entry  = entry_by_train[train_id]
                new_entry  = _HeapEntry(
                    time        = time,
                    tie_breaker = self._counter,
                    event       = TrainEntered(
                        time       = time,
                        train_id   = old_entry.event.train_id,
                        segment_id = seg_id,
                    ),
                )
                self._counter += 1
                new_entries.append(new_entry)

            # Treinen in fcfs_order die niet (meer) in de queue staan worden genegeerd

        heapq.heapify(new_entries)
        self._heap = new_entries

    # ------------------------------------------------------------------
    # Diagnostiek
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"EventQueue({len(self._heap)} events)"