from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
_VALID_QUEUE_MODES = ("fsfs", "fcfs")

class Dispatcher:
    """
    Bewaakt segmentbezetting en beheert een wachtrij per segment.

    Volgorde-principe:
      Treinen die een bezet segment willen betreden komen in een
      waiting-list per segment. De wachtrij vult zich pas op het moment
      dat een trein fysiek klaar is om het volgende resource aan te
      vragen — de globale toekomstvolgorde leeft in het MIP-schedule.

    Queue-modi (queue_mode):
      "fsfs"  (default) — First-Scheduled-First-Served:
        De waiting-list wordt gesorteerd op mip_entry (laagste = hoogste
        prioriteit). MIP-prioriteit is altijd actief — geen TTL-verval.

      "fcfs"  — First-Come-First-Served:
        Pure insertion-order; geen MIP-prioriteit.

    Verantwoordelijkheden:
      - bezetting (_occupied)
      - per-segment waiting-list (_waiting)
      - C2-constraint via min_exit_time
    """

    def __init__(self, timetable, segments, trains, queue_mode: str = "fsfs") -> None:
        if queue_mode not in _VALID_QUEUE_MODES:
            raise ValueError(
                f"Unknown queue_mode: '{queue_mode}'. "
                f"Use one of {_VALID_QUEUE_MODES}."
            )
        self._timetable = timetable
        self._trains = trains
        self._queue_mode = queue_mode

        self._occupied: dict[str, int | None] = {
            seg_id: None for seg_id in segments
        }
        self._waiting: dict[str, list[int]] = {
            seg_id: [] for seg_id in segments
        }


    # ==========================================================================
    # Resource requests
    # ==========================================================================

    def request_entry(
        self,
        train_id: int,
        segment_id: str,
        current_time: float,
        state=None,
    ) -> bool:
        """
        True als deze trein nu het segment mag betreden.

        Faalt (en zet de trein in de queue) als:
          - het segment bezet is, of
          - een andere trein in de waiting-list een hogere prioriteit heeft
            (lagere mip_entry in fsfs-modus, of eerder aangekomen in fcfs-modus).
        """
        waiters = self._waiting[segment_id]

        if self._occupied[segment_id] is not None:
            self._enqueue(train_id, segment_id)
            return False
        if self._queue_mode == "fsfs":
            if waiters and self._priority_winner(segment_id, state, current_time) != train_id:
                self._enqueue(train_id, segment_id)
                return False

        # Toegang verleend — uit queue halen indien aanwezig
        if train_id in waiters:
            waiters.remove(train_id)
        return True

    def confirm_entry(self, train_id: int, segment_id: str) -> None:
        """Markeer segment als bezet door train_id."""
        self._occupied[segment_id] = train_id

    def release(self, train_id: int, segment_id: str) -> None:
        """Geef segment vrij."""
        if self._occupied[segment_id] == train_id:
            self._occupied[segment_id] = None
        else:
            logger.warning(
                "release mismatch: train=%s seg=%s occupied=%s",
                train_id, segment_id, self._occupied[segment_id],
            )

    # ==========================================================================
    # Queue management
    # ==========================================================================

    def next_waiter(self, segment_id: str, state=None, current_time: float | None = None) -> int | None:
        if not self._waiting[segment_id]:
            return None
        if self._queue_mode == "fcfs":
            return self._waiting[segment_id][0]
        return self._priority_winner(segment_id, state, current_time)

    def remove_from_queues(self, train_id: int) -> None:
        """
        Verwijder een trein uit alle waiting-lists.

        Aangeroepen door _apply_solution voor elke trein wiens event
        herplanned wordt — zo blijven alleen treinen in queues staan die
        op dit moment fysiek klaar zijn om het segment te betreden.
        """
        for waiters in self._waiting.values():
            if train_id in waiters:
                waiters.remove(train_id)

    # ==========================================================================
    # Intern
    # ==========================================================================

    def _enqueue(self, train_id: int, segment_id: str) -> None:
        waiters = self._waiting[segment_id]
        if train_id not in waiters:
            waiters.append(train_id)

    def _priority_winner(
        self,
        segment_id: str,
        state,
        current_time: float | None = None,
    ) -> int:
        waiters = self._waiting[segment_id]

        def _priority(tid: int) -> float:
            val = state.mip_entry_for(tid, segment_id) if state is not None else None
            return val if val is not None else float("inf")

        _, winner = min(enumerate(waiters), key=lambda item: (_priority(item[1]), item[0])) # (index, train_id), item[0] voor tiebrake
        return winner

    # ==========================================================================
    # C2-constraint
    # ==========================================================================

    def min_exit_time(
        self,
        train_id: int,
        segment_id: str,
        entry_time: float,
        state,
        planned_segment_id: str | None = None,
    ) -> float:
        """
        Vroegste toegelaten exittijd.

        Voor dwell-segmenten:
        - fallback op scheduled_exit van het geplande segment

        Voor andere segmenten (between-station of passing):
        - entry_time

        planned_segment_id:
            Bij retracking is segment_id het gekozen platform. halt_indicators
            en de timetable zijn geïndexeerd op het geplande segment. Geef
            planned_segment_id mee zodat de halts_at- en scheduled_exit-lookup
            correct werkt voor geretrackte stops.
        """
        train   = self._trains[train_id]
        plan_id = planned_segment_id if planned_segment_id is not None else segment_id

        if not train.halts_at(plan_id):
            return entry_time

        scheduled_ex = self._timetable.scheduled_exit(train_id, plan_id)
        # mip_ex = state.mip_exit_for(train_id, segment_id) if state is not None else None
        # if mip_ex is not None:
        #     return max(entry_time, scheduled_ex, mip_ex)
        return max(entry_time, scheduled_ex)
