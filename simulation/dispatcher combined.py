from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Bewaakt segmentbezetting en beheert een wachtrij per segment.

    Volgorde-principe:
      Treinen die een bezet segment willen betreden komen in een
      waiting-list per segment. De wachtrij vult zich pas op het moment
      dat een trein fysiek klaar is om het volgende resource aan te
      vragen — de globale toekomstvolgorde leeft in het MIP-schedule.

      Prioriteit binnen de queue:
        1. mip_entry uit SystemState (indien beschikbaar)
        2. FIFO op insertion-order (fallback)

      Een trein die de queue niet aanvoert wordt geweigerd zelfs als het
      segment vrij is, zodat de hoogste-prioriteitswachter eerst gaat.

    Verantwoordelijkheden:
      - bezetting (_occupied)
      - per-segment waiting-list (_waiting)
      - C2-constraint via min_exit_time
    """

    def __init__(self, timetable, segments, trains) -> None:
        self._timetable = timetable
        self._trains = trains
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
          - segment bezet is, of
          - segment vrij is maar een hogere-prioriteits-wachter al in de
            queue zit voor dit segment.
        """
        waiters = self._waiting[segment_id]

        if self._occupied[segment_id] is not None:
            self._enqueue(train_id, segment_id)
            return False

        # Segment is vrij — mag déze trein voorgaan?
        if waiters and self._priority_winner(segment_id, state) != train_id:
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

    def next_waiter(self, segment_id: str, state=None) -> int | None:
        """
        Geeft de train_id van de hoogste-prioriteits-wachter, of None.

        Wordt door de simulator opgevraagd direct na een release om de
        wachtende trein te wekken via een nieuw event.
        """
        if not self._waiting[segment_id]:
            return None
        return self._priority_winner(segment_id, state)

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

    def _priority_winner(self, segment_id: str, state) -> int:
        """
        Kies de hoogste-prio waiter: laagste mip_entry, met FIFO als tiebreak
        en als fallback voor wachters zonder mip_entry.
        """
        waiters = self._waiting[segment_id]

        def key(idx_tid):
            idx, tid = idx_tid
            mip = state.mip_entry_for(tid, segment_id) if state is not None else None
            return (mip if mip is not None else float("inf"), idx)

        indexed = list(enumerate(waiters))
        winner_idx, winner_tid = min(indexed, key=key)
        return winner_tid

    # ==========================================================================
    # C2-constraint
    # ==========================================================================

    def min_exit_time(
        self,
        train_id: int,
        segment_id: str,
        entry_time: float,
        state,
    ) -> float:
        """
        Vroegste toegelaten exittijd.

        Voor dwell-segmenten:
        - gebruik MIP-exit indien beschikbaar
        - anders fallback op scheduled_exit

        Voor andere segmenten:
        - entry_time
        """
        train = self._trains[train_id]

        if not train.halts_at(segment_id):
            return entry_time

        # mip_exit = state.mip_exit_for(train_id, segment_id)

        # if mip_exit is not None:
        #     return mip_exit

        fallback = self._timetable.scheduled_exit(train_id, segment_id)

        # logger.debug(
        #     "fallback naar scheduled_exit voor dwell-segment train=%s seg=%s "
        #     "exit=%.1f",
        #     train_id, segment_id, fallback,
        # )

        return fallback
