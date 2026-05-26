# dispatcher FULL FSFS

from __future__ import annotations

import logging

from config.settings import DISPATCHER_PRIORITY_TTL

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

      Prioriteit binnen de queue:
        1. mip_entry uit SystemState (indien beschikbaar)
        2. FIFO op insertion-order (fallback)

      Een trein die de queue niet aanvoert wordt geweigerd zelfs als het
      segment vrij is, zodat de hoogste-prioriteitswachter eerst gaat.

    Strict-order modus (optioneel):
      Wanneer strict_order=True wordt bovenop FCFS-met-prioriteit een
      harde MIP-volgorde afgedwongen: een trein mag een segment pas
      betreden als alle andere niet-afgeronde treinen met een lagere
      mip_entry op dit segment ofwel al binnen zijn, ofwel dit segment
      in hun pad al voorbij zijn. Zie _strict_order_allows.

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

        # Tijdstip van de laatste toegepaste MIP-oplossing.
        # float("-inf") → nooit gerescheduled → prioriteit is direct verouderd.
        self._last_reschedule_time: float = float("-inf")



    def _has_passed(self, train_id: int, segment_id: str, state) -> bool:
        """
        True als train_id segment_id al voorbij is in zijn pad — d.w.z.
        op een later segment binnen is gekomen.

        Defensief tegen oude MIP-entries die nog in state staan voor
        segmenten die de trein in werkelijkheid al gepasseerd is, of
        voor segmenten die niet in zijn pad zitten.

        Bij retracking: segment_id kan een gekozen platform zijn dat niet
        letterlijk in train.path voorkomt. We vertalen eerst naar het geplande
        segment voor de path-lookup, en controleren dan op gekozen segmenten.
        """
        train = self._trains[train_id]
        path  = train.path

        # Vertaal naar gepland segment voor path-index lookup
        planned_seg = state.get_planned_seg_for(train_id, segment_id)
        try:
            idx_seg = path.index(planned_seg)
        except ValueError:
            # segment niet in pad van deze trein — kan dus nooit blokkeren
            return True

        for later_planned in path[idx_seg + 1:]:
            later_actual = state.get_chosen_seg(train_id, later_planned)
            try:
                state.actual_entry(train_id, later_actual)
                return True
            except KeyError:
                continue

        return False

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
          - in strict-order modus: een trein met lagere mip_entry op dit
            segment is nog niet binnen en nog niet voorbij, of
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

    def notify_reschedule(self, current_time: float) -> None:
        """
        Registreer het tijdstip waarop een MIP-oplossing werd toegepast.

        Zolang current_time - _last_reschedule_time < _PRIORITY_TTL gebruikt
        de dispatcher mip_entry als prioriteit (FSFS op MIP-plan).
        Daarna valt hij terug op scheduled_entry (timetable-volgorde) zodat
        verouderde MIP-prioriteiten geen deadlocks kunnen veroorzaken.
        """
        self._last_reschedule_time = current_time

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

        # Bepaal of de MIP-oplossing nog vers genoeg is om te gebruiken.
        # Als current_time onbekend is, val terug op het oude gedrag (gebruik mip_entry).
        use_mip = (
            current_time is None
            or current_time - self._last_reschedule_time < DISPATCHER_PRIORITY_TTL
        )

        if use_mip:
            # FSFS op MIP-plan: laagste mip_entry gaat eerst.
            def key(idx_tid):
                idx, tid = idx_tid
                mip = state.mip_entry_for(tid, segment_id) if state is not None else None
                return (mip if mip is not None else float("inf"), idx)
        else:
            # Fallback: FSFS op timetable (scheduled_entry).
            # Deterministisch en altijd cirkel-vrij — voorkomt deadlocks door
            # verouderde MIP-prioriteiten.
            logger.debug(
                "priority_winner: MIP-prioriteit verouderd (%.0fs geleden, TTL=%.0fs) "
                "voor seg=%s — gebruik scheduled_entry",
                current_time - self._last_reschedule_time,
                DISPATCHER_PRIORITY_TTL,
                segment_id,
            )

            def key(idx_tid):
                idx, tid = idx_tid
                try:
                    # segment_id kan een gekozen platform zijn na retracking;
                    # de timetable is geïndexeerd op geplande segmenten.
                    planned = (
                        state.get_planned_seg_for(tid, segment_id)
                        if state is not None else segment_id
                    )
                    sched = self._timetable.scheduled_entry(tid, planned)
                except (KeyError, AttributeError):
                    sched = None
                return (sched if sched is not None else float("inf"), idx)

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

        # logger.debug(
        #     "fallback naar scheduled_exit voor dwell-segment train=%s seg=%s "
        #     "exit=%.1f",
        #     train_id, segment_id, fallback,
        # )
        fallback = self._timetable.scheduled_exit(train_id, segment_id)
        return fallback
