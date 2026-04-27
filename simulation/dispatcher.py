# simulation/dispatcher.py
#
# Dispatcher — bewaakt segmentbezetting en bepaalt toegangsvolgorde.
#
# Kernprincipe:
#   VOLGORDE  = geplande/MIP tijden — wie het vroegst gepland was, gaat eerst
#   FEASIBLE  = werkelijke bezetting — segment moet fysiek vrij zijn
#
# Een trein krijgt toegang als:
#   1. Het segment fysiek vrij is (werkelijke bezetting)
#   2. De trein de vroegste geplande tijd heeft van alle wachtende treinen
#      (of de hoogste MIP-prioriteit als die beschikbaar is)
#
# Na een MIP-oplossing worden de geplande tijden in de wachtrij overschreven
# door de MIP-tijden via reorder() — de volgorde past zich automatisch aan.
#
# Interactie met simulator.py:
#   enqueue(train_id, segment_id, planned_time)
#                   — trein aanmelden met zijn geplande entrytijd
#   request_entry(train_id, segment_id, current_time) -> bool
#                   — mag de trein het segment betreden?
#   confirm_entry(train_id, segment_id)
#                   — bevestig entry, markeer segment als bezet
#   release(train_id, segment_id)
#                   — segment vrijgeven
#   reorder(order)  — MIP-tijden opslaan, wachtrij hersorteren
#   min_exit_time(train_id, segment_id, entry_time) -> float
#                   — vroegste toegelaten exittijd (C2 constraint)
#   next_in_queue(segment_id) -> int | None
#                   — wie staat eerste in de wachtrij?

from __future__ import annotations

import logging
from dataclasses import dataclass

from domain import Timetable
from domain.segment import SegmentType

logger = logging.getLogger(__name__)


# =============================================================================
# Wachtrij-entry
# =============================================================================

@dataclass(order=True)
class _QueueEntry:
    """
    Wachtrij-entry per segment.

    Gesorteerd op planned_time — wie het vroegst gepland is, staat eerst.
    Bij gelijke planned_time wint de laagste train_id (deterministisch).

    planned_time wordt overschreven door MIP-tijden via reorder().
    """
    planned_time: float
    train_id:     int


# =============================================================================
# Dispatcher
# =============================================================================

class Dispatcher:
    """
    Bewaakt segmentbezetting en bepaalt toegangsvolgorde.

    Parameters
    ----------
    timetable : Timetable          — geplande tijden (voor min_exit_time)
    segments  : dict[str, Segment] — alle segmentobjecten

    Interne datastructuur
    ---------------------
    _occupied : dict[str, int | None]
        Huidig bezettende trein per segment. None = vrij.

    _queue : dict[str, list[_QueueEntry]]
        Gesorteerde wachtrij per segment op planned_time.
        Treinen worden toegevoegd via enqueue() en verwijderd via confirm_entry().
    """

    def __init__(
        self,
        timetable: Timetable,
        segments:  dict,
    ) -> None:
        self._timetable = timetable
        self._segments  = segments

        self._occupied: dict[str, int | None]        = {seg_id: None for seg_id in segments}
        self._queue:    dict[str, list[_QueueEntry]] = {seg_id: []   for seg_id in segments}

    # ==========================================================================
    # Wachtrij beheer
    # ==========================================================================

    def enqueue(self, train_id: int, segment_id: str, planned_time: float) -> None:
        """
        Meldt een trein aan in de wachtrij voor een segment.

        De trein wordt ingevoegd op volgorde van planned_time.
        Wordt aangeroepen door simulator.py bij TrainExited van het vorige segment.

        Parameters
        ----------
        train_id     : int   — treinnummer
        segment_id   : str   — segment waarvoor de trein zich aanmeldt
        planned_time : float — geplande of MIP entry-tijd (seconden)
        """
        # Vermijd dubbele aanmelding
        if any(e.train_id == train_id for e in self._queue[segment_id]):
            return

        entry = _QueueEntry(planned_time=planned_time, train_id=train_id)
        self._queue[segment_id].append(entry)
        self._queue[segment_id].sort()  # gesorteerd op planned_time

        logger.debug(
            f"Trein {train_id} aangemeld voor '{segment_id}' "
            f"op geplande tijd {planned_time:.0f}s"
        )


    def queue_time(self, train_id: int, segment_id: str) -> float:
        """
        Geeft de tijd terug waarop een trein zich aanmeldde in de wachtrij.
        Wordt gebruikt door compute_fcfs_order in controller.py.

        Parameters
        ----------
        train_id   : int — treinnummer
        segment_id : str — segment

        Returns
        -------
        float — tijdstip waarop trein zich aanmeldde in de wachtrij
        """
        for entry in self._queue[segment_id]:
            if entry.train_id == train_id:
                return entry.planned_time
        raise KeyError(f"Trein {train_id} staat niet in wachtrij voor '{segment_id}'")

    # ==========================================================================
    # Toegangsbeslissing
    # ==========================================================================

    def request_entry(self, train_id: int, segment_id: str, current_time: float) -> bool:
        """
        Beslist of een trein een segment mag betreden.

        Toegangslogica:
          1. Segment moet fysiek vrij zijn (werkelijke bezetting)
          2. Trein moet eerste staan in de wachtrij (geplande/MIP volgorde)

        Parameters
        ----------
        train_id     : int   — treinnummer
        segment_id   : str   — segment dat de trein wil betreden
        current_time : float — huidige simulatietijd (werkelijke tijd)

        Returns
        -------
        bool — True als de trein het segment mag betreden
        """
        # 1. Segment moet fysiek vrij zijn
        if self._occupied[segment_id] is not None:
            return False

        # 2. Trein moet eerste in de wachtrij staan
        queue = self._queue[segment_id]
        if not queue:
            return False

        return queue[0].train_id == train_id

    def confirm_entry(self, train_id: int, segment_id: str) -> None:
        """
        Bevestigt dat een trein het segment betreedt.

        Markeert segment als bezet en verwijdert trein uit wachtrij.
        Wordt aangeroepen door simulator.py na een succesvolle request_entry().

        Parameters
        ----------
        train_id   : int — treinnummer
        segment_id : str — segment dat de trein betreedt
        """
        self._occupied[segment_id] = train_id

        queue = self._queue[segment_id]
        if queue and queue[0].train_id == train_id:
            queue.pop(0)

        logger.debug(f"Trein {train_id} betreedt '{segment_id}'")

    def release(self, train_id: int, segment_id: str) -> None:
        """
        Geeft een segment vrij.

        Wordt aangeroepen door simulator.py bij TrainExited,
        vóór record_exit() in SystemState.

        Parameters
        ----------
        train_id   : int — treinnummer
        segment_id : str — segment dat de trein verlaat
        """
        if self._occupied[segment_id] == train_id:
            self._occupied[segment_id] = None
            logger.debug(f"Trein {train_id} verlaat '{segment_id}'")
        else:
            logger.warning(
                f"release() aangeroepen voor trein {train_id} op '{segment_id}' "
                f"maar segment bezet door {self._occupied[segment_id]}"
            )

    # ==========================================================================
    # Volgorde aanpassen — na MIP-oplossing of FCFS-fallback
    # ==========================================================================

    def reorder(self, order: dict[str, list[int]]) -> None:
        """
        Herschrijft de planned_time in de wachtrij op basis van een nieuwe volgorde.

        Na een MIP-oplossing of FCFS-fallback geeft de controller een nieuwe
        volgorde per segment. De dispatcher kent geen echte MIP-tijden —
        hij krijgt enkel de gewenste volgorde als lijst van train_ids.

        Strategie: wijs fictieve tijden toe (0, 1, 2, ...) op basis van
        de gewenste volgorde. De absolute waarden doen er niet toe —
        enkel de onderlinge volgorde.

        Parameters
        ----------
        order : dict[str, list[int]]
            Per segment een geordende lijst van train_ids (hoogste prioriteit eerst).
            Output van _apply_solution() of compute_fcfs_order().
        """
        for seg_id, train_ids in order.items():
            queue = self._queue[seg_id]

            # Bouw mapping train_id → nieuwe prioriteit (lagere waarde = hogere prioriteit)
            priority = {train_id: i for i, train_id in enumerate(train_ids)}

            # Herorden bestaande wachtrij-entries
            for entry in queue:
                if entry.train_id in priority:
                    entry.planned_time = float(priority[entry.train_id])

            queue.sort()
            logger.debug(f"Volgorde voor '{seg_id}' herordend: {train_ids}")

    # ==========================================================================
    # Diagnostiek
    # ==========================================================================

    def next_in_queue(self, segment_id: str) -> int | None:
        """
        Geeft de train_id terug van de eerste trein in de wachtrij.
        None als de wachtrij leeg is.
        """
        queue = self._queue[segment_id]
        return queue[0].train_id if queue else None

    # ==========================================================================
    # Minimale exittijd — C2 constraint
    # ==========================================================================

    def min_exit_time(
        self,
        train_id:   int,
        segment_id: str,
        entry_time: float,
    ) -> float:
        """
        Berekent de vroegste toegelaten exittijd voor een trein op een segment.

        Voor stationssegmenten:
            min_exit = max(entry_time + dwell_time, planned_exit)
            → C2 constraint: trein mag niet vroeger vertrekken dan gepland

        Voor lijnsegmenten:
            min_exit = entry_time
            → geen ondergrens buiten de gesamplede rijtijd

        Parameters
        ----------
        train_id   : int   — treinnummer
        segment_id : str   — segment
        entry_time : float — werkelijke entrytijd in seconden

        Returns
        -------
        float — vroegste toegelaten exittijd in seconden
        """
        segment = self._segments.get(segment_id)
        if segment is None or segment.seg_type != SegmentType.STATION:
            return entry_time

        try:
            dwell_time   = self._timetable.dwell_time(train_id, segment_id)
            planned_exit = self._timetable.scheduled_departure(train_id, segment_id)
            return max(entry_time + dwell_time, planned_exit)
        except (KeyError, ValueError):
            logger.debug(
                f"Trein {train_id}: geen dwell/planned_exit voor '{segment_id}' "
                f"— geen ondergrens"
            )
            return entry_time

    def __repr__(self) -> str:
        n_occupied = sum(1 for v in self._occupied.values() if v is not None)
        n_waiting  = sum(len(q) for q in self._queue.values())
        return (
            f"Dispatcher("
            f"bezet={n_occupied} segmenten, "
            f"wachtend={n_waiting} treinen)"
        )