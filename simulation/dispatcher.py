from __future__ import annotations

import logging

from domain import train

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Bewaakt segmentbezetting en berekent de C2-constraint.

    Volgorde wordt NIET meer bepaald door de dispatcher — dat gebeurt
    impliciet via de event-tijden in de EventQueue (FCFS op event-tijd).
    De MIP stuurt volgorde door TrainReadyToExit-events op de gewenste
    MIP-entry-tijden te plannen via _apply_solution.

    Verantwoordelijkheden:
      - Bijhouden welke trein welk segment bezet (_occupied)
      - Verwachte vrijkomsttijd bijhouden voor smart retry (_expected_release)
      - C2-constraint: min_exit_time voor WITHIN-STATION-DWELL segmenten
    """

    def __init__(self, timetable, segments, trains) -> None:
        self._timetable = timetable
        self._trains = trains
        self._occupied: dict[str, int | None] = {
            seg_id: None for seg_id in segments
        }
        self._expected_release: dict[str, float] = {}

    def request_entry(self, train_id: int, segment_id: str, current_time: float) -> bool:
        """True als segment vrij is."""
        return self._occupied[segment_id] is None

    def confirm_entry(self, train_id: int, segment_id: str) -> None:
        """Markeer segment als bezet door train_id."""
        self._occupied[segment_id] = train_id

    def release(self, train_id: int, segment_id: str) -> None:
        """Geef segment vrij en wis de verwachte vrijkomsttijd."""
        if self._occupied[segment_id] == train_id:
            self._occupied[segment_id] = None
            self._expected_release.pop(segment_id, None)
        else:
            logger.warning(
                "release mismatch: train=%s seg=%s occupied=%s",
                train_id, segment_id, self._occupied[segment_id],
            )

    def set_expected_release(self, segment_id: str, time: float) -> None:
        """
        Registreer wanneer segment_id naar verwachting vrijkomt.
        Gezet na elke confirm_entry met de berekende TrainReadyToExit-tijd.
        Geblokkeerde treinen gebruiken dit als smart retry-tijdstip.
        """
        self._expected_release[segment_id] = time

    def expected_release_time(self, segment_id: str) -> float | None:
        """Geeft de verwachte vrijkomsttijd, of None als onbekend."""
        return self._expected_release.get(segment_id)

    # def min_exit_time(self, train_id: int, segment_id: str, entry_time: float) -> float:
    #     """
    #     C2-constraint: vroegste toegelaten exittijd.

    #     Alleen actief voor WITHIN-STATION-DWELL (row.halts == True).
    #     Voor alle andere segmenten: entry_time.
    #     """
    #     # deze moet toegepast worden op de rescheduled mip
    #     row = self._timetable.get(train_id, segment_id)
    #     if not row.halts:
    #         return entry_time 
    #     return self._timetable.scheduled_departure(train_id, segment_id)

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
        - anders fallback op scheduled_departure

        Voor andere segmenten:
        - entry_time
        """

        train = self._trains[train_id]

        if not train.halts_at(segment_id):
            return entry_time

        mip_exit = state.mip_exit_for(train_id, segment_id)

        if mip_exit is not None:
            return mip_exit

        fallback = self._timetable.scheduled_exit(
            train_id,
            segment_id,
        )

        logger.debug(
            "!! Als dit na de eerste reschedule is, fallback naar scheduled_departure voor dwell-segment: in simulatie, ervoor is dit normaal "
            "train=%s seg=%s fallback_exit=%.1f",
            train_id,
            segment_id,
            fallback,
        )

        return fallback