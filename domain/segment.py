# Segment en SegmentType domeinklassen.
#
# SegmentType: puur infrastructureel onderscheid
#   STATION         — stationsinfrastructuur (perron/spoor)
#   BETWEEN_STATION — lijnsegment tussen twee stations
#
# Treingedrag op een segment (stoppen/doorrijden, dynamiek) zit op Train,
# niet hier — een segment is onafhankelijk van welke trein er over rijdt.

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Enum
# =============================================================================

class SegmentType(str, Enum):
    """
    Infrastructureel type van een segment.

    STATION         : stationsinfrastructuur — dwell of passing constraints
                      van toepassing (C1b, C2). Welke trein stopt of doorrijdt
                      staat op Train.halt_indicators, niet hier.
    BETWEEN_STATION : lijnsegment tussen twee stations — running time
                      constraint van toepassing (C1a).
    """
    STATION         = "station"
    BETWEEN_STATION = "between-station"


# =============================================================================
# Segment
# =============================================================================

@dataclass(frozen=True)
class Segment:
    """
    Domeinrepresentatie van een infrastructuursegment.

    Onveranderlijk na aanmaak. Bestaat onafhankelijk van treinen —
    meerdere treinen kunnen hetzelfde segment gebruiken.

    Attributes
    ----------
    id       : Unieke segment-id, consistent met SECTION in de gold timetable
               Between-station: '{LINE_NO}:{SOURCE}-{TARGET}'
               Station:         '{STATION_NAME}'
    seg_type : STATION of BETWEEN_STATION
    source   : Beginstation van het segment
    target   : Eindstation van het segment
               Voor stationssegmenten geldt source == target
    """
    id:       str
    seg_type: SegmentType
    source:   str
    target:   str

    def __post_init__(self) -> None:
        if self.seg_type == SegmentType.STATION and self.source != self.target:
            raise ValueError(
                f"Segment {self.id}: stationssegment verwacht source == target, "
                f"maar kreeg '{self.source}' != '{self.target}'"
            )
        if self.seg_type == SegmentType.BETWEEN_STATION and self.source == self.target:
            raise ValueError(
                f"Segment {self.id}: between-station segment verwacht source != target, "
                f"maar kreeg '{self.source}' == '{self.target}'"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_station(self) -> bool:
        """True als dit een stationssegment is — relevant voor C1b en C2."""
        return self.seg_type == SegmentType.STATION

    @property
    def is_line(self) -> bool:
        """True als dit een lijnsegment is — relevant voor C1a."""
        return self.seg_type == SegmentType.BETWEEN_STATION

    def __repr__(self) -> str:
        return (
            f"Segment(id='{self.id}', "
            f"type={self.seg_type.value}, "
            f"'{self.source}'→'{self.target}')"
        )