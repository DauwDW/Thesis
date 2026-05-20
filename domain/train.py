# domain/train.py
#
# Train en TrainType domeinklassen.
#
# TrainType  : binaire splitsing P/F — gebruikt door MIP (headway, gewichten)
# TrainSubtype: fijnere indeling IC/L/S/... — gebruikt door reality/ voor
#               vertragingsverdelingen per treinsoort
#
# Treingedrag op segmentniveau:
#   halt_indicators : h_t,s — stopt trein op stationssegment s?
#                     Bepaalt C1b (min dwell) en C2 (no early departure)
#   dynamics        : rijdynamiek per segment (ACC-0 / 0-BR / ACC-BR / 0-0)
#                     Gebruikt door reality/ voor vertragingssampling

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class TrainType(str, Enum):
    """
    Binaire treinclassificatie voor het MIP-model.

    Gebruikt voor:
    - Headway bepaling: H_i,j,s op basis van (type_i, type_j)
    - Prioriteitsgewichten: w_t in de objectieffunctie
    - Filteren: T_p (passenger) en T_f (freight) deelverzamelingen
    """
    PASSENGER = "P"
    FREIGHT   = "F"


class TrainSubtype(str, Enum):
    """
    Verfijnde treinclassificatie voor de reality/ module.

    Gebruikt voor het samplen van vertragingsverdelingen per treinsoort,
    niet voor MIP-constraints.
    """
    IC      = "IC"
    L       = "L"
    S       = "S"
    EURST   = "EURST"
    ICE     = "ICE"
    INT     = "INT"
    FREIGHT = "freight"


# =============================================================================
# Train
# =============================================================================

@dataclass(frozen=True)
class Train:
    """
    Domeinrepresentatie van een trein.

    Bevat alle statische informatie die nodig is voor de MIP-formulering,
    de simulatie en de reality module. Onveranderlijk na aanmaak.

    Attributes
    ----------
    train_no        : Uniek treinnummer (TRAIN_NO in Infrabel data)
    train_type      : P of F — voor MIP headway en gewichten
    train_subtype   : IC / L / S / ... — voor vertragingsverdelingen
    path            : Geordende lijst van segment-ids S(t) = (s_1, ..., s_last)
    halt_indicators : h_t,s per segment — True als trein stopt op s
                      Bepaalt of C1b en C2 actief zijn voor dit segment
    dynamics        : rijdynamiek per segment — ACC-0 / 0-BR / ACC-BR / 0-0
                      Bepaalt welke vertragingsverdeling de reality/ module
                      gebruikt bij het samplen
    """
    train_no:        int
    train_type:      TrainType
    train_subtype:   TrainSubtype
    path:            tuple[str, ...]         
    halt_indicators: dict[str, bool]
    dynamics:        dict[str, str]

    def __post_init__(self) -> None:
        if len(self.path) == 0:
            raise ValueError(f"Trein {self.train_no}: pad mag niet leeg zijn")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> int:
        """Alias voor train_no — gebruikt door instance.py en simulation/."""
        return self.train_no

    @property
    def is_passenger(self) -> bool:
        return self.train_type == TrainType.PASSENGER

    @property
    def is_freight(self) -> bool:
        return self.train_type == TrainType.FREIGHT

    @property
    def first_segment(self) -> str:
        """s_t^1 — eerste segment van het pad."""
        return self.path[0]

    @property
    def last_segment(self) -> str:
        """s_t^last — gebruikt in objectieffunctie."""
        return self.path[-1]

    # ------------------------------------------------------------------
    # Methodes
    # ------------------------------------------------------------------

    def halts_at(self, segment_id: str) -> bool:
        """
        Geeft True als trein stopt op het gegeven stationssegment.
        Bepaalt of C1b (minimum dwell time) en C2 (no early departure)
        actief zijn voor dit segment.
        """
        return self.halt_indicators.get(segment_id, False)

    def dynamics_at(self, segment_id: str) -> str | None:
        """
        Geeft de rijdynamiek op een segment terug.
        Returnt None als segment niet in het pad zit.
        """
        return self.dynamics.get(segment_id)

    def consecutive_segments(self) -> list[tuple[str, str]]:
        """
        Geordende paren (s, s_next) voor constraint C1c.

        Voorbeeld:
            path = (A, B, C) → [(A, B), (B, C)]
        """
        return list(zip(self.path[:-1], self.path[1:]))

    def __repr__(self) -> str:
        return (
            f"Train(no={self.train_no}, "
            f"type={self.train_type.value}, "
            f"subtype={self.train_subtype.value}, "
            f"segments={len(self.path)})"
        )