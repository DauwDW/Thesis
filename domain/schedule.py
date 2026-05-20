# ScheduledTimes en Timetable domeinklassen.
#
# ScheduledTimes : geplande tijden voor één (trein, segment) combinatie
# Timetable      : centrale opslag en opvraging van alle geplande tijden
#
# Tijdsconventie (consistent met data/timetable.py en MIP-variabelen):
#   entry_seconds = A_t,s = moment binnenkomst segment  (vroeger)
#   exit_seconds  = D_t,s = moment verlaten segment     (later)
#   → altijd entry_seconds < exit_seconds

from __future__ import annotations
from dataclasses import dataclass


# =============================================================================
# ScheduledTimes
# =============================================================================

@dataclass(frozen=True)
class ScheduledTimes:
    """
    Geplande tijden voor één (trein, segment) combinatie.

    Nooit los gebruikt — altijd opgezocht via Timetable.get().

    Attributes
    ----------
    entry_seconds : A_t,s — moment trein segment binnenkomt (seconden)
    exit_seconds  : D_t,s — moment trein segment verlaat (seconden)
    running_time  : RT_t,s — minimale rijtijd op lijnsegment (seconden)
                    None voor stationssegmenten
    dwell_time    : DW_t,s — minimale dwell tijd op stationssegment (seconden)
                    None voor lijnsegmenten
    halts         : h_t,s — True als trein stopt op dit stationssegment
                    None voor lijnsegmenten
    """
    entry_seconds: float
    exit_seconds:  float
    running_time:  float | None
    dwell_time:    float | None

    def __post_init__(self) -> None:
        if self.exit_seconds < self.entry_seconds:
            raise ValueError(
                f"exit_seconds ({self.exit_seconds}) moet groter zijn dan "
                f"entry_seconds ({self.entry_seconds})"
            )


# =============================================================================
# Timetable
# =============================================================================

class Timetable:
    """
    Centrale opslag van alle geplande tijden per (train_no, segment_id).

    Onveranderlijk na aanmaak — de geplande timetable wijzigt nooit
    tijdens de simulatie. De simulatie past de actuele tijden aan in
    SystemState, niet hier.

    Aangemaakt door data/input.py via load_timetable_from_gold().
    Gebruikt door model/instance.py en simulation/state.py.
    """

    def __init__(self, data: dict[tuple[int, str], ScheduledTimes]) -> None:
        self._data = data

    # ------------------------------------------------------------------
    # Opvraging — interface voor instance.py
    # ------------------------------------------------------------------

    def scheduled_entry(self, train_no: int, segment_id: str) -> float:
        """A_t,s — geplande aankomsttijd (entry_seconds)."""
        return self._data[(train_no, segment_id)].entry_seconds

    def scheduled_exit(self, train_no: int, segment_id: str) -> float:
        """D_t,s — geplande vertrektijd (exit_seconds)."""
        return self._data[(train_no, segment_id)].exit_seconds

    def running_time(self, train_no: int, segment_id: str) -> float:
        """RT_t,s — minimale rijtijd op lijnsegment."""
        rt = self._data[(train_no, segment_id)].running_time
        if rt is None:
            raise ValueError(
                f"Geen running_time voor trein {train_no} op segment '{segment_id}' "
                f"— is dit een stationssegment?"
            )
        return rt

    def dwell_time(self, train_no: int, segment_id: str) -> float:
        """DW_t,s — minimale dwell tijd op stationssegment."""
        dw = self._data[(train_no, segment_id)].dwell_time
        if dw is None:
            raise ValueError(
                f"Geen dwell_time voor trein {train_no} op segment '{segment_id}' "
                f"— is dit een lijnsegment?"
            )
        return dw

    def get(self, train_no: int, segment_id: str) -> ScheduledTimes:
        """Geeft het volledige ScheduledTimes object terug."""
        return self._data[(train_no, segment_id)]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        n_trains = len({train_no for train_no, _ in self._data})
        return f"Timetable({n_trains} treinen, {len(self._data)} segmenten)"