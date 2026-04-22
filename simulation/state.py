# simulation/state.py
#
# SystemState — centrale toestandsrepresentatie van de simulatie.
#
# Houdt bij:
#   - Huidige simulatietijd
#   - Actuele entry/exit tijden per (trein, segment)
#   - Positie per trein (huidig actief segment)
#
# Wordt aangemaakt door de simulatie-engine en doorgegeven aan:
#   - controller/controller.py  (remaining_path, current_delay, active_train_ids)
#   - controller/triggers.py    (current_delay, active_train_ids)
#   - model/instance.py         (remaining_path, current_delay, is_finished,
#                                current_segment, actual_entry)
#
# Tijdsconventie (consistent met domain/schedule.py en MIP-variabelen):
#   actual_entry(t, s) = A_t,s actueel — moment trein segment binnenkomt (seconden)
#   actual_exit(t, s)  = D_t,s actueel — moment trein segment verlaat   (seconden)
#   actual_exit is None zolang de trein het segment nog niet verlaten heeft.
#
# Schrijf-interface (uitsluitend via de engine):
#   record_entry / record_exit — aangeroepen door simulator.py bij events
#   advance_time               — aangeroepen door simulator.py bij elke tijdstap
#
# Naamgeving afgestemd met partner (model/, controller/):
#   actual_entry  ←→  actual_arrival  in controller/controller.py en model/instance.py
#   → partner past hun code aan naar actual_entry
#
# Verantwoordelijkheidsgrenzen:
#   apply_solution → simulator.py   (herbouwt event-queue op basis van MIP-tijden)
#   apply_fcfs     → event_queue.py (herordent events in de queue)

from __future__ import annotations

from domain import Train, Timetable


class SystemState:
    """
    Centrale toestandsrepresentatie van de lopende simulatie.

    Bevat uitsluitend actuele (gemeten/gesimuleerde) tijden — de geplande
    tijden blijven in Timetable en worden nooit overschreven.

    Parameters
    ----------
    trains     : dict[int, Train]  — alle treinobjecten geïndexeerd op train_no
    timetable  : Timetable         — geplande tijden (referentie, onveranderlijk)
    start_time : float             — starttijd van de simulatie in seconden

    Interne datastructuur
    ---------------------
    _actual : dict[int, dict[str, tuple[float | None, float | None]]]
        _actual[train_id][segment_id] = (actual_entry, actual_exit)
        actual_entry : float       — moment trein segment binnenkwam (seconden)
        actual_exit  : float|None  — moment trein segment verliet, None indien nog actief

    _current_segment : dict[int, str | None]
        Huidig actief segment per trein.
        None als trein nog niet gestart of al klaar is.

    Gedrag voor treinen die nog niet gestart zijn
    ---------------------------------------------
    remaining_path() geeft het volledige pad terug voor treinen zonder
    geregistreerde entry — ze worden door de engine nog niet op het netwerk
    geplaatst maar zijn wel relevant voor conflictdetectie in instance.py.
    """

    def __init__(
        self,
        trains:     dict[int, Train],
        timetable:  Timetable,
        start_time: float = 0.0,
    ) -> None:
        self._trains    = trains
        self._timetable = timetable
        self._time      = start_time

        # Actuele tijden — leeg bij opstart, gevuld door engine via record_entry/record_exit
        self._actual: dict[int, dict[str, tuple[float | None, float | None]]] = {
            train_id: {} for train_id in trains
        }

        # Huidig actief segment per trein — None = nog niet gestart of al klaar
        self._current_segment: dict[int, str | None] = {
            train_id: None for train_id in trains
        }

    # ==========================================================================
    # Tijdbeheer
    # ==========================================================================

    @property
    def current_time(self) -> float:
        """Huidige simulatietijd in seconden."""
        return self._time

    def advance_time(self, new_time: float) -> None:
        """
        Zet de simulatietijd vooruit.

        Parameters
        ----------
        new_time : float — nieuwe simulatietijd (moet >= huidige tijd zijn)
        """
        if new_time < self._time:
            raise ValueError(
                f"Simulatietijd mag niet achteruit: {new_time} < {self._time}"
            )
        self._time = new_time

    # ==========================================================================
    # Event registratie — aangeroepen door simulator.py
    # ==========================================================================

    def record_entry(self, train_id: int, segment_id: str, time: float) -> None:
        """
        Registreert dat trein train_id segment segment_id binnengekomen is.

        Wordt aangeroepen door simulator.py bij een TrainEntered-event.

        Parameters
        ----------
        train_id   : int   — treinnummer
        segment_id : str   — segment-id (= SECTION in gold timetable)
        time       : float — actuele entrytijd in seconden
        """
        self._actual[train_id][segment_id] = (time, None)
        self._current_segment[train_id]    = segment_id

    def record_exit(self, train_id: int, segment_id: str, time: float) -> None:
        """
        Registreert dat trein train_id segment segment_id verlaten heeft.

        Wordt aangeroepen door simulator.py bij een TrainExited-event.
        Als het het laatste segment is, wordt _current_segment op None gezet.

        Parameters
        ----------
        train_id   : int   — treinnummer
        segment_id : str   — segment-id
        time       : float — actuele exittijd in seconden
        """
        entry, _ = self._actual[train_id].get(segment_id, (None, None))
        self._actual[train_id][segment_id] = (entry, time)

        train = self._trains[train_id]
        if segment_id == train.last_segment:
            self._current_segment[train_id] = None
        # Anders: _current_segment wordt bij de volgende record_entry bijgewerkt

    # ==========================================================================
    # Public interface — aangeroepen door controller/ en model/
    # ==========================================================================

    def remaining_path(self, train_id: int) -> list[str]:
        """
        Geeft de nog af te leggen segmenten voor trein train_id.

        Een segment telt als 'resterend' als de trein het nog niet verlaten
        heeft (actual_exit is None). Afgeronde segmenten worden weggelaten.

        Treinen die nog niet gestart zijn (lege _actual) krijgen hun
        volledige pad terug — ze zijn nog niet ingezet maar wél relevant
        voor conflictdetectie in model/instance.py.

        Parameters
        ----------
        train_id : int

        Returns
        -------
        list[str] — geordende lijst van resterende segment-ids
                    leeg als de trein klaar is
        """
        train    = self._trains[train_id]
        segments = self._actual[train_id]

        return [
            seg_id for seg_id in train.path
            if segments.get(seg_id, (None, None))[1] is None
            # exit is None → segment nog niet verlaten (of nog niet betreden)
        ]

    def current_delay(self, train_id: int) -> float:
        """
        Geeft de huidige vertraging van trein train_id in seconden.

        Vertraging = actuele exit van het laatste verlaten segment
                   − geplande exit van datzelfde segment.

        Als de trein nog geen enkel segment verlaten heeft (nog niet gestart),
        is de vertraging 0.0. Negatieve waarden (vroeger dan gepland) worden
        afgekapt op 0.0.

        Parameters
        ----------
        train_id : int

        Returns
        -------
        float — vertraging in seconden (0.0 als trein nog niet gestart)
        """
        train    = self._trains[train_id]
        segments = self._actual[train_id]

        for seg_id in reversed(train.path):
            entry, exit_ = segments.get(seg_id, (None, None))
            if exit_ is not None:
                planned_exit = self._timetable.scheduled_departure(train_id, seg_id)
                return max(0.0, exit_ - planned_exit)

        return 0.0  # trein heeft nog geen segment verlaten

    def is_finished(self, train_id: int) -> bool:
        """
        True als de trein zijn volledige pad afgelegd heeft.

        Een trein is klaar als het laatste segment in zijn pad een
        gekende actual_exit heeft.

        Parameters
        ----------
        train_id : int

        Returns
        -------
        bool
        """
        train    = self._trains[train_id]
        _, exit_ = self._actual[train_id].get(train.last_segment, (None, None))
        return exit_ is not None

    def active_train_ids(self) -> list[int]:
        """
        Geeft de ids van alle treinen die actief zijn (gestart, niet klaar).

        Een trein is actief als hij minstens één segment binnengekomen is
        maar het laatste segment nog niet verlaten heeft.

        Returns
        -------
        list[int]
        """
        return [
            train_id for train_id in self._trains
            if self._actual[train_id]           # minstens één entry geregistreerd
            and not self.is_finished(train_id)
        ]

    def actual_entry(self, train_id: int, segment_id: str) -> float:
        """
        Actuele entrytijd (A_t,s) van trein train_id op segment segment_id.

        Naam afgestemd met partner: model/instance.py en controller/controller.py
        gebruiken state.actual_entry() (niet actual_arrival).

        Parameters
        ----------
        train_id   : int
        segment_id : str

        Returns
        -------
        float — actual entry in seconden

        Raises
        ------
        KeyError als de trein het segment nog niet binnengekomen is
        """
        entry, _ = self._actual[train_id].get(segment_id, (None, None))
        if entry is None:
            raise KeyError(
                f"Geen actual_entry voor trein {train_id} op segment '{segment_id}' "
                f"— trein heeft dit segment nog niet betreden"
            )
        return entry

    def actual_exit(self, train_id: int, segment_id: str) -> float:
        """
        Actuele exittijd (D_t,s) van trein train_id op segment segment_id.

        Parameters
        ----------
        train_id   : int
        segment_id : str

        Returns
        -------
        float — actual exit in seconden

        Raises
        ------
        KeyError als de trein het segment nog niet verlaten heeft
        """
        _, exit_ = self._actual[train_id].get(segment_id, (None, None))
        if exit_ is None:
            raise KeyError(
                f"Geen actual_exit voor trein {train_id} op segment '{segment_id}' "
                f"— trein heeft dit segment nog niet verlaten"
            )
        return exit_

    # ==========================================================================
    # Diagnostiek
    # ==========================================================================

    def current_segment(self, train_id: int) -> str | None:
        """
        Huidig actief segment van trein train_id.

        Returns None als trein nog niet gestart of al klaar is.
        """
        return self._current_segment[train_id]

    def summary(self) -> str:
        """
        Korte samenvatting van de huidige toestand — voor logging en debugging.
        """
        n_total    = len(self._trains)
        n_active   = len(self.active_train_ids())
        n_finished = sum(1 for t_id in self._trains if self.is_finished(t_id))
        n_waiting  = n_total - n_active - n_finished

        return (
            f"SystemState(t={self._time:.0f}s | "
            f"actief={n_active}, klaar={n_finished}, wachtend={n_waiting})"
        )

    def __repr__(self) -> str:
        return self.summary()