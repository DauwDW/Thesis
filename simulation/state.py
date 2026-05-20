# simulation/state.py
#
# SystemState — centrale toestandsrepresentatie van de simulatie.
#
# Houdt bij:
#   - Huidige simulatietijd
#   - Actuele entry/exit tijden per (trein, segment)
#   - Positie per trein (huidig actief segment)
#   - Gesamplede bezettingsduren per (trein, segment)
#   - MIP-geplande entry/exit tijden per (trein, segment)
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
# Atomaire overgang (nieuwe architectuur):
#   record_exit(N) en record_entry(N+1) worden altijd samen aangeroepen
#   in _handle_ready_to_exit. Er bestaat nooit een toestand waarbij
#   actual_exit(N) geregistreerd is maar actual_entry(N+1) ontbreekt,
#   tenzij N het laatste segment is.
#
# Schrijf-interface (uitsluitend via simulator.py):
#   record_entry / record_exit — aangeroepen bij atomaire transfer of terminale exit
#   advance_time               — aangeroepen bij elk event in de event-loop
#
# Verantwoordelijkheidsgrenzen:
#   _apply_solution → simulator.py (herbouwt event-queue op basis van MIP-tijden)
#   _apply_fcfs     → simulator.py (past dispatcher-priorities aan op basis van FCFS)

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

    _sampled_durations : dict[tuple[int, str], float]
        Gesamplede bezettingsduur per (train_id, segment_id).

    _mip_schedule : dict[int, dict[str, tuple[float, float]]]
        MIP-geplande (entry, exit) tijden per trein en segment.

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

        self._actual: dict[int, dict[str, tuple[float | None, float | None]]] = {
            train_id: {} for train_id in trains
        }

        self._current_segment: dict[int, str | None] = {
            train_id: None for train_id in trains
        }

        self._sampled_durations: dict[tuple[int, str], float] = {}

        self._mip_schedule: dict[int, dict[str, tuple[float, float]]] = {
            train_id: {} for train_id in trains
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

        In de atomaire architectuur wordt dit altijd samen met record_exit
        van het vorige segment aangeroepen (behalve voor het eerste segment).

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

        Wordt aangeroepen bij atomaire transfer naar het volgende segment,
        of bij terminale exit (laatste segment van het pad).

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

    # ==========================================================================
    # MIP schedule — aangeroepen door simulator.py via _apply_solution
    # ==========================================================================

    def clear_mip_schedule(self, train_id: int) -> None:
        """Wist het volledige MIP-plan voor trein train_id."""
        self._mip_schedule[train_id] = {}

    def record_mip_schedule(
        self,
        train_id: int,
        segment_id: str,
        mip_entry: float,
        mip_exit: float,
    ) -> None:
        """
        Slaat de MIP-geplande entry- en exittijd op voor trein train_id
        op segment segment_id.

        Parameters
        ----------
        train_id   : int   — treinnummer
        segment_id : str   — segment-id
        mip_entry  : float — MIP-geplande entrytijd in seconden
        mip_exit   : float — MIP-geplande exittijd in seconden
        """
        self._mip_schedule[train_id][segment_id] = (mip_entry, mip_exit)

    def mip_exit_for(self, train_id: int, segment_id: str) -> float | None:
        """
        Geeft de MIP-geplande exittijd voor trein train_id op segment segment_id.

        Returns None als er geen MIP-plan beschikbaar is voor dit segment.
        """
        exit_time = self._mip_schedule[train_id].get(segment_id)
        if exit_time is None:
            return None
        return exit_time[1]

    def mip_entry_for(self, train_id: int, segment_id: str) -> float | None:
        """
        Geeft de MIP-geplande entrytijd voor trein train_id op segment segment_id.

        Returns None als er geen MIP-plan beschikbaar is voor dit segment.
        """
        entry_time = self._mip_schedule[train_id].get(segment_id)
        if entry_time is None:
            return None
        return entry_time[0]

    # ==========================================================================
    # Gesamplede duren — aangeroepen door simulator.py
    # ==========================================================================

    def record_sampled_duration(
        self,
        train_id: int,
        segment_id: str,
        duration: float,
    ) -> None:
        """
        Slaat de gesamplede bezettingsduur op voor trein train_id
        op segment segment_id.
        """
        self._sampled_durations[(train_id, segment_id)] = duration

    def sampled_duration(self, train_id: int, segment_id: str) -> float:
        """
        Geeft de gesamplede bezettingsduur terug.

        Raises
        ------
        KeyError als nog geen duration geregistreerd werd.
        """
        try:
            return self._sampled_durations[(train_id, segment_id)]
        except KeyError:
            raise KeyError(
                f"Geen sampled_duration voor trein {train_id} "
                f"op segment '{segment_id}'"
            )

    # ==========================================================================
    # Public interface — aangeroepen door controller/ en model/
    # ==========================================================================

    def remaining_path(self, train_id: int) -> list[str]:
        """
        Geeft de resterende segmenten in het pad van trein train_id.

        Voor niet-gestarte treinen: volledig pad.
        Voor actieve treinen: huidig segment t/m laatste.
        Voor afgewerkte treinen: lege lijst.
        """
        train = self._trains[train_id]
        current = self._current_segment[train_id]
        actual = self._actual[train_id]

        if current is None:
            if not actual:
                return list(train.path)  # nog niet gestart
            # Safeguard: check of er nog onafgewerkte segmenten zijn
            for i, seg in enumerate(train.path):
                entry, exit_ = actual.get(seg, (None, None))
                if entry is not None and exit_ is None:
                    return list(train.path[i:])
            return []  # effectief klaar

        try:
            idx = train.path.index(current)
            return list(train.path[idx:])
        except ValueError:
            return []

    def current_delay(self, train_id: int) -> float:
        """
        Geeft de huidige vertraging van trein train_id in seconden.

        Vertraging = actuele exit van het laatste verlaten segment
                   − geplande exit van datzelfde segment.

        Afgekapt op 0.0: negatieve offsets tellen niet als vertraging
        voor de trigger-logica. Gebruik actual_exit() voor de ruwe offset.

        Returns 0.0 als de trein nog geen enkel segment verlaten heeft.
        """
        train    = self._trains[train_id]
        segments = self._actual[train_id]

        for seg_id in reversed(train.path):
            entry, exit_ = segments.get(seg_id, (None, None))
            if exit_ is not None:
                planned_exit = self._timetable.scheduled_exit(train_id, seg_id)
                return max(0.0, exit_ - planned_exit)

        return 0.0

    def is_finished(self, train_id: int) -> bool:
        """True als de trein zijn volledige pad afgelegd heeft."""
        train    = self._trains[train_id]
        _, exit_ = self._actual[train_id].get(train.last_segment, (None, None))
        return exit_ is not None

    def active_train_ids(self) -> list[int]:
        """
        Geeft ids van alle treinen die actief zijn op het netwerk.

        Actief = minstens één segment betreden EN nog niet volledig afgewerkt.
        Niet-gestarte treinen zijn hier niet in opgenomen — zie remaining_path()
        voor treinen die het netwerk nog niet betreden hebben.
        """
        active = []
        for train_id in self._trains:
            if not self._actual[train_id]:
                continue
            if self.is_finished(train_id):
                continue
            active.append(train_id)
        return active

    def actual_entry(self, train_id: int, segment_id: str) -> float:
        """
        Actuele entrytijd (A_t,s) van trein train_id op segment segment_id.

        Raises
        ------
        KeyError als de trein het segment nog niet binnengekomen is.
        """
        entry, _ = self._actual[train_id].get(segment_id, (None, None))
        if entry is None:
            raise KeyError(
                f"Geen actual_entry voor trein {train_id} op segment '{segment_id}'"
            )
        return entry

    def actual_exit(self, train_id: int, segment_id: str) -> float:
        """
        Actuele exittijd (D_t,s) van trein train_id op segment segment_id.

        Raises
        ------
        KeyError als de trein het segment nog niet verlaten heeft.
        """
        _, exit_ = self._actual[train_id].get(segment_id, (None, None))
        if exit_ is None:
            raise KeyError(
                f"Geen actual_exit voor trein {train_id} op segment '{segment_id}'"
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