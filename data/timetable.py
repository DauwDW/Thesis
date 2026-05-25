import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import BRONZE_DIR, GOLD_DIR, PASSING_DURATION_PASSENGER

logger = logging.getLogger(__name__)

TRAIN_GROUP = ['DATDEP', 'RELATION_DIRECTION', 'TRAIN_NO']


# =============================================================================
# Bronze data laden
# =============================================================================

def load_bronze(months: list[str]) -> pd.DataFrame:
    """Laadt en combineert meerdere bronze parquet bestanden."""
    dfs = []
    for month in months:
        path = BRONZE_DIR / f"{month}.parquet"
        df = pd.read_parquet(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Timetable opbouw
# =============================================================================

def get_main_timetable(train_no: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Bepaalt de meest voorkomende timetable voor een trein over alle dagen.
    Treinen rijden niet elke dag exact hetzelfde — dit pikt de dominante variant.
    Bij gelijkstand wordt de eerste variant teruggegeven.
    """
    train = df[df['TRAIN_NO'] == train_no]
    timetables = []
    counts = []

    for date in train['DATDEP'].unique():
        timetable = train[train['DATDEP'] == date][
            ['SECTION', 'SOURCE', 'TARGET',
             'PLANNED_ENTRY', 'PLANNED_EXIT',
             'RELATION_DIRECTION']
        ].reset_index(drop=True)

        idx = -1
        for i, existing in enumerate(timetables):
            if existing.equals(timetable):
                idx = i
                break

        if idx != -1:
            counts[idx] += 1
        else:
            timetables.append(timetable)
            counts.append(1)

    max_idx = counts.index(max(counts))
    return timetables[max_idx]


def build_planned_timetable(df: pd.DataFrame, min_frequency: float = 0.85) -> pd.DataFrame:
    """
    Bouwt de geplande timetable op uit bronze data.

    SECTION-conventie:
        Between-station: {LINE_NO_DEP}:{SOURCE}-{TARGET}  (lijncode relevant)
        Dwell:           {SOURCE}                          (lijncode zinloos)

    Args:
        df:            gecombineerde bronze DataFrame
        min_frequency: minimale fractie weekdagen dat een trein moet rijden

    Returns:
        DataFrame met geplande timetable per trein per segment
    """
    df = df.copy()
    df['LINE_NO_DEP'] = df['LINE_NO_DEP'].str.replace('/', '-')

    dwell_mask = df['SOURCE'] == df['TARGET']
    df['SECTION'] = np.where(
        dwell_mask,
        df['SOURCE'].astype(str),
        df['LINE_NO_DEP'].astype(str) + ':' + df['SOURCE'].astype(str) + '-' + df['TARGET'].astype(str)
    )

    total_days = df['DATDEP'].nunique()
    timetables = []

    for train_no in df['TRAIN_NO'].unique():
        train = df[df['TRAIN_NO'] == train_no]
        frequency = len(train['DATDEP'].unique()) / total_days
        if frequency >= min_frequency:
            timetable = get_main_timetable(train_no, df)
            timetable['TRAIN_NO'] = train_no
            timetables.append(timetable)

    return pd.concat(timetables, ignore_index=True)


# =============================================================================
# Tijdsconversie
# =============================================================================

def add_time_in_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converteert geplande tijden naar seconden relatief aan vroegste entry.

    Geen swap nodig — PLANNED_ENTRY <= PLANNED_EXIT altijd.

    Conventie:
        ENTRY_SECONDS = A_t,s = moment binnenkomst segment  (vroeger)
        EXIT_SECONDS  = D_t,s = moment verlaten segment     (later)
    """
    df = df.copy()
    df['PLANNED_ENTRY'] = pd.to_datetime(df['PLANNED_ENTRY'])
    df['PLANNED_EXIT']  = pd.to_datetime(df['PLANNED_EXIT'])

    min_entry = df['PLANNED_ENTRY'].min()
    df['ENTRY_SECONDS'] = (df['PLANNED_ENTRY'] - min_entry).dt.total_seconds()
    df['EXIT_SECONDS']  = (df['PLANNED_EXIT']  - min_entry).dt.total_seconds()

    return df


def sort_timetable(df: pd.DataFrame, time_col: str = 'ENTRY_SECONDS') -> pd.DataFrame:
    """
    Sorteert de timetable zodat segmenten chronologisch correct staan per trein.

    Bij gelijke tijden krijgt SOURCE altijd voorrang, daarna WITHIN-STATION-*,
    dan BETWEEN-STATION — zodat passing-segmenten altijd vóór het vertrekkende
    between-station segment staan.

    Args:
        time_col: kolom om op te sorteren — standaard 'ENTRY_SECONDS'.
    """
    type_order = {
        'SOURCE':                 0,
        'WITHIN-STATION-DWELL':   1,
        'WITHIN-STATION-PASSING': 1,
        'BETWEEN-STATION':        2,
    }
    df = df.copy()
    df['_sort_key'] = df['TYPE'].map(type_order)
    df = df.sort_values(
        ['TRAIN_NO', time_col, '_sort_key']
    ).drop(columns='_sort_key').reset_index(drop=True)
    return df


def add_passing_time(df: pd.DataFrame) -> pd.DataFrame:
    df = sort_timetable(df.copy())

    passing_mask = df['TYPE'] == 'WITHIN-STATION-PASSING'

    # 1. Geef passing duur
    df.loc[passing_mask, 'EXIT_SECONDS'] += PASSING_DURATION_PASSENGER
    df.loc[passing_mask, 'PLANNED_EXIT'] += pd.Timedelta(seconds=PASSING_DURATION_PASSENGER)


    # 2. Binnen elke trein: markeer "vorige was passing"
    prev_was_passing = (
        df.groupby('TRAIN_NO')['TYPE']
        .shift(1)
        .eq('WITHIN-STATION-PASSING')
    )

    # 3. Shift ENTRY van volgende segment
    df.loc[prev_was_passing, 'ENTRY_SECONDS'] += PASSING_DURATION_PASSENGER
    df.loc[prev_was_passing, 'PLANNED_ENTRY'] += pd.Timedelta(seconds=PASSING_DURATION_PASSENGER)


    return sort_timetable(df, time_col='ENTRY_SECONDS')

# =============================================================================
# Periode
# =============================================================================

def add_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt dagperiode toe per segment op basis van geplande aankomsttijd (PLANNED_EXIT).
    Moet aangeroepen worden VOOR add_time_in_seconds.
    """
    df = df.copy()
    df['PLANNED_EXIT'] = pd.to_datetime(df['PLANNED_EXIT'])

    def get_period(dt: pd.Timestamp) -> str:
        hour = dt.hour
        if 0 <= hour < 6:
            return 'NIGHT'
        elif 6 <= hour < 9:
            return 'MORNING PEAK'
        elif 9 <= hour < 16:
            return 'DAYTIME'
        elif 16 <= hour < 19:
            return 'EVENING PEAK'
        else:
            return 'EVENING'

    df['PERIOD'] = df['PLANNED_EXIT'].apply(get_period)
    return df


# =============================================================================
# Segmenttype en dynamiek
# =============================================================================

def add_segment_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classificeert elk segment als SOURCE, BETWEEN-STATION,
    WITHIN-STATION-PASSING of WITHIN-STATION-DWELL.

    Werkt op PLANNED_ENTRY / PLANNED_EXIT datetime kolommen.
    Moet aangeroepen worden VOOR add_time_in_seconds.

    PLANNED_ENTRY <= PLANNED_EXIT altijd — geen abs() nodig.
    Duur = 0: passing. Duur > 5s: dwell.
    """
    df = df.copy()
    df['is_source'] = df.groupby('TRAIN_NO').cumcount() == 0

    df['_duration'] = (
        pd.to_datetime(df['PLANNED_EXIT']) -
        pd.to_datetime(df['PLANNED_ENTRY'])
    ).dt.total_seconds()

    df['TYPE'] = np.where(
        df['is_source'], 'SOURCE',
        np.where(
            df['SOURCE'] != df['TARGET'], 'BETWEEN-STATION',
            np.where(
                df['_duration'] <= 5,
                'WITHIN-STATION-PASSING',
                'WITHIN-STATION-DWELL'
            )
        )
    )
    return df.drop(columns=['is_source', '_duration'])


def add_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt rijdynamiek toe per segment op basis van voor- en volgend segmenttype.
    ACC-0  : trein trekt op, geen remming
    0-BR   : geen optrekken, trein remt
    ACC-BR : trein trekt op én remt
    0-0    : geen optrekken, geen remming

    Voor WITHIN-STATION-DWELL segmenten is DYNAMICS altijd '0-0' —
    stilstaande treinen hebben geen rijdynamiek.

    Moet aangeroepen worden NA add_passing_time (zodat volgorde correct is).
    """
    df = sort_timetable(df, time_col='ENTRY_SECONDS')
    df = df.copy()

    df['PREVIOUS_TYPE'] = df.groupby('TRAIN_NO')['TYPE'].shift(1)
    df['NEXT_TYPE']     = df.groupby('TRAIN_NO')['TYPE'].shift(-1)

    df['ACCELERATION'] = ~df['PREVIOUS_TYPE'].isin(
        ['BETWEEN-STATION', 'WITHIN-STATION-PASSING']       # is dit niet hetzelfde als df['PREVIOUS_TYPE].isin['WITHIN-STATION-DWELL']
    )
    df['BREAKING'] = df['NEXT_TYPE'].isin(['WITHIN-STATION-DWELL'])

    df['DYNAMICS'] = ''
    df.loc[ df['ACCELERATION'] & ~df['BREAKING'], 'DYNAMICS'] = 'ACC-0'
    df.loc[~df['ACCELERATION'] &  df['BREAKING'], 'DYNAMICS'] = '0-BR'
    df.loc[ df['ACCELERATION'] &  df['BREAKING'], 'DYNAMICS'] = 'ACC-BR'
    df.loc[~df['ACCELERATION'] & ~df['BREAKING'], 'DYNAMICS'] = '0-0'

    # Dwell-segmenten hebben geen rijdynamiek
    df.loc[df['TYPE'] == 'WITHIN-STATION-DWELL', 'DYNAMICS'] = '0-0'

    return df.drop(columns=['PREVIOUS_TYPE', 'NEXT_TYPE', 'ACCELERATION', 'BREAKING'])


def add_train_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = df['RELATION_DIRECTION'].str.extract(r'^([^:\s]+)')[0]
    df['TRAIN_TYPE'] = raw.str.replace(r'^S\d.*', 'S', regex=True)
    return df

# =============================================================================
# Platformtoewijzing
# =============================================================================

@dataclass
class TrackAssignmentDiagnostics:
    """Bijhoudt conflictstatistieken per station en perrongroep."""
    conflicts:   dict = field(default_factory=dict)
    assignments: dict = field(default_factory=dict)
    invalid:     dict = field(default_factory=dict)

    def log_conflict(self, station: str, groep: str) -> None:
        key = (station, groep)
        self.conflicts[key] = self.conflicts.get(key, 0) + 1

    def log_assignment(self, station: str, groep: str) -> None:
        key = (station, groep)
        self.assignments[key] = self.assignments.get(key, 0) + 1

    def log_invalid(self, station: str, groep: str, count: int) -> None:
        key = (station, groep)
        self.invalid[key] = self.invalid.get(key, 0) + count

    def summary(self) -> str:
        lines = ["Track assignment diagnostics:"]
        for key in sorted(self.assignments):
            station, groep = key
            n = self.assignments[key]
            c = self.conflicts.get(key, 0)
            i = self.invalid.get(key, 0)
            lines.append(
                f"  {station} ({groep}): {n} toewijzingen, "
                f"{c} capaciteitsconflicten, {i} ongeldige intervallen"
            )
        return "\n".join(lines)


def _assign_tracks_by_overlap(
    df:           pd.DataFrame,
    mask:         pd.Series,
    perron_groep: str,
    platforms:    list,
    station:      str,
    diagnostics:  TrackAssignmentDiagnostics,
) -> None:
    """
    Wijst individuele platforms toe binnen één eilandperron via greedy
    interval scheduling (klassiek 'earliest-available machine first').
    Modifieert df['PERRON'] in-place.

    Volgorde: treinen op ENTRY_SECONDS oplopend, met TRAIN_NO als
    deterministische tiebreak. Per trein wordt het platform gekozen
    dat het vroegst vrijkomt; bij gelijkstand alfabetisch eerste.
    Als dat platform nog bezet is op moment van aankomst, wordt het
    conflict gelogd en de trein krijgt alsnog het minst-bezette spoor.
    """
    groep_mask   = mask & (df['PERRON_GROEP'] == perron_groep)
    invalid_mask = groep_mask & (df['EXIT_SECONDS'] <= df['ENTRY_SECONDS'])
    n_invalid    = invalid_mask.sum()

    if n_invalid > 0:
        diagnostics.log_invalid(station, perron_groep, n_invalid)
        logger.warning(
            f"{station} ({perron_groep}): {n_invalid} ongeldige intervallen "
            f"(EXIT_SECONDS <= ENTRY_SECONDS) — worden overgeslagen"
        )

    work_mask = groep_mask & ~invalid_mask
    if not work_mask.any():
        return

    sorted_idx = (
        df.loc[work_mask, ['ENTRY_SECONDS', 'TRAIN_NO']]
          .sort_values(['ENTRY_SECONDS', 'TRAIN_NO'])
          .index
    )
    free_from = {p: -1 for p in sorted(platforms)}

    for idx in sorted_idx:
        entry = df.at[idx, 'ENTRY_SECONDS']
        exit_ = df.at[idx, 'EXIT_SECONDS']
        diagnostics.log_assignment(station, perron_groep)

        chosen = min(free_from, key=lambda p: (free_from[p], p))

        if free_from[chosen] > entry:
            overlap = free_from[chosen] - entry
            diagnostics.log_conflict(station, perron_groep)
            logger.debug(
                f"{station} ({perron_groep}): capaciteitsconflict op "
                f"{chosen} — overlap van {overlap:.0f}s"
            )

        df.at[idx, 'PERRON']  = chosen
        free_from[chosen]     = exit_
        

def assign_platforms(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, TrackAssignmentDiagnostics]:
    """
    Verfijnt SECTION voor station-segmenten door individueel platform toe te voegen.

    Methodologie per stationstype:
    - Vaste platform-lijn toewijzing: Brussel-Noord, Jette, Bockstael,
      Brussel-Schuman, Brussel-West, Simonis, Thurn en Taxis,
      Vorst-Oost, Zellik, Sint-Agatha-Berchem
    - Tijdsoverlap via greedy interval scheduling: Brussel-Centraal,
      Brussel-Congres, Brussel-Kapellekerk, Anderlecht
    - RELATION_DIRECTION als proxy: Schaarbeek, Brussel-Zuid
    """
    df = df.copy()
    diagnostics = TrackAssignmentDiagnostics()

    df['PREVIOUS_SECTION'] = df.groupby('TRAIN_NO')['SECTION'].shift(1)
    df['NEXT_SECTION']     = df.groupby('TRAIN_NO')['SECTION'].shift(-1)
    df['PREVIOUS_SECTION'] = df['PREVIOUS_SECTION'].str.split(':').str[0]
    df['NEXT_SECTION']     = df['NEXT_SECTION'].str.split(':').str[0]
    df['PREVIOUS_STATION'] = df.groupby('TRAIN_NO')['SOURCE'].shift(1)
    df['NEXT_STATION']     = df.groupby('TRAIN_NO')['TARGET'].shift(-1)
    df['PERRON']           = None
    df['PERRON_GROEP']     = None

    # -------------------------------------------------------------------------
    # BRUSSEL-NOORD
    # Vaste platform-lijn toewijzing op basis van aankomende/vertrekkende lijn.
    # Bron: domeinkennis spoorinfrastructuur Brussel-Noord (Mariska, 2024).
    #
    # Twee gevallen:
    #   Geval 1: PREVIOUS_SECTION is een echte lijncode (50, 36N, 25, ...)
    #            → trein komt van buiten het station
    #            → gebruik bxl_noord_in: aankomende lijn bepaalt platform
    #   Geval 2: PREVIOUS_SECTION is een interne code (0-x)
    #            → trein komt van intern (vorige dwell of passing)
    #            → gebruik NEXT_SECTION + bxl_noord_out: vertrekkende lijn bepaalt platform
    #
    # Assumptie: elke spoorlijn heeft een vast toegewezen platform voor
    # aankomst en een vast platform voor vertrek.
    # -------------------------------------------------------------------------
    bxl_noord_in = {
        '50': 'platform 1', '36N': 'platform 3', '25N': 'platform 3',
        '161': 'platform 7', '161N': 'platform 7', '161-2': 'platform 7',
        '36': 'platform 9', '27': 'platform 9', '25': 'platform 11',
    }
    bxl_noord_out = {
        '50': 'platform 2', '36N': 'platform 4', '25N': 'platform 4',
        '161': 'platform 8', '161N': 'platform 8', '161-2': 'platform 8',
        '36': 'platform 10', '27': 'platform 10', '25': 'platform 12',
    }

    mask = (df['SOURCE'] == 'BRUSSEL-NOORD') & (df['TARGET'] == 'BRUSSEL-NOORD')
    prev_is_internal = df['PREVIOUS_SECTION'].str.startswith('0-', na=False)
    next_is_internal = df['NEXT_SECTION'].str.startswith('0-', na=False)

    df.loc[mask & ~prev_is_internal, 'PERRON'] = \
        df.loc[mask & ~prev_is_internal, 'PREVIOUS_SECTION'].map(bxl_noord_in)
    df.loc[mask & prev_is_internal & ~next_is_internal, 'PERRON'] = \
        df.loc[mask & prev_is_internal & ~next_is_internal, 'NEXT_SECTION'].map(bxl_noord_out)

    both_internal = mask & prev_is_internal & next_is_internal
    df.loc[both_internal, 'PERRON_GROEP'] = 'alle'
    _assign_tracks_by_overlap(
        df, both_internal, 'alle',
        ['platform 1', 'platform 2', 'platform 3', 'platform 4',
         'platform 5', 'platform 6', 'platform 7', 'platform 8',
         'platform 9', 'platform 10', 'platform 11', 'platform 12'],
        'BRUSSEL-NOORD', diagnostics
    )

    # -------------------------------------------------------------------------
    # BRUSSEL-CENTRAAL
    # 6 platforms op 3 eilandperrons op lijn 0 (Noord-Zuidverbinding).
    # PREVIOUS_SECTION en NEXT_SECTION bevatten altijd interne 0-x codes —
    # richtingsbepaling via lijncode is niet mogelijk omdat het station
    # volledig op de interne verbindingslijn ligt.
    # Alle 6 platforms worden als één pool behandeld via tijdsoverlap scheduling.
    # Assumptie: greedy interval scheduling geeft een realistische toewijzing
    # bij gebrek aan lijncode-informatie.
    # Bron: PerronAnalyse-Brussel.pdf
    # -------------------------------------------------------------------------
    mask_bc = (df['SOURCE'] == 'BRUSSEL-CENTRAAL') & (df['TARGET'] == 'BRUSSEL-CENTRAAL')
    df.loc[mask_bc, 'PERRON_GROEP'] = 'alle'
    _assign_tracks_by_overlap(
        df, mask_bc, 'alle',
        ['platform 1', 'platform 2', 'platform 3',
         'platform 4', 'platform 5', 'platform 6'],
        'BRUSSEL-CENTRAAL', diagnostics
    )

    # -------------------------------------------------------------------------
    # BRUSSEL-CONGRES
    # 4 platforms op 2 eilandperrons op lijn 0 (Noord-Zuidverbinding).
    # Zelfde situatie als Brussel-Centraal: PREVIOUS_SECTION en NEXT_SECTION
    # bevatten altijd interne 0-x codes — geen lijncode-informatie beschikbaar.
    # Alle 4 platforms worden als één pool behandeld via tijdsoverlap scheduling.
    # Assumptie: greedy interval scheduling geeft een realistische toewijzing.
    # Bron: PerronAnalyse-Brussel.pdf
    # -------------------------------------------------------------------------
    mask_bcong = (df['SOURCE'] == 'BRUSSEL-CONGRES') & (df['TARGET'] == 'BRUSSEL-CONGRES')
    df.loc[mask_bcong, 'PERRON_GROEP'] = 'alle'
    _assign_tracks_by_overlap(
        df, mask_bcong, 'alle',
        ['platform 1', 'platform 2', 'platform 3', 'platform 4'],
        'BRUSSEL-CONGRES', diagnostics
    )

    # -------------------------------------------------------------------------
    # BRUSSEL-KAPELLEKERK
    # 4 platforms op lijn 0 — uitsluitend S1-dienst (Nijvel ↔ Antwerpen).
    # Één lijn, twee richtingen — geen opsplitsing per richting nodig.
    # Alle 4 platforms als één pool via tijdsoverlap scheduling.
    # Bron: PerronAnalyse-Brussel.pdf, NMBS S1-dienstregeling
    # -------------------------------------------------------------------------
    mask_bkap = (df['SOURCE'] == 'BRUSSEL-KAPELLEKERK') & (df['TARGET'] == 'BRUSSEL-KAPELLEKERK')
    df.loc[mask_bkap, 'PERRON_GROEP'] = 'alle'
    _assign_tracks_by_overlap(
        df, mask_bkap, 'alle',
        ['platform 1', 'platform 2', 'platform 3', 'platform 4'],
        'BRUSSEL-KAPELLEKERK', diagnostics
    )

    # -------------------------------------------------------------------------
    # ANDERLECHT
    # 4 sporen op 2 zijsporen van spoorlijn 50C.
    # In de data komen treinen enkel aan via lijn 50C — geen 50A aanwezig.
    # Alle 4 platforms als één pool via tijdsoverlap scheduling.
    # Bron: Wikipedia — Station Anderlecht, NMBS dienstregeling
    # -------------------------------------------------------------------------
    mask_and = (df['SOURCE'] == 'ANDERLECHT') & (df['TARGET'] == 'ANDERLECHT')
    df.loc[mask_and, 'PERRON_GROEP'] = 'alle'
    _assign_tracks_by_overlap(
        df, mask_and, 'alle',
        ['platform 1', 'platform 2', 'platform 3', 'platform 4'],
        'ANDERLECHT', diagnostics
    )

    # -------------------------------------------------------------------------
    # BRUSSEL-ZUID
    # 22 platforms — RELATION_DIRECTION als proxy.
    # Bron: PerronAnalyse-Brussel.pdf
    # -------------------------------------------------------------------------
    mask = (df['SOURCE'] == 'BRUSSEL-ZUID') & (df['TARGET'] == 'BRUSSEL-ZUID')
    df.loc[mask, 'PERRON'] = df.loc[mask, 'RELATION_DIRECTION']

    # -------------------------------------------------------------------------
    # SCHAARBEEK
    # 13 platforms, variabele toewijzing — RELATION_DIRECTION als proxy.
    # Bron: PerronAnalyse-Brussel.pdf
    # -------------------------------------------------------------------------
    mask = (df['SOURCE'] == 'SCHAARBEEK') & (df['TARGET'] == 'SCHAARBEEK')
    df.loc[mask, 'PERRON'] = df.loc[mask, 'RELATION_DIRECTION']

    # -------------------------------------------------------------------------
    # JETTE
    # 4 platforms op basis van aankomende/vertrekkende spoorlijn.
    # PREVIOUS_STATION is onbruikbaar — altijd JETTE zelf (vorige dwell).
    # PREVIOUS_SECTION bevat lijncode 50 of 60.
    # Lijn 50: richting Zellik/Brussel-Noord → platform 1 aankomst, 3 vertrek
    # Lijn 60: richting Sint-Agatha-Berchem/Simonis → platform 2 aankomst, 4 vertrek
    # Assumptie: lijncode bepaalt rijrichting en platform.
    # -------------------------------------------------------------------------
    mask = (df['SOURCE'] == 'JETTE') & (df['TARGET'] == 'JETTE')
    prev_is_internal = df['PREVIOUS_SECTION'].str.startswith('0-', na=False)

    df.loc[mask, 'PERRON'] = np.where(
        prev_is_internal[mask],
        np.where(df.loc[mask, 'NEXT_SECTION'] == '50', 'platform 3',
        np.where(df.loc[mask, 'NEXT_SECTION'] == '60', 'platform 4',
        'platform 3')),
        np.where(df.loc[mask, 'PREVIOUS_SECTION'] == '50', 'platform 1',
        np.where(df.loc[mask, 'PREVIOUS_SECTION'] == '60', 'platform 2',
        'platform 1'))
    )

    # -------------------------------------------------------------------------
    # Overige stations
    # -------------------------------------------------------------------------
    mask = (df['SOURCE'] == 'BOCKSTAEL') & (df['TARGET'] == 'BOCKSTAEL')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'JETTE', 'platform 1', 'platform 2'
    )

    mask = (df['SOURCE'] == 'BRUSSEL-SCHUMAN') & (df['TARGET'] == 'BRUSSEL-SCHUMAN')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'BRUSSEL-NOORD', 'platform 1',
        np.where(df.loc[mask, 'PREVIOUS_STATION'] == 'BRUSSEL-LUXEMBURG', 'platform 2',
        np.where(df.loc[mask, 'PREVIOUS_STATION'] == 'BOCKSTAEL', 'platform 3',
        'platform 4'))
    )

    mask = (df['SOURCE'] == 'BRUSSEL-WEST') & (df['TARGET'] == 'BRUSSEL-WEST')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'BRUSSEL-ZUID', 'platform 1', 'platform 2'
    )

    mask = (df['SOURCE'] == 'SIMONIS') & (df['TARGET'] == 'SIMONIS')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'BRUSSEL-WEST', 'platform 1', 'platform 2'
    )

    mask = (df['SOURCE'] == 'THURN EN TAXIS') & (df['TARGET'] == 'THURN EN TAXIS')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'SIMONIS', 'platform 1', 'platform 2'
    )

    mask = (df['SOURCE'] == 'VORST-OOST') & (df['TARGET'] == 'VORST-OOST')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'UKKEL-STALLE', 'platform 1', 'platform 2'
    )

    mask = (df['SOURCE'] == 'ZELLIK') & (df['TARGET'] == 'ZELLIK')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'JETTE', 'platform 2', 'platform 1'
    )

    mask = (df['SOURCE'] == 'SINT-AGATHA-BERCHEM') & (df['TARGET'] == 'SINT-AGATHA-BERCHEM')
    df.loc[mask, 'PERRON'] = np.where(
        df.loc[mask, 'PREVIOUS_STATION'] == 'JETTE', 'platform 2', 'platform 1'
    )

    df['SECTION_MACRO'] = df['SECTION']
    df['SECTION'] = np.where(
        df['PERRON'].notna(),
        df['SECTION'] + ' -- ' + df['PERRON'],
        df['SECTION'],
    )

    logger.info(diagnostics.summary())

    return df.drop(
        columns=[
            'PREVIOUS_SECTION', 'NEXT_SECTION',
            'PREVIOUS_STATION', 'NEXT_STATION',
            'PERRON', 'PERRON_GROEP',
        ],
        errors='ignore',
    ), diagnostics


# =============================================================================
# Platform-alternatieven voor retracking
# =============================================================================

# Stations waarvan de platforms NIET als vrije pool worden behandeld.
# Brussel-Noord: lijn-gemapte platforms (lijn 50 → platform 1, enz.) — geen vrije keuze.
_RETRACK_EXCLUDED: set[str] = {"BRUSSEL-NOORD"}


def get_platform_alternatives(
    df: pd.DataFrame,
    excluded: set[str] | None = None,
) -> dict[tuple[int, str], list[str]]:
    """
    Scant de combined timetable op stations met '-- platform X' segmenten
    en retourneert voor elke (train_id, planned_segment) de alternatieven.

    Alleen stations waarbij assign_platforms alle platforms als één pool
    behandelt (greedy interval scheduling) hebben zinvolle alternatieven.
    Stations met richtings-specifieke segmentnamen (Brussel-Zuid, Schaarbeek)
    bevatten geen '-- platform X' segmenten en vallen automatisch buiten scope.

    Parameters
    ----------
    df       : combined timetable DataFrame (output van combine_timetables)
    excluded : set van stationnamen die uitgesloten worden; default = _RETRACK_EXCLUDED

    Returns
    -------
    dict {(train_id, planned_segment_id): [alt_segment_id, ...]}
    planned_segment_id is NIET opgenomen in de lijst van alternatieven.
    """
    import re

    if excluded is None:
        excluded = _RETRACK_EXCLUDED

    pat = re.compile(r'^(.+?) -- platform \d')

    # Verzamel alle platform-X segmenten per station
    station_pools: dict[str, set[str]] = {}
    station_mask = df['TYPE'].isin(['WITHIN-STATION-DWELL', 'WITHIN-STATION-PASSING'])
    for seg in df.loc[station_mask, 'SECTION'].unique():
        m = pat.match(seg)
        if m:
            station = m.group(1)
            if station not in excluded:
                station_pools.setdefault(station, set()).add(seg)

    # Stations met slechts één segment hebben geen alternatief
    station_pools = {s: pool for s, pool in station_pools.items() if len(pool) > 1}

    # Voor elke (train, station-segment) in scope: alternatieven = pool minus gepland
    result: dict[tuple[int, str], list[str]] = {}
    for _, row in df.loc[station_mask].iterrows():
        seg = row['SECTION']
        m = pat.match(seg)
        if not m:
            continue
        station = m.group(1)
        if station not in station_pools:
            continue
        alts = sorted(station_pools[station] - {seg})
        if alts:
            result[(int(row['TRAIN_NO']), seg)] = alts

    return result


# =============================================================================
# Opslaan / laden
# =============================================================================

def save_gold(df: pd.DataFrame, source: str = 'passenger', n_trains: int | None = None) -> None:
    """Slaat de finale timetable op als parquet."""
    if source not in ('passenger', 'freight', 'combined'):
        raise ValueError(f"Onbekende source '{source}'.")
    if source in ('freight', 'combined') and n_trains is None:
        raise ValueError(f"n_trains is verplicht voor source='{source}'")

    if source == 'passenger':
        source_dir = GOLD_DIR / 'passenger'
        filename   = 'planned_timetable.parquet'
    elif source == 'freight':
        source_dir = GOLD_DIR / 'freight' / str(n_trains)
        filename   = 'freight_timetable.parquet'
    else:
        source_dir = GOLD_DIR / 'combined' / str(n_trains)
        filename   = 'combined_timetable.parquet'

    source_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(source_dir / filename, index=False)
    print(f"Opgeslagen ({source}): {df['TRAIN_NO'].nunique()} treinen, {len(df)} segmenten → {source_dir}")


def load_gold(source: str = 'passenger', n_trains: int | None = None) -> pd.DataFrame:
    """Laadt een opgeslagen gold timetable."""
    if source not in ('passenger', 'freight', 'combined'):
        raise ValueError(f"Onbekende source '{source}'.")
    if source in ('freight', 'combined') and n_trains is None:
        raise ValueError(f"n_trains is verplicht voor source='{source}'")

    if source == 'passenger':
        path = GOLD_DIR / 'passenger' / 'planned_timetable.parquet'
    elif source == 'freight':
        path = GOLD_DIR / 'freight' / str(n_trains) / 'freight_timetable.parquet'
    else:
        path = GOLD_DIR / 'combined' / str(n_trains) / 'combined_timetable.parquet'

    df = pd.read_parquet(path)
    logger.info(f"Gold geladen ({source}): {df['TRAIN_NO'].nunique()} treinen, {len(df)} segmenten")
    return df