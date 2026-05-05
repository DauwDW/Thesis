import logging

import numpy as np
import pandas as pd

from config.settings import PASSING_DURATION_FREIGHT

logger = logging.getLogger(__name__)

# =============================================================================
# Traject definitie
# =============================================================================

TRAJECTEN: dict[str, list[str]] = {
    "T1a": ["BRUSSEL-NOORD", "SCHAARBEEK"],
    "T1b": ["SCHAARBEEK", "BRUSSEL-NOORD"],
    "T2a": ["BRUSSEL-NOORD", "SCHAARBEEK", "THURN EN TAXIS", "SIMONIS", "BRUSSEL-WEST", "BRUSSEL-ZUID"],
    "T2b": ["BRUSSEL-ZUID", "BRUSSEL-WEST", "SIMONIS", "THURN EN TAXIS", "SCHAARBEEK", "BRUSSEL-NOORD"],
    "T3a": ["BRUSSEL-ZUID", "ANDERLECHT"],
    "T3b": ["ANDERLECHT", "BRUSSEL-ZUID"],
    "T4a": ["SCHAARBEEK", "BRUSSEL-SCHUMAN"],
    "T4b": ["BRUSSEL-SCHUMAN", "SCHAARBEEK"],
    "T5a": ["BRUSSEL-NOORD", "BOCKSTAEL", "JETTE", "ZELLIK"],
    "T5b": ["ZELLIK", "JETTE", "BOCKSTAEL", "BRUSSEL-NOORD"],
}

# =============================================================================
# Periodes
# =============================================================================

PERIOD_RANGES: dict[str, tuple[int, int]] = {
    "NIGHT":        (0,  6),
    "MORNING PEAK": (6,  9),
    "DAYTIME":      (9,  16),
    "EVENING PEAK": (16, 19),
    "EVENING":      (19, 24),
}

DEFAULT_PERIOD_PROBS: dict[str, float] = {
    "NIGHT":        0.40,
    "MORNING PEAK": 0.05,
    "DAYTIME":      0.35,
    "EVENING PEAK": 0.05,
    "EVENING":      0.15,
}

BASE_DATE = pd.Timestamp("2025-01-01")

# =============================================================================
# Fallback rijtijden en lijncodes voor secties zonder passenger data
# =============================================================================

_FALLBACK_PASSENGER_RUNNING_TIMES: dict[tuple[str, str], float] = {
    ('SCHAARBEEK',      'THURN EN TAXIS'):  318,
    ('THURN EN TAXIS',  'SCHAARBEEK'):      318,
    ('SCHAARBEEK',      'BRUSSEL-SCHUMAN'): 431,
    ('BRUSSEL-SCHUMAN', 'SCHAARBEEK'):      431,
}

_FALLBACK_LINE_NOS: dict[tuple[str, str], str] = {
    ('SCHAARBEEK',      'THURN EN TAXIS'):  '28',
    ('THURN EN TAXIS',  'SCHAARBEEK'):      '28',
    ('SCHAARBEEK',      'BRUSSEL-SCHUMAN'): '26',
    ('BRUSSEL-SCHUMAN', 'SCHAARBEEK'):      '26',
}

# =============================================================================
# Lookups uit passenger data
# =============================================================================

def build_line_no_lookup(passenger_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    """
    Leidt per (SOURCE, TARGET) de meest voorkomende lijncode af uit passenger SECTION.
    Secties zonder passenger data worden aangevuld via _FALLBACK_LINE_NOS.
    """
    between = passenger_df[passenger_df['SOURCE'] != passenger_df['TARGET']].copy()
    parsed = between['SECTION'].str.extract(r'^(?P<LINE>[^:]+):')
    parsed['SOURCE'] = between['SOURCE'].values
    parsed['TARGET'] = between['TARGET'].values

    return (
        parsed.groupby(['SOURCE', 'TARGET'])['LINE']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    ) | _FALLBACK_LINE_NOS


def build_running_time_lookup(passenger_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """
    Berekent de mediane geplande rijtijd (seconden) per (SOURCE, TARGET) uit passenger data.
    Freight running time = mediane passenger rijtijd x 1.3.
    """
    between = passenger_df[passenger_df['SOURCE'] != passenger_df['TARGET']].copy()
    between['PLANNED_ENTRY'] = pd.to_datetime(between['PLANNED_ENTRY'])
    between['PLANNED_EXIT']  = pd.to_datetime(between['PLANNED_EXIT'])
    between['RUNNING_TIME_SEC'] = (
        between['PLANNED_EXIT'] - between['PLANNED_ENTRY']
    ).dt.total_seconds()

    between = between[between['RUNNING_TIME_SEC'] > 0]

    return (
        between.groupby(['SOURCE', 'TARGET'])['RUNNING_TIME_SEC']
        .median()
        .mul(1.3)
        .to_dict()
    ) | {k: v * 1.3 for k, v in _FALLBACK_PASSENGER_RUNNING_TIMES.items()}


# =============================================================================
# Normalisatie passenger tijden naar basisdatum
# =============================================================================

def normalize_to_base_date(passenger_df: pd.DataFrame) -> pd.DataFrame:
    df = passenger_df.copy()
    df['PLANNED_ENTRY'] = pd.to_datetime(df['PLANNED_ENTRY'])
    df['PLANNED_EXIT']  = pd.to_datetime(df['PLANNED_EXIT'])

    first_time = df.groupby('TRAIN_NO')['PLANNED_ENTRY'].transform('min')
    first_date = first_time.dt.normalize()
    offset = BASE_DATE - first_date

    df['PLANNED_ENTRY'] = df['PLANNED_ENTRY'] + offset
    df['PLANNED_EXIT']  = df['PLANNED_EXIT']  + offset

    return df


# =============================================================================
# Interne hulpfuncties
# =============================================================================

def _sample_time_in_period(rng: np.random.Generator, period: str) -> pd.Timestamp:
    start, end = PERIOD_RANGES[period]
    return BASE_DATE + pd.Timedelta(hours=float(rng.uniform(start, end)))


def _get_period_end(period: str) -> pd.Timestamp:
    _, end = PERIOD_RANGES[period]
    return BASE_DATE + pd.Timedelta(hours=end)


def _build_segments(stations: list[str]) -> list[tuple[str, str]]:
    return [(stations[i], stations[i + 1]) for i in range(len(stations) - 1)]


def _normalize_probs(probs: dict) -> tuple[list, np.ndarray]:
    keys = list(probs.keys())
    p = np.array(list(probs.values()), dtype=float)
    p /= p.sum()
    return keys, p


def _shift_until_feasible(
    section:      str,
    desired_time: pd.Timestamp,
    duration_sec: float,
    period:       str,
    occupied:     dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    max_iter:     int,
) -> tuple[pd.Timestamp, bool]:
    """
    Verschuift een vertrektijdstip totdat het interval [entry, exit)
    niet overlapt met bestaande intervallen op dezelfde sectie.
    """
    period_end = _get_period_end(period)
    time = desired_time

    for _ in range(max_iter):
        exit_time = time + pd.Timedelta(seconds=duration_sec)

        if exit_time > period_end:
            return time, False

        conflict_end = None
        for entry, exit_ in sorted(occupied.get(section, [])):
            if time < exit_ and exit_time > entry:
                conflict_end = exit_
                break

        if conflict_end is None:
            return time, True

        time = conflict_end

    return time, False


# =============================================================================
# Main generator
# =============================================================================

def generate_freight_timetable(
    passenger_df:    pd.DataFrame,
    n_trains:        int,
    traject_probs:   dict[str, float],
    period_probs:    dict[str, float] | None = None,
    seed:            int = 42,
    base_train_no:   int = 900000,
    max_iter:        int = 20,
) -> pd.DataFrame:
    """
    Genereert een synthetische freight timetable voor de Brusselse corridor.

    Returns:
        DataFrame met freight timetable, compatibel met passenger pipeline.
        Kolommen: TRAIN_NO, SOURCE, TARGET, SECTION, PLANNED_ENTRY,
                  PLANNED_EXIT, RELATION_DIRECTION, TYPE
    """
    if period_probs is None:
        period_probs = DEFAULT_PERIOD_PROBS

    rng = np.random.default_rng(seed)

    passenger_norm = normalize_to_base_date(passenger_df)

    line_no_lookup      = build_line_no_lookup(passenger_norm)
    running_time_lookup = build_running_time_lookup(passenger_norm)

    traject_keys, traject_p = _normalize_probs(traject_probs)
    period_keys,  period_p  = _normalize_probs(period_probs)

    occupied: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    passenger_between = passenger_norm[passenger_norm['SOURCE'] != passenger_norm['TARGET']]
    for _, row in passenger_between.iterrows():
        occupied.setdefault(row['SECTION'], []).append(
            (row['PLANNED_ENTRY'], row['PLANNED_EXIT'])
        )

    rows = []
    n_infeasible = 0

    for i in range(n_trains):
        train_id    = base_train_no + i
        traject_key = rng.choice(traject_keys, p=traject_p)
        stations    = TRAJECTEN[traject_key]
        period      = rng.choice(period_keys, p=period_p)
        segments    = _build_segments(stations)

        first_source, first_target = segments[0]
        first_line_no = line_no_lookup.get((first_source, first_target))
        if first_line_no is None:
            logger.warning(f"Trein {train_id}: geen lijncode voor ({first_source}, {first_target}) — overgeslagen")
            continue

        first_duration = running_time_lookup.get((first_source, first_target))
        if first_duration is None:
            logger.warning(f"Trein {train_id}: geen rijtijd voor ({first_source}, {first_target}) — overgeslagen")
            continue

        first_section = f"{first_line_no}:{first_source}-{first_target}"
        desired_time  = _sample_time_in_period(rng, period)

        departure, feasible = _shift_until_feasible(
            first_section, desired_time, first_duration, period, occupied, max_iter
        )

        if not feasible:
            n_infeasible += 1
            logger.warning(
                f"Trein {train_id} ({traject_key}): geen conflict-vrij vertrekslot op "
                f"{first_section} binnen periode {period} — ingepland met conflict"
            )

        arrival = departure + pd.Timedelta(seconds=first_duration)
        occupied.setdefault(first_section, []).append((departure, arrival))

        current_time = departure

        for seg_idx, (source, target) in enumerate(segments):
            line_no = line_no_lookup.get((source, target))
            if line_no is None:
                logger.warning(f"Trein {train_id}: geen lijncode voor ({source}, {target}) — traject afgebroken")
                break

            duration_sec = running_time_lookup.get((source, target))
            if duration_sec is None:
                logger.warning(f"Trein {train_id}: geen rijtijd voor ({source}, {target}) — traject afgebroken")
                break

            section = f"{line_no}:{source}-{target}"
            arrival = current_time + pd.Timedelta(seconds=duration_sec)

            rows.append({
                "TRAIN_NO":           train_id,
                "SOURCE":             source,
                "TARGET":             target,
                "SECTION":            section,
                "PLANNED_ENTRY":      current_time,
                "PLANNED_EXIT":       arrival,
                "RELATION_DIRECTION": traject_key,
                "TYPE":               "BETWEEN-STATION",
            })
            current_time = arrival

            # --- Passing-segment (niet voor eindstation) ---
            if seg_idx < len(segments) - 1:
                passing_exit = current_time + pd.Timedelta(seconds=PASSING_DURATION_FREIGHT)
                rows.append({
                    "TRAIN_NO":           train_id,
                    "SOURCE":             target,
                    "TARGET":             target,
                    "SECTION":            target,
                    "PLANNED_ENTRY":      current_time,   # aankomsttijd op station
                    "PLANNED_EXIT":       passing_exit,   # vertrektijd uit station
                    "RELATION_DIRECTION": traject_key,
                    "TYPE":               "WITHIN-STATION-PASSING",
                })
                current_time = passing_exit

    if n_infeasible > 0:
        logger.warning(f"Freight timetable: {n_infeasible} trein(en) ingepland met conflict op vertrekslot")

    logger.info(
        f"Freight timetable gegenereerd: {n_trains} treinen, "
        f"{len(rows)} segmenten, {n_infeasible} conflict(en) op vertrekslot"
    )

    return pd.DataFrame(rows)


def add_freight_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt DYNAMICS kolom toe aan freight timetable.
    Alle freight segmenten krijgen '0-0'.
    """
    df = df.copy()
    df['DYNAMICS'] = '0-0'
    return df