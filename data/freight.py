import logging

import numpy as np
import pandas as pd

from config.settings import PASSING_DURATION_FREIGHT, FREIGHT_RUNNING_TIME_SCALE, FREIGHT_POOL_TYPES

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
# Peak-onderdrukte verdelingen voor de versoepelingstrap.
# Level 1 = milde peak-onderdrukking, level 2 = peak ≈ 0.
_PEAK_SUPPRESSED_PROBS: dict[int, dict[str, float]] = {
    1: {
        "NIGHT":        0.45,
        "MORNING PEAK": 0.02,
        "DAYTIME":      0.38,
        "EVENING PEAK": 0.02,
        "EVENING":      0.13,
    },
    2: {
        "NIGHT":        0.48,
        "MORNING PEAK": 0.005,
        "DAYTIME":      0.40,
        "EVENING PEAK": 0.005,
        "EVENING":      0.11,
    },
}

# Versoepelingstrap: per stap een (buffer, peak-level) combinatie.
# Eerst peak-massa verschuiven (level 0 → 1 → 2), dan pas de buffer verkleinen.
DEFAULT_RELAXATION_BUFFERS_SEC: tuple[int, ...] = (120, 60, 0)
DEFAULT_RELAXATION_LEVELS:      tuple[int, ...] = (0, 1, 2)


BASE_DATE = pd.Timestamp("2025-01-01")

# =============================================================================
# Fallback voor freight-only goederenlijnen zonder passenger data
#
# Lijnen 26 en 28 zijn freight-only — er rijden per definitie geen
# passagierstreinen, dus geen data-gedreven mediaan mogelijk. Schatting
# komt uit infrastructuurkennis (lengte + maximumsnelheid voor freight).
# =============================================================================

_FALLBACK_LINE_NOS: dict[tuple[str, str], str] = {
    ('SCHAARBEEK',      'THURN EN TAXIS'):  '28',
    ('THURN EN TAXIS',  'SCHAARBEEK'):      '28',
    ('SCHAARBEEK',      'BRUSSEL-SCHUMAN'): '26',
    ('BRUSSEL-SCHUMAN', 'SCHAARBEEK'):      '26',
}

# Rijtijden in seconden voor freight-only lijnen — al inclusief freight-snelheid,
# wordt NIET met FREIGHT_RUNNING_TIME_SCALE vermenigvuldigd in de lookup.
_FREIGHT_ONLY_RUNNING_TIMES: dict[str, float] = {
    '28:SCHAARBEEK-THURN EN TAXIS':  318 *FREIGHT_RUNNING_TIME_SCALE,
    '28:THURN EN TAXIS-SCHAARBEEK':  318 *FREIGHT_RUNNING_TIME_SCALE,
    '26:SCHAARBEEK-BRUSSEL-SCHUMAN': 431 *FREIGHT_RUNNING_TIME_SCALE,
    '26:BRUSSEL-SCHUMAN-SCHAARBEEK': 431 *FREIGHT_RUNNING_TIME_SCALE,
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


def _hour_to_period(hour: int) -> str:
    """Zet een uur (0-23) om naar een dagperiode-label."""
    if hour < 6:  return "NIGHT"
    if hour < 9:  return "MORNING PEAK"
    if hour < 16: return "DAYTIME"
    if hour < 19: return "EVENING PEAK"
    return "EVENING"


def build_running_time_lookup(passenger_df: pd.DataFrame) -> dict:
    """
    Berekent mediane geplande rijtijden (seconden × FREIGHT_RUNNING_TIME_SCALE),
    gegroepeerd per SECTION (lijncode + stations) zoals in reality/sampling.py.

    Poolt enkel over FREIGHT_POOL_TYPES (IC, L, S) — geen Eurostar/ICE/INT.

    Fallback-hiërarchie:
      l1: (SECTION, PERIOD)         — specifieke lijn + periode
      l2: (SECTION)                 — specifieke lijn, alle periodes
      l3: (SOURCE, TARGET, PERIOD)  — alle lijnen, specifieke periode
      l4: (SOURCE, TARGET)          — alle lijnen, alle periodes
    """
    between = passenger_df[passenger_df['SOURCE'] != passenger_df['TARGET']].copy()

    # Filter op trage treintypen — consistent met FREIGHT_POOL_TYPES in sampling.py
    if 'TRAIN_TYPE' in between.columns:
        between = between[between['TRAIN_TYPE'].isin(FREIGHT_POOL_TYPES)]

    between['PLANNED_ENTRY'] = pd.to_datetime(between['PLANNED_ENTRY'])
    between['PLANNED_EXIT']  = pd.to_datetime(between['PLANNED_EXIT'])
    between['RUNNING_TIME_SEC'] = (
        between['PLANNED_EXIT'] - between['PLANNED_ENTRY']
    ).dt.total_seconds()
    between = between[between['RUNNING_TIME_SEC'] > 0]

    # Periode afleiden uit entry-uur
    between['PERIOD'] = between['PLANNED_ENTRY'].dt.hour.apply(_hour_to_period)

    # Niveau 1: (SECTION, PERIOD) — meest specifiek
    l1 = (
        between.groupby(['SECTION', 'PERIOD'])['RUNNING_TIME_SEC']
        .median()
        .mul(FREIGHT_RUNNING_TIME_SCALE)
        .to_dict()
    )

    # Niveau 2: (SECTION) — alle periodes gepoold
    l2 = (
        between.groupby('SECTION')['RUNNING_TIME_SEC']
        .median()
        .mul(FREIGHT_RUNNING_TIME_SCALE)
        .to_dict()
    )

    # Niveau 3: (SOURCE, TARGET, PERIOD) — andere lijnen tussen zelfde stations
    l3 = (
        between.groupby(['SOURCE', 'TARGET', 'PERIOD'])['RUNNING_TIME_SEC']
        .median()
        .mul(FREIGHT_RUNNING_TIME_SCALE)
        .to_dict()
    )

    # Niveau 4: (SOURCE, TARGET) — laatste vangnet
    l4 = (
        between.groupby(['SOURCE', 'TARGET'])['RUNNING_TIME_SEC']
        .median()
        .mul(FREIGHT_RUNNING_TIME_SCALE)
        .to_dict()
    )

    return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}


def get_running_time(
    lookup:  dict,
    section: str,
    source:  str,
    target:  str,
    period:  str | None = None,
) -> float | None:
    """
    Zoekt de freight rijtijd op via de fallback-hiërarchie:
      1. (SECTION, PERIOD)                 — data-gedreven
      2. (SECTION)                         — data-gedreven
      3. (SOURCE, TARGET, PERIOD)          — data-gedreven
      4. (SOURCE, TARGET)                  — data-gedreven
      5. _FREIGHT_ONLY_RUNNING_TIMES       — hardcoded, enkel freight-only lijnen
      6. None — niets gevonden, error gelogd

    Parameters
    ----------
    period : str | None
        Dagperiode (bv. "NIGHT"). Indien None worden l1 en l3 overgeslagen.
    """
    if period is not None:
        v = lookup["l1"].get((section, period))
        if v is not None:
            return v

    v = lookup["l2"].get(section)
    if v is not None:
        return v

    if period is not None:
        v = lookup["l3"].get((source, target, period))
        if v is not None:
            return v

    v = lookup["l4"].get((source, target))
    if v is not None:
        return v

    # Freight-only lijnen: geen passenger data mogelijk, schatting uit infrastructuur
    v = _FREIGHT_ONLY_RUNNING_TIMES.get(section)
    if v is not None:
        return v

    logger.error(
        f"Geen rijtijd gevonden voor section='{section}', "
        f"({source} → {target}), period={period} — geen fallback beschikbaar"
    )
    return None


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
    buffer_sec:   float = 0.0,
) -> tuple[pd.Timestamp, bool]:
    """
    Verschuift een vertrektijdstip totdat het interval [entry, exit)
    minstens `buffer_sec` verwijderd is van bestaande intervallen op
    dezelfde sectie. De buffer wordt symmetrisch toegepast (headway).
    """
    period_end = _get_period_end(period)
    buffer_td  = pd.Timedelta(seconds=buffer_sec)
    time = desired_time

    for _ in range(max_iter):
        exit_time = time + pd.Timedelta(seconds=duration_sec)

        if exit_time > period_end:
            return time, False

        conflict_end = None
        for entry, exit_ in sorted(occupied.get(section, [])):
            # if time < exit_ and exit_time > entry:
            #     conflict_end = exit_
            #!!!
            if time < exit_ + buffer_td and exit_time > entry - buffer_td:
                conflict_end = exit_ + buffer_td

                break

        if conflict_end is None:
            return time, True

        time = conflict_end

    return time, False

#!!!

def _build_relaxation_stages(
    base_period_probs: dict[str, float],
    buffers_sec:       tuple[int, ...] = DEFAULT_RELAXATION_BUFFERS_SEC,
    peak_levels:       tuple[int, ...] = DEFAULT_RELAXATION_LEVELS,
) -> list[dict]:
    """
    Bouwt de versoepelingstrap. Per stap: een buffer (in seconden) en
    een periodeverdeling (level 0 = base, 1 = milde peak-onderdrukking,
    2 = peak ≈ 0). Volgorde: eerst alle peak-levels bij de grootste buffer,
    pas dan de buffer verkleinen.
    """
    stages: list[dict] = []
    for buf in buffers_sec:
        for level in peak_levels:
            probs = base_period_probs if level == 0 else _PEAK_SUPPRESSED_PROBS[level]
            stages.append({
                "buffer_sec":   buf,
                "peak_level":   level,
                "period_probs": probs,
            })
    return stages


def _format_probs(probs: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in probs.items())
# =============================================================================
# Main generator
# =============================================================================

# def generate_freight_timetable(
#     passenger_df:    pd.DataFrame,
#     n_trains:        int,
#     traject_probs:   dict[str, float],
#     period_probs:    dict[str, float] | None = None,
#     seed:            int = 42,
#     base_train_no:   int = 900000,
#     max_iter:        int = 20,
# ) -> pd.DataFrame:
#     """
#     Genereert een synthetische freight timetable voor de Brusselse corridor.

#     Returns:
#         DataFrame met freight timetable, compatibel met passenger pipeline.
#         Kolommen: TRAIN_NO, SOURCE, TARGET, SECTION, PLANNED_ENTRY,
#                   PLANNED_EXIT, RELATION_DIRECTION, TYPE
#     """
#     if period_probs is None:
#         period_probs = DEFAULT_PERIOD_PROBS

#     rng = np.random.default_rng(seed)

#     passenger_norm = normalize_to_base_date(passenger_df)

#     line_no_lookup      = build_line_no_lookup(passenger_norm)
#     running_time_lookup = build_running_time_lookup(passenger_norm)

#     traject_keys, traject_p = _normalize_probs(traject_probs)
#     period_keys,  period_p  = _normalize_probs(period_probs)

#     occupied: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
#     passenger_between = passenger_norm[passenger_norm['SOURCE'] != passenger_norm['TARGET']]
#     for _, row in passenger_between.iterrows():
#         occupied.setdefault(row['SECTION'], []).append(
#             (row['PLANNED_ENTRY'], row['PLANNED_EXIT'])
#         )

#     rows = []
#     n_infeasible = 0

#     for i in range(n_trains):
#         train_id    = base_train_no + i
#         traject_key = rng.choice(traject_keys, p=traject_p)
#         stations    = TRAJECTEN[traject_key]
#         period      = rng.choice(period_keys, p=period_p)
#         segments    = _build_segments(stations)

#         first_source, first_target = segments[0]
#         first_line_no = line_no_lookup.get((first_source, first_target))
#         if first_line_no is None:
#             logger.warning(f"Trein {train_id}: geen lijncode voor ({first_source}, {first_target}) — overgeslagen")
#             continue

#         first_duration = running_time_lookup.get((first_source, first_target))
#         if first_duration is None:
#             logger.warning(f"Trein {train_id}: geen rijtijd voor ({first_source}, {first_target}) — overgeslagen")
#             continue

#         first_section = f"{first_line_no}:{first_source}-{first_target}"
#         desired_time  = _sample_time_in_period(rng, period)

#         departure, feasible = _shift_until_feasible(
#             first_section, desired_time, first_duration, period, occupied, max_iter
#         )

#         if not feasible:
#             n_infeasible += 1
#             logger.warning(
#                 f"Trein {train_id} ({traject_key}): geen conflict-vrij vertrekslot op "
#                 f"{first_section} binnen periode {period} — ingepland met conflict"
#             )

#         arrival = departure + pd.Timedelta(seconds=first_duration)
#         occupied.setdefault(first_section, []).append((departure, arrival))

#         current_time = departure

#         for seg_idx, (source, target) in enumerate(segments):
#             line_no = line_no_lookup.get((source, target))
#             if line_no is None:
#                 logger.warning(f"Trein {train_id}: geen lijncode voor ({source}, {target}) — traject afgebroken")
#                 break

#             duration_sec = running_time_lookup.get((source, target))
#             if duration_sec is None:
#                 logger.warning(f"Trein {train_id}: geen rijtijd voor ({source}, {target}) — traject afgebroken")
#                 break

#             section = f"{line_no}:{source}-{target}"
#             arrival = current_time + pd.Timedelta(seconds=duration_sec)

#             rows.append({
#                 "TRAIN_NO":           train_id,
#                 "SOURCE":             source,
#                 "TARGET":             target,
#                 "SECTION":            section,
#                 "PLANNED_ENTRY":      current_time,
#                 "PLANNED_EXIT":       arrival,
#                 "RELATION_DIRECTION": traject_key,
#                 "TYPE":               "BETWEEN-STATION",
#             })
#             current_time = arrival

#             # --- Passing-segment (niet voor eindstation) ---
#             if seg_idx < len(segments) - 1:
#                 passing_exit = current_time + pd.Timedelta(seconds=PASSING_DURATION_FREIGHT)
#                 rows.append({
#                     "TRAIN_NO":           train_id,
#                     "SOURCE":             target,
#                     "TARGET":             target,
#                     "SECTION":            target,
#                     "PLANNED_ENTRY":      current_time,   # aankomsttijd op station
#                     "PLANNED_EXIT":       passing_exit,   # vertrektijd uit station
#                     "RELATION_DIRECTION": traject_key,
#                     "TYPE":               "WITHIN-STATION-PASSING",
#                 })
#                 current_time = passing_exit

#     if n_infeasible > 0:
#         logger.warning(f"Freight timetable: {n_infeasible} trein(en) ingepland met conflict op vertrekslot")

#     logger.info(
#         f"Freight timetable gegenereerd: {n_trains} treinen, "
#         f"{len(rows)} segmenten, {n_infeasible} conflict(en) op vertrekslot"
#     )

#     return pd.DataFrame(rows)


def generate_freight_timetable(
    passenger_df:         pd.DataFrame,
    n_trains:             int,
    traject_probs:        dict[str, float],
    period_probs:         dict[str, float] | None = None,
    seed:                 int = 42,
    base_train_no:        int = 900000,
    max_iter:             int = 20,
    n_attempts_per_stage: int = 5,
    relaxation_buffers_sec: tuple[int, ...] = DEFAULT_RELAXATION_BUFFERS_SEC,
    relaxation_peak_levels: tuple[int, ...] = DEFAULT_RELAXATION_LEVELS,
) -> pd.DataFrame:
    """
    Genereert een synthetische freight timetable voor de Brusselse corridor.
    
    Plaatsing gebeurt globaal via een versoepelingstrap: alle treinen worden
    eerst aangeboden met initiële buffer en originele periodeverdeling; wat
    niet past, schuift naar een strengere off-peak verdeling en uiteindelijk
    naar een kleinere buffer. Treinen die na alle stappen niet passen worden
    gedropt (niet ingepland met conflict).


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

    occupied: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    passenger_between = passenger_norm[passenger_norm['SOURCE'] != passenger_norm['TARGET']]
    for _, row in passenger_between.iterrows():
        occupied.setdefault(row['SECTION'], []).append(
            (row['PLANNED_ENTRY'], row['PLANNED_EXIT'])
        )

    # --- Bouw lijst van aan-te-plannen treinen (traject vast per trein) ---
    pending: list[dict] = []
    n_skipped_lookup = 0


    for i in range(n_trains):
        train_id    = base_train_no + i
        traject_key = rng.choice(traject_keys, p=traject_p)
        stations    = TRAJECTEN[traject_key]
        segments    = _build_segments(stations)

        first_source, first_target = segments[0]
        first_line_no  = line_no_lookup.get((first_source, first_target))

        if first_line_no is None:
            logger.warning(
                f"Trein {train_id}: geen lijncode voor ({first_source}, {first_target}) — overgeslagen"
            )
            n_skipped_lookup += 1
            continue

        first_section = f"{first_line_no}:{first_source}-{first_target}"
        # Periode nog onbekend bij pending-check → l1/l3 worden overgeslagen
        first_duration = get_running_time(
            running_time_lookup, first_section, first_source, first_target
        )

        if first_duration is None:
            logger.warning(
                f"Trein {train_id}: geen rijtijd voor {first_section} — overgeslagen"
            )
            n_skipped_lookup += 1
            continue

        pending.append({
            "train_id":      train_id,
            "traject_key":   traject_key,
            "segments":      segments,
            "first_section": first_section,
            "first_duration": first_duration,
        })
            # --- Globale plaatsing via versoepelingstrap ---
    stages = _build_relaxation_stages(
        period_probs,
        buffers_sec=relaxation_buffers_sec,
        peak_levels=relaxation_peak_levels,
    )

    logger.info(
        f"Freight scheduling: {len(pending)} treinen te plaatsen via "
        f"{len(stages)} versoepelingsstappen ({n_attempts_per_stage} samples/stap)"
    )

    placed:        list[dict] = []
    stage_summary: list[dict] = []

    for stage_idx, stage in enumerate(stages, start=1):
        if not pending:
            break

        buffer_sec     = stage["buffer_sec"]
        stage_probs    = stage["period_probs"]
        period_keys, period_p = _normalize_probs(stage_probs)

        logger.info(
            f"[freight stage {stage_idx}/{len(stages)}] start: "
            f"{len(pending)} pending, buffer={buffer_sec}s, "
            f"peak_level={stage['peak_level']}, "
            f"periodeverdeling=[{_format_probs(stage_probs)}]"
        )
        still_pending: list[dict] = []
        n_placed_this_stage = 0
        for train in pending:
            scheduled = False
            first_source, first_target = train["segments"][0]
            for _ in range(n_attempts_per_stage):
                period       = rng.choice(period_keys, p=period_p)
                desired_time = _sample_time_in_period(rng, period)
                # Herbereken met de gekozen periode — fallback-hiërarchie intern
                duration = get_running_time(
                    running_time_lookup,
                    train["first_section"],
                    first_source,
                    first_target,
                    period,
                )
                departure, feasible = _shift_until_feasible(
                    train["first_section"],
                    desired_time,
                    duration,
                    period,
                    occupied,
                    max_iter,
                    buffer_sec=buffer_sec,
                )

                if feasible:
                    arrival = departure + pd.Timedelta(seconds=duration)
                    occupied.setdefault(train["first_section"], []).append(
                        (departure, arrival)
                    )
                    placed.append({
                        **train,
                        "departure":  departure,
                        "period":     period,
                        "stage":      stage_idx,
                        "buffer_sec": buffer_sec,
                        "peak_level": stage["peak_level"],
                    })
                    n_placed_this_stage += 1
                    scheduled = True
                    break

            if not scheduled:
                still_pending.append(train)

        stage_summary.append({
            "stage":      stage_idx,
            "buffer_sec": buffer_sec,
            "peak_level": stage["peak_level"],
            "placed":     n_placed_this_stage,
            "remaining":  len(still_pending),
        })

        logger.info(
            f"[freight stage {stage_idx}/{len(stages)}] einde: "
            f"{n_placed_this_stage} geplaatst, {len(still_pending)} nog pending"
        )

        pending = still_pending

    # --- Gedropte treinen (na alle versoepelingsstappen) ---
    n_dropped = len(pending)
    if n_dropped > 0:
        logger.warning(
            f"Freight timetable: {n_dropped} trein(en) gedropt na alle "
            f"{len(stages)} versoepelingsstappen — geen conflict-vrij slot gevonden"
        )
        for train in pending:
            logger.debug(
                f"  gedropt: trein {train['train_id']} ({train['traject_key']}) "
                f"op {train['first_section']}"
            )

    # --- Materialiseer rows voor geplaatste treinen ---
    rows: list[dict] = []
    for train in placed:
        current_time = train["departure"]
        traject_key  = train["traject_key"]
        segments     = train["segments"]
        train_id     = train["train_id"]


        for seg_idx, (source, target) in enumerate(segments):
            line_no = line_no_lookup.get((source, target))
            if line_no is None:
                logger.warning(
                    f"Trein {train_id}: geen lijncode voor ({source}, {target}) — traject afgebroken"
                )
                break

            section = f"{line_no}:{source}-{target}"
            duration_sec = get_running_time(
                running_time_lookup, section, source, target, train["period"]
            )
            if duration_sec is None:
                logger.warning(
                    f"Trein {train_id}: geen rijtijd voor {section} — traject afgebroken"
                )
                break

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


    logger.info(
        f"Freight timetable gegenereerd: gevraagd={n_trains}, "
        f"geplaatst={len(placed)}, gedropt={n_dropped}, "
        f"lookup-overgeslagen={n_skipped_lookup}, {len(rows)} segmenten"
    )
    logger.info("Freight scheduling samenvatting per stap:")
    for s in stage_summary:
        logger.info(
            f"  stap {s['stage']}: buffer={s['buffer_sec']}s, "
            f"peak_level={s['peak_level']}, geplaatst={s['placed']}, "
            f"resterend={s['remaining']}"
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