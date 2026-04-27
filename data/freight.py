# Genereert een synthetische vrachtverkeer-timetable voor de Brusselse corridor.
#
# Tijdsconventie — consistent met timetable.py:
#   Between-station: PLANNED_DEPARTURE = vertrek SOURCE, PLANNED_ARRIVAL = aankomst TARGET
#   Dwell:           PLANNED_ARRIVAL = binnenkomst, PLANNED_DEPARTURE = vertrek
#                    (PLANNED_DEPARTURE > PLANNED_ARRIVAL — zelfde semantiek als passenger brondata)
#   add_time_in_seconds() in combine_timetable.py corrigeert dit uniform.
#
# SECTION-conventie — identiek aan passenger pipeline:
#   Between-station: {LINE_NO_DEP}:{SOURCE}-{TARGET}
#   Dwell:           {SOURCE}
#   Lijncode wordt afgeleid uit passenger data via build_line_no_lookup().
#   Zo is assign_platforms() uit timetable.py rechtstreeks toepasbaar op freight.
#
# Periodeverdeling (default):
#   Freight rijdt overwegend buiten de passagierspiekuren omdat passagierstreinen
#   overdag prioriteit krijgen op gedeeld spoor. De default verdeling weerspiegelt
#   dit: zwaar naar NIGHT en DAYTIME, licht naar peaks en avond.
#   Dit is een modelkeuze, geen empirisch gegeven — aanpasbaar via period_probs.

import logging

import numpy as np
import pandas as pd

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

# Default periodekansen voor freight:
#   Freight vermijdt passagierspiekuren → zwaar naar NIGHT en DAYTIME.
#   Modelkeuze, geen empirisch gegeven.
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
#
# Deze secties hebben geen passenger treinen in de gold timetable en worden
# niet gevonden door build_running_time_lookup(). Rijtijden worden geschat
# via afstand / gemiddelde snelheid × FREIGHT_FACTOR.
#
# Kalibratie via SCHAARBEEK ↔ BRUSSEL-NOORD:
#   Afstand (Infrabel open data): 2.5 km
#   Passenger mediane rijtijd:    234s (beide richtingen)
#   Impliciete snelheid:          10.68 m/s ≈ 38.5 km/h
#
# Ontbrekende secties (afstanden: Infrabel open data):
#   SCHAARBEEK ↔ THURN EN TAXIS:  3.4 km → passenger ≈ 318s → freight ≈ 414s
#   SCHAARBEEK ↔ BRUSSEL-SCHUMAN: 4.6 km → passenger ≈ 431s → freight ≈ 560s
#
# Lijncodes:
#   SCHAARBEEK ↔ THURN EN TAXIS:  lijn 28 (consistent met THURN EN TAXIS ↔ SIMONIS)
#   SCHAARBEEK ↔ BRUSSEL-SCHUMAN: lijn 26

# Passenger rijtijden in seconden (voor freight: × FREIGHT_FACTOR in lookup)
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

    Passenger SECTION formaat: {LINE_NO_DEP}:{SOURCE}-{TARGET}

    Returns:
        dict van (SOURCE, TARGET) → lijncode string
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
    Secties zonder passenger data worden aangevuld via _FALLBACK_PASSENGER_RUNNING_TIMES.

    Richtingsasymmetrie (bv. T1a vs T1b) is impliciet aanwezig: de lookup
    groepeert per (SOURCE, TARGET) apart, dus verschillende rijtijden per
    richting worden automatisch overgenomen uit de passenger data.

    Returns:
        dict van (SOURCE, TARGET) -> freight rijtijd in seconden (× 1.3 toegepast)
    """
    between = passenger_df[passenger_df['SOURCE'] != passenger_df['TARGET']].copy()
    between['PLANNED_DEPARTURE'] = pd.to_datetime(between['PLANNED_DEPARTURE'])
    between['PLANNED_ARRIVAL']   = pd.to_datetime(between['PLANNED_ARRIVAL'])
    between['RUNNING_TIME_SEC']  = (
        between['PLANNED_ARRIVAL'] - between['PLANNED_DEPARTURE']
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
    df['PLANNED_DEPARTURE'] = pd.to_datetime(df['PLANNED_DEPARTURE'])
    df['PLANNED_ARRIVAL']   = pd.to_datetime(df['PLANNED_ARRIVAL'])

    # Vroegste tijdstip per trein als referentie
    first_time = df.groupby('TRAIN_NO')['PLANNED_DEPARTURE'].transform('min')
    first_date = first_time.dt.normalize()

    offset = BASE_DATE - first_date

    df['PLANNED_DEPARTURE'] = df['PLANNED_DEPARTURE'] + offset
    df['PLANNED_ARRIVAL']   = df['PLANNED_ARRIVAL']   + offset

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
    Verschuift een vertrektijdstip totdat het interval [departure, arrival)
    niet overlapt met bestaande intervallen op dezelfde sectie.

    Gebruikt een pure overlap-check zonder headway-buffer — de secties zijn
    macro (bv. BRUSSEL-NOORD -> SCHAARBEEK), dus de relevante constraint is
    exclusieve bezetting: één trein per sectie tegelijk. Fijnkorrelige
    headway-modellering is de verantwoordelijkheid van het MIP-model.

    Geldt voor zowel between-station secties als dwell-secties (stationsnaam).

    Blijft binnen de opgegeven periode. Bij geen feasible slot na max_iter:
    geeft de best beschikbare tijd terug met feasible=False zodat de aanroeper
    het conflict kan loggen. De trein wordt toch ingepland (consistent met
    _assign_tracks_by_overlap in timetable.py).

    Returns:
        (vertrektijd, feasible)
    """
    period_end = _get_period_end(period)
    time = desired_time

    for _ in range(max_iter):
        arrival = time + pd.Timedelta(seconds=duration_sec)

        if arrival > period_end:
            return time, False

        conflict_end = None
        for dep, arr in sorted(occupied.get(section, [])):
            if time < arr and arrival > dep:   # overlap: max(starts) < min(ends)
                conflict_end = arr
                break

        if conflict_end is None:
            return time, True

        time = conflict_end   # schuif naar einde conflicterend interval

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
    dwell_time_sec:  int = 60,
    max_iter:        int = 20,
) -> pd.DataFrame:
    """
    Genereert een synthetische freight timetable voor de Brusselse corridor.

    Planningsstrategie — conflict-check enkel aan het startstation:
        1. Zoek een conflict-vrij vertrekslot op de eerste sectie van het traject.
        2. Rijd daarna door zonder verdere conflictchecks — running times zijn
           vast (mediane passenger rijtijd × 1.3), dwell times zijn vast.
        3. De planned timetable heeft dus geen wachttijden onderweg.

    Motivatie: een freight dispatcher plant een trein in aan het begin van het
    traject en verwacht dat hij ononderbroken doorrijdt. Conflicten onderweg
    zijn verstoringen die het MIP-model oplost — niet iets wat de geplande
    timetable moet modelleren. Dit vermijdt onrealistisch lange planned arrivals
    door cascading delays.

    SECTION formaat is identiek aan passenger pipeline zodat assign_platforms()
    rechtstreeks toepasbaar is.

    Bij geen feasible vertrekslot binnen de periode wordt de trein toch ingepland
    met een waarschuwing (consistent met _assign_tracks_by_overlap).

    Args:
        passenger_df:    gold passenger timetable (zonder ENTRY_SECONDS/EXIT_SECONDS).
                         Datums worden intern genormaliseerd via normalize_to_base_date().
        n_trains:        aantal te genereren freight treinen
        traject_probs:   dict van traject_key -> relatief gewicht (hoeft niet te sommeren tot 1)
        period_probs:    dict van periode -> relatief gewicht.
                         Default: DEFAULT_PERIOD_PROBS (zwaar naar NIGHT en DAYTIME)
        seed:            random seed voor reproduceerbaarheid
        base_train_no:   laagste TRAIN_NO voor freight (vermijd overlap met passenger)
        dwell_time_sec:  vaste dwell tijd in seconden per tussenstation (default: 60s,
                         gebaseerd op passagetijd van ~500m trein aan 38.5 km/h)
        max_iter:        maximaal aantal verschuivingspogingen voor vertrekslot

    Returns:
        DataFrame met freight timetable, compatibel met passenger pipeline.
        Kolommen: TRAIN_NO, SOURCE, TARGET, SECTION, PLANNED_DEPARTURE,
                  PLANNED_ARRIVAL, RELATION_DIRECTION, TYPE
    """
    if period_probs is None:
        period_probs = DEFAULT_PERIOD_PROBS

    rng = np.random.default_rng(seed)

    passenger_norm = normalize_to_base_date(passenger_df)

    line_no_lookup      = build_line_no_lookup(passenger_norm)
    running_time_lookup = build_running_time_lookup(passenger_norm)

    traject_keys, traject_p = _normalize_probs(traject_probs)
    period_keys,  period_p  = _normalize_probs(period_probs)

    # Bezettingsregister: section -> lijst van (start, end) intervallen
    # Geinitialiseerd met alle passenger between-station segmenten.
    # Enkel gebruikt voor conflictcheck op eerste sectie van elk traject.
    occupied: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    passenger_between = passenger_norm[passenger_norm['SOURCE'] != passenger_norm['TARGET']]
    for _, row in passenger_between.iterrows():
        occupied.setdefault(row['SECTION'], []).append(
            (row['PLANNED_DEPARTURE'], row['PLANNED_ARRIVAL'])
        )

    rows = []
    n_infeasible = 0

    for i in range(n_trains):
        train_id     = base_train_no + i
        traject_key  = rng.choice(traject_keys, p=traject_p)
        stations     = TRAJECTEN[traject_key]
        period       = rng.choice(period_keys, p=period_p)
        segments     = _build_segments(stations)

        # --- Zoek conflict-vrij vertrekslot op eerste sectie ---
        first_source, first_target = segments[0]
        first_line_no = line_no_lookup.get((first_source, first_target))
        if first_line_no is None:
            logger.warning(f"Trein {train_id}: geen lijncode voor ({first_source}, {first_target}) — overgeslagen")
            continue

        first_duration = running_time_lookup.get((first_source, first_target))
        if first_duration is None:
            logger.warning(f"Trein {train_id}: geen rijtijd voor ({first_source}, {first_target}) — overgeslagen")
            continue

        first_section  = f"{first_line_no}:{first_source}-{first_target}"
        desired_time   = _sample_time_in_period(rng, period)

        departure, feasible = _shift_until_feasible(
            first_section, desired_time, first_duration, period, occupied, max_iter
        )

        if not feasible:
            n_infeasible += 1
            logger.warning(
                f"Trein {train_id} ({traject_key}): geen conflict-vrij vertrekslot op "
                f"{first_section} binnen periode {period} — ingepland met conflict"
            )

        # Registreer eerste sectie in occupied zodat volgende treinen
        # dit vertrekslot vermijden
        arrival = departure + pd.Timedelta(seconds=first_duration)
        occupied.setdefault(first_section, []).append((departure, arrival))

        # --- Rijd door zonder verdere conflictchecks ---
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
                "PLANNED_DEPARTURE":  current_time,
                "PLANNED_ARRIVAL":    arrival,
                "RELATION_DIRECTION": traject_key,
                "TYPE":               "BETWEEN-STATION",
            })
            current_time = arrival

            # --- Dwell-segment (niet voor eindstation) ---
            if seg_idx < len(segments) - 1:
                dwell_exit = current_time + pd.Timedelta(seconds=dwell_time_sec)
                rows.append({
                    "TRAIN_NO":           train_id,
                    "SOURCE":             target,
                    "TARGET":             target,
                    "SECTION":            target,        # consistent met passenger conventie
                    "PLANNED_DEPARTURE":  dwell_exit,    # DEPARTURE > ARRIVAL voor dwell:
                    "PLANNED_ARRIVAL":    current_time,  # zelfde semantiek als passenger brondata
                    "RELATION_DIRECTION": traject_key,
                    "TYPE":               "WITHIN-STATION-DWELL",
                })
                current_time = dwell_exit

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

    Alle freight segmenten krijgen '0-0' — freight treinen passeren de
    Brusselse corridor als doorgaand verkeer op kruissnelheid.
    ACC/BR zijn niet van toepassing binnen het beperkte studiegebied.

    Args:
        df: freight timetable DataFrame

    Returns:
        Kopie met DYNAMICS kolom toegevoegd
    """
    df = df.copy()
    df['DYNAMICS'] = '0-0'
    return df