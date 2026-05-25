# data/loader.py
#
# Brug tussen gold parquet bestanden en domain objecten.
#
# Laadt de gecombineerde timetable en construeert:
#   - dict[int, Train]    — alle treinen geïndexeerd op train_no
#   - dict[str, Segment]  — alle segmenten geïndexeerd op segment_id (SECTION)
#   - Timetable           — alle geplande tijden per (train_no, segment_id)
#
# Gebruik:
#   from data.loader import load_all
#   trains, segments, timetable = load_all(n_freight=300)

import logging

import pandas as pd

from data.combine_timetable import combine_timetables
from data.timetable import get_platform_alternatives
from domain import Train, TrainType, TrainSubtype
from domain import Segment, SegmentType
from domain import Timetable, ScheduledTimes

logger = logging.getLogger(__name__)


# =============================================================================
# Mappings van ruwe string waarden naar domain enums
# =============================================================================

_TRAIN_TYPE_MAP: dict[str, TrainType] = {
    "IC":      TrainType.PASSENGER,
    "L":       TrainType.PASSENGER,
    "S":       TrainType.PASSENGER,
    "EURST":   TrainType.PASSENGER,
    "ICE":     TrainType.PASSENGER,
    "INT":     TrainType.PASSENGER,
    "freight": TrainType.FREIGHT,
}

_TRAIN_SUBTYPE_MAP: dict[str, TrainSubtype] = {
    "IC":      TrainSubtype.IC,
    "L":       TrainSubtype.L,
    "S":       TrainSubtype.S,
    "EURST":   TrainSubtype.EURST,
    "ICE":     TrainSubtype.ICE,
    "INT":     TrainSubtype.INT,
    "freight": TrainSubtype.FREIGHT,
}

_SEGMENT_TYPE_MAP: dict[str, SegmentType] = {
    # SOURCE is het eerste segment van een trein — artefact van de dataverwerking,
    # gedraagt zich als lijnsegment in het MIP
    "SOURCE":                 SegmentType.BETWEEN_STATION,
    "BETWEEN-STATION":        SegmentType.BETWEEN_STATION,
    "WITHIN-STATION-PASSING": SegmentType.STATION,
    "WITHIN-STATION-DWELL":   SegmentType.STATION,
}


# =============================================================================
# Laadfu ncties
# =============================================================================

def load_trains(df: pd.DataFrame) -> dict[int, Train]:
    """
    Construeert Train objecten uit de gecombineerde gold timetable.

    Parameters
    ----------
    df : gecombineerde timetable DataFrame (output van combine_timetables)

    Returns
    -------
    dict[int, Train] geïndexeerd op train_no
    """
    trains: dict[int, Train] = {}

    for train_no, group in df.groupby("TRAIN_NO"):
        group = group.sort_values("ENTRY_SECONDS")

        raw_type = group["TRAIN_TYPE"].iloc[0]

        train_type    = _TRAIN_TYPE_MAP.get(raw_type)
        train_subtype = _TRAIN_SUBTYPE_MAP.get(raw_type)

        if train_type is None or train_subtype is None:
            logger.warning(
                f"Trein {train_no}: onbekend TRAIN_TYPE '{raw_type}' — overgeslagen"
            )
            continue

        path = tuple(group["SECTION"].tolist())

        halt_indicators: dict[str, bool] = {}
        dynamics:        dict[str, str]  = {}

        for _, row in group.iterrows():
            segment_id = row["SECTION"]

            # Stopt de trein op dit segment?
            halt_indicators[segment_id] = (row["TYPE"] == "WITHIN-STATION-DWELL")

            # Rijdynamiek — enkel ingevuld voor lijnsegmenten
            dyn = row.get("DYNAMICS")
            if pd.notna(dyn) and dyn != "":
                dynamics[segment_id] = dyn

        trains[train_no] = Train(
            train_no        = train_no,
            train_type      = train_type,
            train_subtype   = train_subtype,
            path            = path,
            halt_indicators = halt_indicators,
            dynamics        = dynamics,
        )

    logger.info(f"Treinen geladen: {len(trains)}")
    return trains


def load_segments(df: pd.DataFrame) -> dict[str, Segment]:
    """
    Construeert Segment objecten uit de gecombineerde gold timetable.

    Elk uniek SECTION-id levert precies één Segment op — segmenten
    zijn infrastructuur en bestaan onafhankelijk van treinen.

    Parameters
    ----------
    df : gecombineerde timetable DataFrame (output van combine_timetables)

    Returns
    -------
    dict[str, Segment] geïndexeerd op segment_id (= SECTION)
    """
    segments: dict[str, Segment] = {}

    for _, row in df.drop_duplicates(subset="SECTION").iterrows():
        segment_id = row["SECTION"]
        raw_type   = row["TYPE"]

        seg_type = _SEGMENT_TYPE_MAP.get(raw_type)
        if seg_type is None:
            logger.warning(
                f"Segment '{segment_id}': onbekend TYPE '{raw_type}' — overgeslagen"
            )
            continue

        segments[segment_id] = Segment(
            id       = segment_id,
            seg_type = seg_type,
            source   = row["SOURCE"],
            target   = row["TARGET"],
        )

    logger.info(f"Segmenten geladen: {len(segments)}")
    return segments


def load_timetable(df: pd.DataFrame) -> Timetable:
    """
    Construeert een Timetable object uit de gecombineerde gold timetable.

    Parameters
    ----------
    df : gecombineerde timetable DataFrame (output van combine_timetables)

    Returns
    -------
    Timetable met alle geplande tijden per (train_no, segment_id)
    """
    data: dict[tuple[int, str], ScheduledTimes] = {}

    for _, row in df.iterrows():
        train_no   = int(row["TRAIN_NO"])
        segment_id = row["SECTION"]
        raw_type   = row["TYPE"]

        entry = float(row["ENTRY_SECONDS"])
        exit_ = float(row["EXIT_SECONDS"])

        if raw_type in ("SOURCE", "BETWEEN-STATION"):
            running_time = exit_ - entry
            dwell_time   = None
            halts        = None
        else:
            running_time = None
            dwell_time   = exit_ - entry
            halts        = (raw_type == "WITHIN-STATION-DWELL")

        data[(train_no, segment_id)] = ScheduledTimes(
            entry_seconds = entry,
            exit_seconds  = exit_,
            running_time  = running_time,
            dwell_time    = dwell_time,
        )

    logger.info(
        f"Timetable geladen: {len(data)} (train_no, segment_id) combinaties"
    )
    return Timetable(data)


# =============================================================================
# Convenience functie
# =============================================================================

def load_all(
    n_freight: int,
) -> tuple[dict[int, Train], dict[str, Segment], Timetable, dict[tuple[int, str], list[str]]]:
    """
    Laadt de gecombineerde timetable en construeert alle domain objecten.

    Parameters
    ----------
    n_freight : aantal freight treinen in de gecombineerde timetable

    Returns
    -------
    (trains, segments, timetable, platform_alternatives)

    platform_alternatives : dict {(train_id, planned_seg): [alt_seg, ...]}
        Platform-alternatieven voor retracking. Alleen voor stations waarbij
        assign_platforms greedy interval scheduling gebruikt (alle platforms
        als vrije pool). Brussel-Noord is uitgesloten.

    Gebruik
    -------
    trains, segments, timetable, platform_alternatives = load_all(n_freight=300)
    """
    df = combine_timetables(n_freight)

    trains                = load_trains(df)
    segments              = load_segments(df)
    timetable             = load_timetable(df)
    platform_alternatives = get_platform_alternatives(df)

    logger.info(
        f"Platform-alternatieven geladen: "
        f"{len(platform_alternatives)} (train, segment) paren met ≥1 alternatief"
    )

    return trains, segments, timetable, platform_alternatives