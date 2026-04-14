import logging

import pandas as pd

from data.timetable import load_gold, add_time_in_seconds
from data.freight import normalize_to_base_date


logger = logging.getLogger(__name__)


def combine_timetables(n_freight: int) -> pd.DataFrame:
    """
    Laadt passenger en freight gold timetables, combineert ze en voegt
    ENTRY_SECONDS/EXIT_SECONDS toe op de gecombineerde dataset.

    Dropt SECTION_MACRO en RELATION_DIRECTION — niet nodig na combinatie:
        SECTION_MACRO:      backup van SECTION vóór platformtoewijzing, overbodig na assign_platforms
        RELATION_DIRECTION: vervangen door TRAIN_TYPE voor passenger, enkel traject key voor freight

    ENTRY_SECONDS en EXIT_SECONDS worden afgerond naar integers.

    Args:
        n_freight: aantal freight treinen — bepaalt welke freight submap geladen wordt

    Returns:
        Gecombineerde DataFrame met TRAIN_TYPE kolom en integer ENTRY_SECONDS/EXIT_SECONDS
    """
    passenger = load_gold('passenger')
    freight   = load_gold('freight', n_trains=n_freight)
    
    passenger = normalize_to_base_date(passenger)  # 2025-03-03 → 2025-01-01
    freight['TRAIN_TYPE'] = 'freight'

    combined = pd.concat([passenger, freight], ignore_index=True)

    combined = add_time_in_seconds(combined)

    combined['PLANNED_DEPARTURE'] = pd.to_datetime(combined['PLANNED_DEPARTURE']).dt.round('s')
    combined['PLANNED_ARRIVAL']   = pd.to_datetime(combined['PLANNED_ARRIVAL']).dt.round('s')
    combined['ENTRY_SECONDS'] = combined['ENTRY_SECONDS'].round().astype(int)
    combined['EXIT_SECONDS']  = combined['EXIT_SECONDS'].round().astype(int)

    combined = combined.drop(
        columns=['SECTION_MACRO', 'RELATION_DIRECTION'],
        errors='ignore',
    )

    logger.info(
        f"Gecombineerd: {combined['TRAIN_NO'].nunique()} treinen "
        f"({passenger['TRAIN_NO'].nunique()} passenger, "
        f"{freight['TRAIN_NO'].nunique()} freight), "
        f"{len(combined)} segmenten"
    )

    return combined