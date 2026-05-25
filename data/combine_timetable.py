import logging

import pandas as pd

from data.timetable import load_gold, add_time_in_seconds, assign_platforms
from data.freight import normalize_to_base_date


logger = logging.getLogger(__name__)


def combine_timetables(n_freight: int) -> pd.DataFrame:
    """
    Laadt passenger en freight gold timetables, combineert ze en voegt
    ENTRY_SECONDS/EXIT_SECONDS toe op de gecombineerde dataset.

    Dropt SECTION_MACRO en RELATION_DIRECTION — niet nodig na combinatie:
        SECTION_MACRO:      backup van SECTION vóór platformtoewijzing
        RELATION_DIRECTION: vervangen door TRAIN_TYPE

    ENTRY_SECONDS en EXIT_SECONDS worden afgerond naar integers.

    Args:
        n_freight: aantal freight treinen

    Returns:
        Gecombineerde DataFrame met TRAIN_TYPE kolom en integer ENTRY_SECONDS/EXIT_SECONDS
    """
    passenger = load_gold('passenger')
    freight   = load_gold('freight', n_trains=n_freight)

    passenger = normalize_to_base_date(passenger)  # 2025-03-03 → 2025-01-01
    freight['TRAIN_TYPE'] = 'freight'

    overlap = set(passenger['TRAIN_NO']) & set(freight['TRAIN_NO'])
    if overlap:
        raise ValueError(
            "Passenger en freight bevatten overlappende TRAIN_NO waarden; "
            "dit veroorzaakt key-collisions in de Timetable. "
            f"Voorbeeld overlap: {sorted(overlap)[:5]}"
        )


    combined = pd.concat([passenger, freight], ignore_index=True)

    combined = add_time_in_seconds(combined)

    # De passenger gold-bestanden bevatten al een '-- platform X' suffix in SECTION
    # (opgeslagen na de eerste assign_platforms aanroep).  Als we hier opnieuw
    # assign_platforms aanroepen zonder reset, krijgen we dubbele suffixen:
    #   'BRUSSEL-CENTRAAL -- platform 4 -- platform 3'
    # SECTION_MACRO bevat de originele segmentnaam zonder suffix.
    # We resetten SECTION → SECTION_MACRO zodat assign_platforms schoon opnieuw
    # kan toewijzen en we correcte enkelvoudige suffixen krijgen.
    if 'SECTION_MACRO' in combined.columns:
        has_macro = combined['SECTION_MACRO'].notna()
        combined.loc[has_macro, 'SECTION'] = combined.loc[has_macro, 'SECTION_MACRO']
        combined = combined.drop(columns=['SECTION_MACRO'])

    combined, platform_diagnostics = assign_platforms(combined)
    logger.info(platform_diagnostics.summary())

    combined['PLANNED_ENTRY'] = pd.to_datetime(combined['PLANNED_ENTRY']).dt.round('s')
    combined['PLANNED_EXIT']  = pd.to_datetime(combined['PLANNED_EXIT']).dt.round('s')
    combined['ENTRY_SECONDS'] = combined['ENTRY_SECONDS'].round().astype(int)
    combined['EXIT_SECONDS']  = combined['EXIT_SECONDS'].round().astype(int)



    invalid = combined['EXIT_SECONDS'] < combined['ENTRY_SECONDS']
    if invalid.any():
        n_invalid = int(invalid.sum())
        raise ValueError(
            f"Gecombineerde timetable bevat {n_invalid} segment(en) met EXIT_SECONDS < ENTRY_SECONDS"
        )

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