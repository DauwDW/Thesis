import pandas as pd
from pathlib import Path
from config.settings import RAW_DATA_DIR, BRONZE_DIR

BRUSSELS = [
    'JETTE', 'SCHAARBEEK', 'BRUSSEL-NOORD', 'BRUSSEL-CENTRAAL',
    'BRUSSEL-CONGRES', 'BRUSSEL-KAPELLEKERK', 'BRUSSEL-ZUID',
    'VORST-OOST', 'BRUSSEL-WEST', 'SIMONIS', 'THURN EN TAXIS',
    'BOCKSTAEL', 'SINT-AGATHA-BERCHEM', 'ZELLIK', 'ANDERLECHT',
    'BRUSSEL-SCHUMAN'
]

COLUMNS = [
    'DATDEP', 'RELATION_DIRECTION', 'TRAIN_NO',
    'REAL_DATE_ARR', 'REAL_TIME_ARR',
    'REAL_DATE_DEP', 'REAL_TIME_DEP',
    'PLANNED_DATE_ARR', 'PLANNED_TIME_ARR',
    'PLANNED_DATE_DEP', 'PLANNED_TIME_DEP',
    'PTCAR_LG_NM_NL', 'PTCAR_NO', 'LINE_NO_DEP'
]

TRAIN_GROUP = ['DATDEP', 'RELATION_DIRECTION', 'TRAIN_NO']


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Converteert alle datum- en tijdkolommen naar correcte types."""
    for col in ['DATDEP', 'PLANNED_DATE_ARR', 'PLANNED_DATE_DEP',
                'REAL_DATE_ARR', 'REAL_DATE_DEP']:
        df[col] = pd.to_datetime(df[col], format='%d%b%Y')

    for col in ['PLANNED_TIME_ARR', 'PLANNED_TIME_DEP',
                'REAL_TIME_ARR', 'REAL_TIME_DEP']:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S').dt.time

    return df


def _to_edge_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """Converteert node-georiënteerde data naar edge-georiënteerd (SOURCE → TARGET)."""
    df['SOURCE'] = df['PTCAR_LG_NM_NL']
    df['SOURCE_NO'] = df['PTCAR_NO']
    df['TARGET'] = df.groupby(TRAIN_GROUP)['SOURCE'].shift(-1)
    df['TARGET_NO'] = df.groupby(TRAIN_GROUP)['PTCAR_NO'].shift(-1)

    # Shift aankomsttijden: aankomst van edge = vertrek van volgende node
    for col in ['PLANNED_DATE_ARR', 'PLANNED_TIME_ARR',
                'REAL_DATE_ARR', 'REAL_TIME_ARR']:
        df[col] = df.groupby(TRAIN_GROUP)[col].shift(-1)

    return df


def _add_dwell_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt dwell-segmenten toe: SOURCE == TARGET voor stilstaande treinen.
    Een dwell ontstaat wanneer een trein twee opeenvolgende edges heeft.
    """
    next_source = df.groupby(TRAIN_GROUP)['SOURCE'].shift(-1)
    next_planned_dep_date = df.groupby(TRAIN_GROUP)['PLANNED_DATE_DEP'].shift(-1)
    next_planned_dep_time = df.groupby(TRAIN_GROUP)['PLANNED_TIME_DEP'].shift(-1)
    next_real_dep_date = df.groupby(TRAIN_GROUP)['REAL_DATE_DEP'].shift(-1)
    next_real_dep_time = df.groupby(TRAIN_GROUP)['REAL_TIME_DEP'].shift(-1)

    # Dwell-conditie: er bestaat een volgende stop binnen dezelfde trein
    dwell_mask = next_source.notna()

    dwells = df[dwell_mask].copy()
    dwells['TARGET'] = dwells['SOURCE']
    dwells['TARGET_NO'] = dwells['SOURCE_NO']
    dwells['PLANNED_DATE_DEP'] = next_planned_dep_date[dwell_mask]
    dwells['PLANNED_TIME_DEP'] = next_planned_dep_time[dwell_mask]
    dwells['REAL_DATE_DEP'] = next_real_dep_date[dwell_mask]
    dwells['REAL_TIME_DEP'] = next_real_dep_time[dwell_mask]

    return pd.concat([df, dwells], ignore_index=True)


def _combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combineert datum + tijd kolommen naar één datetime per event."""
    for prefix in ['PLANNED_DEPARTURE', 'PLANNED_ARRIVAL',
                   'REAL_DEPARTURE', 'REAL_ARRIVAL']:
        date_col = prefix.replace('DEPARTURE', 'DATE_DEP').replace('ARRIVAL', 'DATE_ARR')
        time_col = prefix.replace('DEPARTURE', 'TIME_DEP').replace('ARRIVAL', 'TIME_ARR')
        df[prefix] = pd.to_datetime(
            df[date_col].astype(str) + ' ' + df[time_col].astype(str)
        )
    return df


def load_month(month: str) -> pd.DataFrame:
    """
    Laadt en verwerkt één maand ruwe punctualiteitsdata.
    
    Stappen:
        1. Lees CSV, selecteer kolommen
        2. Converteer datum/tijd
        3. Weekdagen filteren
        4. Node → edge orientatie
        5. Eerste Brussels filter (minstens één kant)
        6. Dwell-segmenten toevoegen
        7. Tweede Brussels filter (beide kanten)
        8. NaN verwijderen, datetime combineren
    
    Args:
        month: bv. '202403'
    Returns:
        Edge-georiënteerde DataFrame met PLANNED en REAL tijden per segment
    """
    path = RAW_DATA_DIR / f"Data_raw_punctuality_{month}.csv"
    df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)

    df = _parse_dates(df)

    # Weekdagen
    df['DAY'] = df['DATDEP'].dt.day_name()
    df = df[df['DAY'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]

    df = _to_edge_orientation(df)

    # Filter 1: minstens één kant in Brussel
    df = df[(df['SOURCE'].isin(BRUSSELS)) | (df['TARGET'].isin(BRUSSELS))]

    df = df.sort_values(by=['DATDEP', 'TRAIN_NO', 'PLANNED_TIME_ARR'])
    df = _add_dwell_segments(df)

    # Filter 2: beide kanten in Brussel
    df = df[(df['SOURCE'].isin(BRUSSELS)) & (df['TARGET'].isin(BRUSSELS))]

    df.dropna(subset=['SOURCE', 'TARGET', 'PLANNED_DATE_DEP', 'PLANNED_DATE_ARR'],
              inplace=True)

    df = _combine_datetime(df)

    return df[['DATDEP', 'RELATION_DIRECTION', 'TRAIN_NO',
               'PLANNED_DEPARTURE', 'PLANNED_ARRIVAL',
               'REAL_DEPARTURE', 'REAL_ARRIVAL',
               'LINE_NO_DEP', 'SOURCE', 'TARGET']].sort_values(
        by=['DATDEP', 'TRAIN_NO', 'PLANNED_ARRIVAL']
    ).reset_index(drop=True)


def save_bronze(month: str) -> None:
    """Verwerkt één maand en slaat op als parquet in de bronze map."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_month(month)
    df.to_parquet(BRONZE_DIR / f"{month}.parquet", index=False)
    print(f"Opgeslagen: {month}.parquet ({len(df)} rijen)")