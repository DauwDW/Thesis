import pandas as pd
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
    """
    Converteert node-georiënteerde data naar edge-georiënteerd (SOURCE → TARGET).

    Na deze stap geldt voor elk segment:
        PLANNED_DATE_ENTRY / PLANNED_TIME_ENTRY = vertrek uit SOURCE (= binnenkomst segment)
        PLANNED_DATE_EXIT  / PLANNED_TIME_EXIT  = aankomst in TARGET (= verlaten segment)
    """
    df = df.sort_values(by=TRAIN_GROUP + ['PLANNED_DATE_DEP', 'PLANNED_TIME_DEP'])
    df['SOURCE']    = df['PTCAR_LG_NM_NL']
    df['SOURCE_NO'] = df['PTCAR_NO']
    df['TARGET']    = df.groupby(TRAIN_GROUP)['SOURCE'].shift(-1)
    df['TARGET_NO'] = df.groupby(TRAIN_GROUP)['PTCAR_NO'].shift(-1)

    for col in ['PLANNED_DATE_ARR', 'PLANNED_TIME_ARR',
                'REAL_DATE_ARR',    'REAL_TIME_ARR']:
        df[col] = df.groupby(TRAIN_GROUP)[col].shift(-1)

    return df.rename(columns={
        'PLANNED_DATE_DEP': 'PLANNED_DATE_ENTRY',
        'PLANNED_TIME_DEP': 'PLANNED_TIME_ENTRY',
        'PLANNED_DATE_ARR': 'PLANNED_DATE_EXIT',
        'PLANNED_TIME_ARR': 'PLANNED_TIME_EXIT',
        'REAL_DATE_DEP':    'REAL_DATE_ENTRY',
        'REAL_TIME_DEP':    'REAL_TIME_ENTRY',
        'REAL_DATE_ARR':    'REAL_DATE_EXIT',
        'REAL_TIME_ARR':    'REAL_TIME_EXIT',
    })


def _add_dwell_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Voegt within-station segmenten toe voor alle tussenstops.

    Voor elk tussenstation (niet eerste, niet laatste stop van de trein):
        ENTRY = EXIT van de vorige edge  (aankomsttijd op dit station)
        EXIT  = ENTRY van de huidige edge (vertrektijd uit dit station)
        → duur = 0: passing (trein rijdt door zonder te stoppen)
        → duur > 0: dwell  (trein staat stil)

    Condities voor aanmaken within-station segment:
        - prev_target.notna(): er is een vorige stop (= niet het eerste station)
        - TARGET.notna():      de huidige edge heeft een geldig doel (= niet de laatste NaN-rij)
    """
    df = df.sort_values(
        by=TRAIN_GROUP + ['PLANNED_DATE_ENTRY', 'PLANNED_TIME_ENTRY',
                          'PLANNED_DATE_EXIT',  'PLANNED_TIME_EXIT']
    ).reset_index(drop=True)

    prev_target = df.groupby(TRAIN_GROUP)['TARGET'].shift(1)
    dwell_mask  = prev_target.notna() & df['TARGET'].notna()

    dwells = df[dwell_mask].copy()
    dwells['SOURCE']    = prev_target[dwell_mask]
    dwells['TARGET']    = dwells['SOURCE']
    dwells['SOURCE_NO'] = df.groupby(TRAIN_GROUP)['TARGET_NO'].shift(1)[dwell_mask]
    dwells['TARGET_NO'] = dwells['SOURCE_NO']

    # ENTRY = EXIT van de vorige edge (aankomsttijd op dit station)
    for col in ['PLANNED_DATE_EXIT', 'PLANNED_TIME_EXIT', 'REAL_DATE_EXIT', 'REAL_TIME_EXIT']:
        dwells[col.replace('EXIT', 'ENTRY')] = df.groupby(TRAIN_GROUP)[col].shift(1)[dwell_mask]

    # EXIT = ENTRY van de huidige edge (vertrektijd uit dit station)
    dwells['PLANNED_DATE_EXIT'] = df.loc[dwell_mask, 'PLANNED_DATE_ENTRY'].values
    dwells['PLANNED_TIME_EXIT'] = df.loc[dwell_mask, 'PLANNED_TIME_ENTRY'].values
    dwells['REAL_DATE_EXIT']    = df.loc[dwell_mask, 'REAL_DATE_ENTRY'].values
    dwells['REAL_TIME_EXIT']    = df.loc[dwell_mask, 'REAL_TIME_ENTRY'].values

    return pd.concat([df, dwells], ignore_index=True)


def _combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combineert datum + tijd kolommen naar één datetime per event.

    PLANNED_ENTRY = binnenkomst segment → altijd vroeger
    PLANNED_EXIT  = verlaten segment    → altijd later
    → geen swap nodig in add_time_in_seconds
    """
    df['PLANNED_ENTRY'] = pd.to_datetime(
        df['PLANNED_DATE_ENTRY'].astype(str) + ' ' + df['PLANNED_TIME_ENTRY'].astype(str)
    )
    df['PLANNED_EXIT'] = pd.to_datetime(
        df['PLANNED_DATE_EXIT'].astype(str) + ' ' + df['PLANNED_TIME_EXIT'].astype(str)
    )
    df['REAL_ENTRY'] = pd.to_datetime(
        df['REAL_DATE_ENTRY'].astype(str) + ' ' + df['REAL_TIME_ENTRY'].astype(str)
    )
    df['REAL_EXIT'] = pd.to_datetime(
        df['REAL_DATE_EXIT'].astype(str) + ' ' + df['REAL_TIME_EXIT'].astype(str)
    )
    return df


def load_month(month: str) -> pd.DataFrame:
    """
    Laadt en verwerkt één maand ruwe punctualiteitsdata.

    Stappen:
        1. Lees CSV, selecteer kolommen
        2. Converteer datum/tijd
        3. Weekdagen filteren
        4. Node → edge orientatie (hernoemd naar ENTRY/EXIT)
        5. Voorlopige filter: behoud treinen met minstens één Brusselse stop
        6. Within-station segmenten toevoegen
        7. Filter 1: minstens één kant in Brussel
        8. Filter 2: beide kanten in Brussel
        9. NaN verwijderen, datetime combineren

    Returns:
        Edge-georiënteerde DataFrame met PLANNED_ENTRY <= PLANNED_EXIT altijd.
    """
    path = RAW_DATA_DIR / f"Data_raw_punctuality_{month}.csv"
    df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)

    df = _parse_dates(df)
    df['DAY'] = df['DATDEP'].dt.day_name()
    df = df[df['DAY'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]

    df = _to_edge_orientation(df)

    brusselse_treinen = df[
        df['SOURCE'].isin(BRUSSELS) | df['TARGET'].isin(BRUSSELS)
    ][['DATDEP', 'TRAIN_NO']].drop_duplicates()
    df = df.merge(brusselse_treinen, on=['DATDEP', 'TRAIN_NO'], how='inner')

    # Filter 2 vóór _add_dwell_segments
    df = df[(df['SOURCE'].isin(BRUSSELS)) & (df['TARGET'].isin(BRUSSELS))]

    df = _add_dwell_segments(df)

    df = _combine_datetime(df)

    return df[['DATDEP', 'RELATION_DIRECTION', 'TRAIN_NO',
               'PLANNED_ENTRY', 'PLANNED_EXIT',
               'REAL_ENTRY', 'REAL_EXIT',
               'LINE_NO_DEP', 'SOURCE', 'TARGET']].sort_values(
        by=['DATDEP', 'TRAIN_NO', 'PLANNED_EXIT']
    ).reset_index(drop=True)


def save_bronze(month: str) -> None:
    """Verwerkt één maand en slaat op als parquet in de bronze map."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_month(month)
    df.to_parquet(BRONZE_DIR / f"{month}.parquet", index=False)
    print(f"Opgeslagen: {month}.parquet ({len(df)} rijen)")