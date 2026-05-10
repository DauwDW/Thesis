# notebooks/utils/metrics.py
#
# Metriekenberekening voor experimentele evaluatie.
#
# Gebruik:
#   from utils.metrics import compute_metrics, build_result_row, save_result_row, aggregate_config
#
# Input:
#   df — output van run_simulation.results_to_dataframe(), met kolommen:
#        train_id, train_type, train_subtype, segment_id,
#        planned_entry, planned_exit, actual_entry, actual_exit,
#        entry_delay, exit_delay
#
# Opslagstructuur
# ---------------
# Per configuratie wordt één map aangemaakt onder results/:
#
#   results/
#     <config_name>/
#       raw_runs.csv      — één rij per seed (ruwe data, voor reproduceerbaarheid)
#       aggregated.csv    — één rij, geaggregeerd over alle seeds (voor analyse)
#
# Bij een fout in één configuratie verwijder je gewoon die map en herstart je
# enkel die configuratie. De rest blijft onaangetast.
#
# Metrics overzicht
# -----------------
#
# PERFORMANTIEMETRIEKEN
# ---------------------
# TED_passenger  : Total End Delay passagierstreinen (s)
#                  Som van entry_delay op het laatste segment per passagierstrein.
#                  Primaire performantiemaatstaf, vergelijkbaar met Mariska (2024).
#
# TED_freight    : Total End Delay goederentreinen (s)
#                  Som van entry_delay op het laatste segment per goederentrein.
#                  Toont het effect van priority mechanisms op goederentreinen.
#
# TED_combined   : TED_passenger + TED_freight (s)
#                  Gecombineerde eindvertraging over alle treinen.
#                  Primaire metric voor Pareto-analyse (performance vs effort).
#
# TAD_passenger  : Total Arrival Delay passagierstreinen (s)
#                  Som van entry_delay over ALLE segmenten per passagierstrein.
#                  Penaliseert tussentijdse vertragingen die voor het eindpunt
#                  ingehaald worden — in tegenstelling tot TED.
#                  Primaire metric in Fase 2 voor runs met arrival_delay objective.
#
# TAD_freight    : Total Arrival Delay goederentreinen (s)
#                  Som van entry_delay over alle segmenten per goederentrein.
#
# TAD_combined   : TAD_passenger + TAD_freight (s)
#
# VERDELINGSMETRIEKEN
# -------------------
# delay_ratio    : (TED_freight / n_freight) / (TED_passenger / n_passenger)
#                  Gemiddelde eindvertraging per goederentrein gedeeld door
#                  gemiddelde eindvertraging per passagierstrein.
#                  Kwantificeert de per-trein ongelijkheid tussen verkeerstypes.
#                  DR > 1 betekent dat goederentreinen proportioneel meer vertraging
#                  dragen dan passagierstreinen.
#                  Primaire metric voor priority mechanism analyse (Fase 1 en 3).
#                  None als één van beide types niet aanwezig is of TED_passenger = 0.
#
# EFFORT METRIEKEN
# ----------------
# total_solve_time  : Cumulatieve solver tijd over alle MIP-aanroepen (s).
#                     Tweede as voor Pareto-analyse naast TED_combined.
#
# n_rescheduled     : Aantal succesvolle MIP-aanroepen tijdens de run.
#                     Consistent met Mariska (2024) als tweede effort-metric.
#                     Enkel gerapporteerd in Fase 1 waar timing strategy varieert.
#
# n_fcfs_fallback   : Aantal keren dat de solver faalde en FCFS werd gebruikt.
#
# n_skipped         : Aantal keren dat de trigger niet vuurde.
#
# ROBUUSTHEIDSMETRIEKEN
# ---------------------
# infeasible_run : True als de solver minstens één keer faalde (n_fcfs_fallback > 0).
#                  Indicator van configuratiestabiliteit.
#                  Kritisch bij hoge freight percentages (Fase 3).
#
# Gebruik per fase
# ----------------
# Fase 1 (Timing × Priority):
#   Primair:    TED_combined, TED_passenger, TED_freight, delay_ratio
#   Effort:     total_solve_time, n_rescheduled
#   Robuustheid: infeasible_run
#
# Fase 2 (Objective Function):
#   Primair:    TED_combined, TED_passenger, TED_freight (altijd gemeten)
#               TAD_combined, TAD_passenger, TAD_freight (altijd gemeten)
#               delay_ratio
#   Effort:     total_solve_time
#   Robuustheid: infeasible_run
#
# Fase 3 (Sensitiviteitsanalyse):
#   Primair:    TED_combined, TED_passenger, TED_freight
#               delay_ratio (primair voor interactieplot freight × upgrade)
#   Effort:     total_solve_time
#   Robuustheid: infeasible_run

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Metric kolommen waarover mean en std berekend worden bij aggregatie
_METRIC_COLS = [
    'TED_passenger', 'TED_freight', 'TED_combined',
    'TAD_passenger', 'TAD_freight', 'TAD_combined',
    'delay_ratio',
    'total_solve_time', 'n_rescheduled', 'n_fcfs_fallback', 'n_skipped',
]

# Configuratiekolommen die identiek zijn voor alle seeds van dezelfde configuratie
_CONFIG_COLS = [
    'phase', 'strategy_type', 'periodic_rf', 'event_driven_rf',
    'controller_freq', 'threshold_conf', 'priority',
    'weight_passenger', 'weight_freight', 'gamma', 'upgrade',
    'objective', 'freight_pct', 'mc_delay_per_train', 'solver_timeout',
]


# =============================================================================
# Metrics berekenen
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Berekent alle evaluatiemetrieken uit het df van run_simulation.

    Parameters
    ----------
    df : pd.DataFrame — output van run_simulation.results_to_dataframe()
         Vereiste kolommen: train_id, train_type, planned_entry, entry_delay

    Returns
    -------
    dict met alle metrics als floats. Zie module-docstring voor definities.
    """
    if df.empty:
        return _empty_metrics()

    # --- Laatste segment per trein ---
    # Laatste segment = rij met de hoogste planned_entry per trein
    last_seg = (
        df.sort_values('planned_entry')
          .groupby(['train_id', 'train_type'])
          .last()
          .reset_index()
    )

    pass_mask    = last_seg['train_type'] == 'P'
    freight_mask = last_seg['train_type'] == 'F'

    # --- Total End Delay (enkel eindpunt) ---
    ted_p = float(last_seg[pass_mask]['entry_delay'].fillna(0.0).sum())
    ted_f = float(last_seg[freight_mask]['entry_delay'].fillna(0.0).sum())

    # --- Total Arrival Delay (alle segmenten) ---
    tad_p = float(df[df['train_type'] == 'P']['entry_delay'].fillna(0.0).sum())
    tad_f = float(df[df['train_type'] == 'F']['entry_delay'].fillna(0.0).sum())

    # --- Aantallen treinen die eindbestemming bereikten ---
    n_pass    = int(pass_mask.sum())
    n_freight = int(freight_mask.sum())

    # --- Delay Ratio: gemiddelde eindvertraging freight / passenger ---
    if n_pass > 0 and n_freight > 0:
        avg_ted_p = ted_p / n_pass
        avg_ted_f = ted_f / n_freight
        delay_ratio = avg_ted_f / avg_ted_p if avg_ted_p > 0 else None
    else:
        delay_ratio = None

    return {
        'TED_passenger': ted_p,
        'TED_freight':   ted_f,
        'TED_combined':  ted_p + ted_f,
        'TAD_passenger': tad_p,
        'TAD_freight':   tad_f,
        'TAD_combined':  tad_p + tad_f,
        'delay_ratio':   delay_ratio,
        'n_passenger':   n_pass,
        'n_freight':     n_freight,
    }


def _empty_metrics() -> dict:
    """Retourneert lege metrics als de simulatie geen data produceerde."""
    return {
        'TED_passenger': None,
        'TED_freight':   None,
        'TED_combined':  None,
        'TAD_passenger': None,
        'TAD_freight':   None,
        'TAD_combined':  None,
        'delay_ratio':   None,
        'n_passenger':   0,
        'n_freight':     0,
    }


# =============================================================================
# Resultatenrij opbouwen
# =============================================================================

def build_result_row(
    seed:       int,
    phase:      int,
    metrics:    dict,
    meta:       dict,
    # --- Timing parameters ---
    strategy_type:      str,
    periodic_rf:        int   | None = None,
    event_driven_rf:    int   | None = None,
    controller_freq:    int   | None = None,
    threshold_conf:     float | None = None,
    # --- Priority parameters ---
    priority:           str          = 'no_priority',
    weight_passenger:   int          = 1,
    weight_freight:     int          = 1,
    gamma:              float | None = None,
    upgrade:            int   | None = None,
    # --- Objective ---
    objective:          str          = 'final_delay',
    # --- Vaste parameters ---
    freight_pct:        float        = 0.15,
    mc_delay_per_train: float        = 10.0,
    solver_timeout:     int          = 60,
) -> dict:
    """
    Bouwt één volledige resultatenrij op voor opslag in raw_runs.csv.

    Combineert:
    - Identificatie (seed, fase)
    - Experimentele configuratie (timing, priority, objective, vaste params)
    - Performantiemetrieken (uit compute_metrics)
    - Solver effort (uit meta['controller_summary'] van run_simulation)

    Parameters
    ----------
    seed               : int   — random seed van deze run
    phase              : int   — 1, 2 of 3
    metrics            : dict  — output van compute_metrics()
    meta               : dict  — output van run_simulation() (derde return waarde)
    strategy_type      : str   — 'periodic', 'event_driven', 'hybrid'
    periodic_rf        : int   — periodieke reschedulingfrequentie (s)
    event_driven_rf    : int   — event-driven reschedulingfrequentie (s)
    controller_freq    : int   — controllerfrequentie (s)
    threshold_conf     : float — drempelconfidentie MC (0.4/0.6/0.8)
    priority           : str   — 'no_priority', 'static', 'dynamic'
    weight_passenger   : int   — gewicht passagierstrein in objective
    weight_freight     : int   — gewicht goederentrein in objective
    gamma              : float — vertragingsdrempel dynamic priority (s)
    upgrade            : int   — upgradegrootte dynamic priority (+1/+2/+3)
    objective          : str   — 'final_delay' of 'arrival_delay'
    freight_pct        : float — aandeel goederentreinen (0.05/0.15/0.25)
    mc_delay_per_train : float — MC threshold per actieve trein (s)
    solver_timeout     : int   — max solver tijd per aanroep (s)

    Returns
    -------
    dict — één rij klaar voor opslag in raw_runs.csv
    """
    ctrl = meta.get('controller_summary', {})

    row = {
        # --- Identificatie ---
        'seed':               seed,
        'phase':              phase,

        # --- Timing parameters ---
        'strategy_type':      strategy_type,
        'periodic_rf':        periodic_rf,
        'event_driven_rf':    event_driven_rf,
        'controller_freq':    controller_freq,
        'threshold_conf':     threshold_conf,

        # --- Priority parameters ---
        'priority':           priority,
        'weight_passenger':   weight_passenger,
        'weight_freight':     weight_freight,
        'gamma':              gamma,
        'upgrade':            upgrade,

        # --- Objective ---
        'objective':          objective,

        # --- Vaste parameters ---
        'freight_pct':        freight_pct,
        'mc_delay_per_train': mc_delay_per_train,
        'solver_timeout':     solver_timeout,

        # --- Solver effort — uit meta['controller_summary'] ---
        'total_solve_time':   ctrl.get('total_solver_runtime_s'),
        'n_rescheduled':      ctrl.get('n_rescheduled'),
        'n_fcfs_fallback':    ctrl.get('n_fcfs_fallback'),
        'n_skipped':          ctrl.get('n_skipped'),
        'infeasible_run':     ctrl.get('n_fcfs_fallback', 0) > 0,
    }

    row.update(metrics)
    return row


# =============================================================================
# Resultaten opslaan — per configuratie
# =============================================================================

def save_result_row(row: dict, config_dir: Path | str) -> None:
    """
    Voegt één seed-rij toe aan raw_runs.csv binnen de configuratiemap.

    Maakt de map en het bestand aan als ze nog niet bestaan.
    Appended anders zonder header.

    Parameters
    ----------
    row        : dict        — output van build_result_row()
    config_dir : Path | str  — pad naar de configuratiemap
                               (bv. results/n182_periodic_static_wp2_wf1_g180)
    """
    config_dir  = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    output_path = config_dir / 'raw_runs.csv'

    df_row       = pd.DataFrame([row])
    write_header = not output_path.exists()
    df_row.to_csv(output_path, mode='a', header=write_header, index=False)


def aggregate_config(config_dir: Path | str) -> pd.DataFrame:
    """
    Aggregeert alle seed-runs van één configuratie naar één rij.

    Leest raw_runs.csv uit de configuratiemap, berekent mean en std
    voor alle metric kolommen, en slaat het resultaat op als aggregated.csv.

    Parameters
    ----------
    config_dir : Path | str — pad naar de configuratiemap

    Returns
    -------
    pd.DataFrame met één rij — de geaggregeerde resultaten
    """
    config_dir  = Path(config_dir)
    raw_path    = config_dir / 'raw_runs.csv'

    if not raw_path.exists():
        raise FileNotFoundError(f"raw_runs.csv niet gevonden in {config_dir}")

    df = pd.read_csv(raw_path)

    # Configuratiekolommen — neem eerste rij (identiek voor alle seeds)
    config_vals = {
        col: df[col].iloc[0]
        for col in _CONFIG_COLS
        if col in df.columns
    }

    # Robuustheid
    config_vals['n_runs']       = len(df)
    config_vals['n_infeasible'] = int(df['infeasible_run'].sum()) if 'infeasible_run' in df.columns else 0

    # Mean en std per metric kolom
    for col in _METRIC_COLS:
        if col in df.columns:
            config_vals[f'{col}_mean'] = df[col].mean()
            config_vals[f'{col}_std']  = df[col].std()

    agg_df = pd.DataFrame([config_vals])
    agg_df.to_csv(config_dir / 'aggregated.csv', index=False)

    return agg_df


def load_all_aggregated(results_dir: Path | str = 'results') -> pd.DataFrame:
    """
    Laadt alle aggregated.csv bestanden uit alle configuratiemappen.

    Gebruik dit voor de volledige analyse over alle fasen en configuraties.

    Parameters
    ----------
    results_dir : Path | str — hoofdmap met alle configuratiemappen

    Returns
    -------
    pd.DataFrame met één rij per configuratie
    """
    results_dir = Path(results_dir)
    frames      = []

    for agg_path in sorted(results_dir.rglob('aggregated.csv')):
        df = pd.read_csv(agg_path)
        df['config_dir'] = str(agg_path.parent.name)
        frames.append(df)

    if not frames:
        print(f"Geen aggregated.csv bestanden gevonden in '{results_dir}'.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)