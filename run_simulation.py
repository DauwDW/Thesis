"""
run_simulation.py

Centrale pipeline voor de thesis-simulatie. Laadt gold-data, configureert
de controller en voert de discrete-event simulatie uit.

Gebruik
-------
    from run_simulation import run_simulation, load_all_runs

    state, df, meta = run_simulation(
        n_freight          = 300,
        trigger_strategy   = "hybrid",
        objective_strategy = "static",
        seed               = 42,
    )

    runs = load_all_runs()
    runs[runs["seed"] == 42]
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import config.settings as settings
from data.loader           import load_all
from controller.controller import Controller
from controller.triggers   import make_trigger
from simulation.simulator  import Simulator


# =============================================================================
# Helpers
# =============================================================================

def _build_trigger(trigger_strategy, trains, segments, timetable,
                   event_driven_freq, controller_freq,
                   periodic_freq, threshold_confidence, rng):
    """Bouwt de juiste trigger op basis van de gekozen strategie."""
    if trigger_strategy == "periodic":
        return make_trigger("periodic",
                            periodic_freq=periodic_freq)

    if trigger_strategy == "event_driven":
        return make_trigger("event_driven",
                            trains=trains,
                            segments=segments,
                            timetable=timetable,
                            event_driven_freq=event_driven_freq,
                            controller_freq=controller_freq,
                            threshold_confidence=threshold_confidence,
                            rng=rng)

    if trigger_strategy == "hybrid":
        return make_trigger("hybrid",
                            trains=trains,
                            segments=segments,
                            timetable=timetable,
                            event_driven_freq=event_driven_freq,
                            controller_freq=controller_freq,
                            periodic_freq=periodic_freq,
                            threshold_confidence=threshold_confidence,
                            rng=rng)

    raise ValueError(f"Onbekende trigger_strategy: '{trigger_strategy}'. "
                     f"Kies uit: 'periodic', 'event_driven', 'hybrid'.")


def results_to_dataframe(state, trains, timetable):
    """
    Bouwt een DataFrame met één rij per (train_id, segment_id).

    Kolommen
    --------
    train_id, train_type, train_subtype, segment_id,
    planned_entry, planned_exit, actual_entry, actual_exit,
    entry_delay, exit_delay
    """
    records = []

    for train_id, train in trains.items():
        for seg_id in train.path:
            try:
                actual_entry = state.actual_entry(train_id, seg_id)
                actual_exit  = state.actual_exit(train_id, seg_id)
            except KeyError:
                continue

            try:
                planned_entry = timetable.scheduled_arrival(train_id, seg_id)
                planned_exit  = timetable.scheduled_departure(train_id, seg_id)
            except KeyError:
                planned_entry = None
                planned_exit  = None

            records.append({
                "train_id":      train_id,
                "train_type":    train.train_type.value,
                "train_subtype": train.train_subtype.value,
                "segment_id":    seg_id,
                "planned_entry": planned_entry,
                "planned_exit":  planned_exit,
                "actual_entry":  actual_entry,
                "actual_exit":   actual_exit,
                "entry_delay":   max(0.0, actual_entry - planned_entry) if planned_entry is not None else None,
                "exit_delay":    max(0.0, actual_exit  - planned_exit)  if planned_exit  is not None else None,
            })

    return pd.DataFrame(records)


def _build_run_name(params):
    """Unieke bestandsnaam op basis van alle run-parameters."""
    return (
        f"n{params['n_freight']}"
        f"_{params['trigger_strategy']}"
        f"_{params['objective_strategy']}"
        f"_wp{params['weight_passenger']}"
        f"_wf{params['weight_freight']}"
        f"_g{int(params['dynamic_threshold'])}"
        f"_edf{int(params['event_driven_freq'])}"
        f"_cf{int(params['controller_freq'])}"
        f"_pf{int(params['periodic_freq'])}"
        f"_tc{params['threshold_confidence']}"
        f"_seed{params['seed']}"
    )


def _save_results(df, params, controller_summary, output_dir):
    """
    Slaat het DataFrame (parquet) en de metadata (JSON) op.
    Runs met dezelfde configuratie én seed zijn per definitie identiek —
    de run-index vangt het geval op dat iemand toch twee keer exact hetzelfde draait.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = _build_run_name(params)
    run_index = len(list(output_dir.glob(f"{base_name}_run*.parquet"))) + 1
    run_name  = f"{base_name}_run{run_index}"

    parquet_path = output_dir / f"{run_name}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {**params, "run_index": run_index, "controller_summary": controller_summary}
    with open(output_dir / f"{run_name}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Resultaten opgeslagen:\n  {parquet_path}")
    return meta


# =============================================================================
# Resultaten ophalen
# =============================================================================

def load_all_runs(results_dir: Path | str = settings.RESULTS_DIR) -> pd.DataFrame:
    """
    Laadt alle run-metadata uit de JSON-bestanden in results_dir.

    Elke rij is één run, met alle parameters én controller_summary als kolommen.
    De kolom 'parquet_path' geeft het pad naar de bijbehorende resultaten.

    Gebruik
    -------
        runs = load_all_runs()

        # Filter op specifieke parameters
        runs[runs["seed"] == 42]
        runs[(runs["objective_strategy"] == "dynamic") & (runs["n_freight"] == 300)]

        # Gemiddelde over meerdere seeds van dezelfde configuratie
        runs.groupby(["trigger_strategy", "objective_strategy"])["ctrl_n_rescheduled"].mean()

        # Laad resultaten van een specifieke run
        df = pd.read_parquet(runs.iloc[0]["parquet_path"])
    """
    records = []
    for json_path in sorted(Path(results_dir).glob("*.json")):
        with open(json_path) as f:
            meta = json.load(f)
        for key, value in meta.pop("controller_summary", {}).items():
            meta[f"ctrl_{key}"] = value
        meta["parquet_path"] = str(json_path.with_suffix(".parquet"))
        records.append(meta)

    if not records:
        print(f"Geen runs gevonden in '{results_dir}'.")
        return pd.DataFrame()

    return pd.DataFrame(records)


# =============================================================================
# Hoofdfunctie
# =============================================================================

def run_simulation(
    # Data
    n_freight:            int   = 300,

    # Trigger
    trigger_strategy:     str   = "hybrid",   # "periodic" | "event_driven" | "hybrid"
    event_driven_freq:    float = 900,         # min interval tussen solver calls (s)
    controller_freq:      float = 300,         # evaluatiefrequentie trigger (s); <= event_driven_freq
    periodic_freq:        float = 1800,        # interval periodic / harde deadline hybrid (s)
    threshold_confidence: float = 0.6,         # MC: P(delay > drempel) om solver te vuren

    # MIP
    objective_strategy:   str   = "static",   # "static" | "dynamic" | "timetable_deviation"
    weight_passenger:     int   = 2,
    weight_freight:       int   = 1,
    dynamic_threshold:    float = 300,         # γ: drempel voor dynamic priority upgrade (s)

    # Reproduceerbaarheid
    seed:                 int   = settings.SIMULATION_SEED,

    # Output
    output_dir:           Path | str = settings.RESULTS_DIR,
    save:                 bool  = True,
) -> tuple:
    """
    Voert de volledige simulatiepipeline uit vanuit de gold-bestanden.

    De seed wordt gebruikt voor:
      - Rijtijdsampling in de simulator (sample_duration)
      - Monte Carlo rollouts in de trigger (child-RNGs afgeleid van trigger_rng)

    Returns
    -------
    (state, df, meta)
        state : SystemState  — eindtoestand van de simulatie
        df    : pd.DataFrame — resultaten per (train_id, segment_id)
        meta  : dict         — parameters + controller summary
    """
    params = dict(
        n_freight            = n_freight,
        trigger_strategy     = trigger_strategy,
        event_driven_freq    = event_driven_freq,
        controller_freq      = controller_freq,
        periodic_freq        = periodic_freq,
        threshold_confidence = threshold_confidence,
        objective_strategy   = objective_strategy,
        weight_passenger     = weight_passenger,
        weight_freight       = weight_freight,
        dynamic_threshold    = dynamic_threshold,
        seed                 = seed,
    )

    # 1. Data laden
    print("Data laden...")
    trains, segments, timetable = load_all(n_freight)

    # 2. Trigger bouwen
    # Aparte RNG voor de trigger zodat simulator- en trigger-streams
    # onafhankelijk zijn maar beide volledig reproduceerbaar via seed.
    print(f"Controller bouwen (trigger={trigger_strategy}, objective={objective_strategy})...")
    trigger = _build_trigger(
        trigger_strategy     = trigger_strategy,
        trains               = trains,
        segments             = segments,
        timetable            = timetable,
        event_driven_freq    = event_driven_freq,
        controller_freq      = controller_freq,
        periodic_freq        = periodic_freq,
        threshold_confidence = threshold_confidence,
        rng                  = np.random.default_rng(seed),
    )

    # 3. Controller bouwen
    controller = Controller(
        trigger            = trigger,
        trains             = trains,
        segments           = segments,
        timetable          = timetable,
        objective_strategy = objective_strategy,
        weight_passenger   = weight_passenger,
        weight_freight     = weight_freight,
        gamma              = dynamic_threshold,
    )

    # 4. Simulatie uitvoeren
    print("Simulatie starten...")
    state = Simulator(
        trains     = trains,
        segments   = segments,
        timetable  = timetable,
        controller = controller,
        seed       = seed,
    ).run()

    # 5. Resultaten verwerken
    print("Resultaten verwerken...")
    df                 = results_to_dataframe(state, trains, timetable)
    controller_summary = controller.summary()

    # 6. Opslaan
    if save:
        meta = _save_results(df, params, controller_summary, output_dir)
    else:
        meta = {**params, "controller_summary": controller_summary}

    print(f"Klaar. {len(df)} rijen, {df['train_id'].nunique()} treinen.")
    print(f"Controller: {controller_summary}")

    return state, df, meta