"""
run_simulation.py

Centrale pipeline voor de thesis-simulatie. Laadt gold-data, configureert
de controller en voert de discrete-event simulatie uit.

Gebruik
-------
    from run_simulation import run_simulation

    state, df, meta, config_dir = run_simulation(
        n_freight          = 182,
        trigger_strategy   = "hybrid",
        objective_strategy = "static",
        seed               = 42,
    )

Deadlock-afhandeling
--------------------
Als de simulator een DeadlockDetected exception gooit, wordt de run als
incomplete gemarkeerd via meta["deadlock_detected"] = True.

Twee dingen worden opgeslagen (als save=True):
  1. Partiële df als parquet met suffix _deadlock
     (bevat segmenten tot op het deadlock-moment)
  2. config_dir wordt altijd teruggegeven zodat de caller
     een result row met deadlock=True kan opslaan

Opslagstructuur (save=True)
---------------------------
    <output_dir>/
      <config_name>/
        <run_name>.parquet           — complete run
        <run_name>_deadlock.parquet  — incomplete run (deadlock)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config.settings as settings
from data.loader           import load_all
from controller.controller import Controller
from controller.triggers   import make_trigger
from simulation.simulator  import Simulator, DeadlockDetected


# =============================================================================
# Helpers
# =============================================================================

def _build_trigger(trigger_strategy, trains, segments, timetable,
                   event_driven_freq, controller_freq,
                   periodic_freq, threshold_confidence, mc_delay_per_train, rng):
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
                            mc_delay_per_train=mc_delay_per_train,
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
                            mc_delay_per_train=mc_delay_per_train,
                            rng=rng)

    raise ValueError(f"Onbekende trigger_strategy: '{trigger_strategy}'. "
                     f"Kies uit: 'periodic', 'event_driven', 'hybrid'.")


def results_to_dataframe(state, trains, timetable):
    """
    Bouwt een DataFrame met één rij per (train_id, segment_id).
    Bij een deadlock-run bevat dit alleen de segmenten die voor
    de deadlock volledig afgerond waren.
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


def _build_config_name(params):
    strategy  = params['trigger_strategy']
    objective = params['objective_strategy']

    base = (
        f"n{params['n_freight']}"
        f"_{strategy}"
        f"_{objective}"
        f"_wp{params['weight_passenger']}"
        f"_wf{params['weight_freight']}"
    )

    if objective == 'dynamic':
        base += f"_g{int(params['dynamic_threshold'])}"

    if strategy == 'periodic':
        base += f"_pf{int(params['periodic_freq'])}"
    elif strategy == 'event_driven':
        base += (
            f"_edf{int(params['event_driven_freq'])}"
            f"_cf{int(params['controller_freq'])}"
            f"_tc{params['threshold_confidence']}"
        )
    elif strategy == 'hybrid':
        base += (
            f"_edf{int(params['event_driven_freq'])}"
            f"_cf{int(params['controller_freq'])}"
            f"_pf{int(params['periodic_freq'])}"
            f"_tc{params['threshold_confidence']}"
        )

    return base


def _build_run_name(params):
    return _build_config_name(params) + f"_seed{params['seed']}"


def _ensure_config_dir(params, output_dir) -> Path:
    """Maak config_dir altijd aan — ook bij deadlock."""
    config_dir = Path(output_dir) / _build_config_name(params)
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _save_results(df, params, config_dir, deadlock: bool = False) -> None:
    """
    Sla df op als parquet.
    Normale run:   <run_name>.parquet
    Deadlock run:  <run_name>_deadlock.parquet
    """
    run_name = _build_run_name(params)
    suffix   = "_deadlock" if deadlock else ""
    df.to_parquet(config_dir / f"{run_name}{suffix}.parquet", index=False)


# =============================================================================
# Hoofdfunctie
# =============================================================================

def run_simulation(
    # Data
    n_freight:            int   = 182,

    # Trigger
    trigger_strategy:     str   = "hybrid",
    event_driven_freq:    float = 900,
    controller_freq:      float = 300,
    periodic_freq:        float = 1800,
    threshold_confidence: float = 0.6,
    mc_delay_per_train:   float = settings.MC_DELAY_PER_TRAIN,

    # MIP
    objective_strategy:   str   = "static",
    weight_passenger:     int   = 1,
    weight_freight:       int   = 1,
    dynamic_threshold:    float = 180,

    # Reproduceerbaarheid
    seed:                 int   = settings.SIMULATION_SEED,

    # Output
    output_dir:           Path | str = settings.RESULTS_DIR,
    save:                 bool  = True,
) -> tuple:
    """
    Voert de volledige simulatiepipeline uit vanuit de gold-bestanden.

    Returns
    -------
    (state, df, meta, config_dir)
        state      : SystemState  — eindtoestand (of toestand op deadlock-moment)
        df         : pd.DataFrame — resultaten per (train_id, segment_id)
                                    partieel bij deadlock
        meta       : dict         — parameters + controller summary +
                                    deadlock_detected (bool)
        config_dir : Path         — pad naar configuratiemap (altijd gevuld)
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

    # config_dir altijd aanmaken — ook als de run later deadlockt
    config_dir = _ensure_config_dir(params, output_dir)

    # 1. Data laden
    print("Loading data...")
    trains, segments, timetable = load_all(n_freight)

    # 2. Trigger bouwen
    print(f"Building controller (trigger={trigger_strategy}, objective={objective_strategy})...")
    trigger = _build_trigger(
        trigger_strategy     = trigger_strategy,
        trains               = trains,
        segments             = segments,
        timetable            = timetable,
        event_driven_freq    = event_driven_freq,
        controller_freq      = controller_freq,
        periodic_freq        = periodic_freq,
        threshold_confidence = threshold_confidence,
        mc_delay_per_train   = mc_delay_per_train,
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
    print("Running simulation...")
    simulator = Simulator(
        trains     = trains,
        segments   = segments,
        timetable  = timetable,
        controller = controller,
        seed       = seed,
    )

    deadlock_detected = False
    try:
        state = simulator.run()
    except DeadlockDetected as e:
        print(f"[DEADLOCK] {e}")
        state = simulator._state
        deadlock_detected = True

    # 5. Resultaten verwerken
    print("Processing results...")
    df                 = results_to_dataframe(state, trains, timetable)
    controller_summary = controller.summary()
    meta               = {
        **params,
        "controller_summary": controller_summary,
        "deadlock_detected":  deadlock_detected,
    }

    # 6. Opslaan
    if save:
        _save_results(df, params, config_dir, deadlock=deadlock_detected)
        run_name = _build_run_name(params)
        suffix   = "_deadlock" if deadlock_detected else ""
        print(f"Saved: {config_dir / run_name}{suffix}.parquet")

    status = "DEADLOCK" if deadlock_detected else "Done"
    print(f"{status}. {len(df)} rows, {df['train_id'].nunique()} trains.")
    print(f"Controller: {controller_summary}")

    return state, df, meta, config_dir