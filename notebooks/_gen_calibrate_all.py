"""
Genereert Calibrate_all.ipynb via nbformat.
Uitvoeren: python3 _gen_calibrate_all.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# TITEL
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# Calibrate_all — Parameter-kalibratie rescheduling systeem

**Datum:** 27-05-2026
**Vaste instellingen voor alle secties:**
- Trigger: `periodic`, freq = 900s
- Prioriteit: `static`, weight_passenger = 1, weight_freight = 1
- Retracking: `True`
- Seeds per configuratie: 25

**Volgorde:**
1. Rescheduling Window × Conflict Window (3×3 interactiegrid)
2. `RETRACK_CONFLICT_WINDOW` (RCW) sweep
3. `SWITCH_PENALTY` × `min_objective_improvement` (5×5 interactiegrid)
4. `mc_delay_per_train` (MC_threshold, event-driven)
5. `min_objective_threshold` — empirische kalibratie via MIP-objectiefverdeling

**Noten:**
- `SWITCH_PENALTY` en `min_objective_improvement` (sectie 3) worden gezamenlijk gekalibreerd
  omdat ze op dezelfde objectiefschaal opereren en een sterk interactie-effect hebben.
- `SWITCH_PENALTY = 0` = geen penalty (vrij switchen); negatief = incentive om te switchen.
- Resultaten worden opgeslagen in `../calibratie27_05/`.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Setup
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""\
import sys, io, warnings
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path.cwd().parent))

import config.settings          as _settings
import model.instance            as _inst_mod
import model.mip_model           as _mip_mod
import simulation.dispatcher     as _disp_mod
from run_simulation import run_simulation
from utils.metrics  import compute_metrics

CALIB_DIR = Path.cwd().parent / 'calibratie27_05'
CALIB_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS   = 25
N_FREIGHT = 182

# Vaste basisconfig voor alle experimenten
BASE_CFG = dict(
    n_freight          = N_FREIGHT,
    trigger_strategy   = 'periodic',
    periodic_freq      = 900,
    event_driven_freq  = 900,
    controller_freq    = 900,
    objective_strategy = 'static',
    weight_passenger   = 1,
    weight_freight     = 1,
    use_retracking     = True,
    save               = False,
)

sns.set_theme(style='whitegrid', palette='muted')
print(f'Output: {CALIB_DIR}')
print(f'Seeds per config: {N_SEEDS}')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Settings patcher + run_batch
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_code_cell("""\
# ── Settings monkey-patching ─────────────────────────────────────────────────
# Settings zoals RESCHEDULING_HORIZON worden op module-niveau geïmporteerd.
# We moeten zowel config.settings als de consumerende modules patchen.
_PATCH_TARGETS = {
    'RESCHEDULING_HORIZON':             [_settings, _inst_mod],
    'CONFLICT_WINDOW':                  [_settings, _inst_mod, _mip_mod],
    'RETRACK_CONFLICT_WINDOW':          [_settings, _mip_mod],
    'SWITCH_PENALTY':                   [_settings, _mip_mod],
}
_originals: dict = {}

def patch_settings(**kwargs):
    for attr, val in kwargs.items():
        if attr not in _originals:
            _originals[attr] = getattr(_settings, attr)
        for mod in _PATCH_TARGETS.get(attr, [_settings]):
            if hasattr(mod, attr):
                setattr(mod, attr, val)

def restore_settings():
    for attr, orig in list(_originals.items()):
        for mod in _PATCH_TARGETS.get(attr, [_settings]):
            if hasattr(mod, attr):
                setattr(mod, attr, orig)
    _originals.clear()


# ── Objectives log (gevuld door run_batch) ───────────────────────────────────
_objectives_log: dict = {}


# ── Deadlock-hulpfuncties ─────────────────────────────────────────────────────
def _split_deadlocks(df: pd.DataFrame, group_cols):
    \"\"\"
    Filtert deadlocked seeds uit df en telt ze per groep.

    Retourneert (df_clean, dl_counts) waarbij:
      df_clean  : df zonder deadlocked rijen (voor metric-berekeningen)
      dl_counts : pd.Series { groep → n_deadlocks }  (voor plot-annotaties)

    Gebruik dl_counts.get(groepswaarde, 0) om het aantal per x-positie op te vragen.
    \"\"\"
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    # Defensieve cast: CSV-roundtrip levert strings "True"/"False" i.p.v. bool.
    # astype(bool) werkt niet: bool("False") == True. Gebruik string-vergelijking.
    df = df.copy()
    df['deadlock'] = df['deadlock'].astype(str).str.lower() == 'true'
    df_clean  = df[~df['deadlock']].copy()
    dl_counts = df.groupby(group_cols)['deadlock'].sum().astype(int)
    n_total   = len(df)
    n_dl      = int(df['deadlock'].sum())
    if n_dl:
        pct = n_dl / n_total * 100
        print(f'  ⚠  {n_dl}/{n_total} seeds geëxcludeerd wegens deadlock ({pct:.1f}%)'
              f' — metrics berekend op schone subset.')
    else:
        print('  Geen deadlocks gedetecteerd.')
    return df_clean, dl_counts


def _annotate_deadlocks(ax, positions, group_values, dl_counts):
    \"\"\"
    Voegt rode ⚠-labels toe ónder de x-ticks van een lijnplot
    voor elke positie waar minstens één seed deadlockte.
    Retourneert True als er annotaties zijn geplaatst.
    \"\"\"
    _any = False
    for pos, gv in zip(list(positions), list(group_values)):
        # dl_counts is een pd.Series; .get() werkt ook op tuples (multi-index)
        _key = gv if not isinstance(gv, (list,)) else tuple(gv)
        ndl  = int(dl_counts.get(_key, 0))
        if ndl > 0:
            _any = True
            ax.text(
                pos, 0, f'⚠ {ndl}×DL',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8, color='#CC0000',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFE5E5', alpha=0.85),
            )
    return _any


# ── run_batch ────────────────────────────────────────────────────────────────
def run_batch(
    settings_override: dict,
    n_seeds: int = N_SEEDS,
    run_kwargs: dict | None = None,
    label: str = '',
    verbose: bool = True,
    checkpoint_csv: 'Path | None' = None,
) -> pd.DataFrame:
    \"\"\"
    Voert n_seeds simulaties uit met gegeven settings-override.
    Retourneert DataFrame met één rij per seed.

    settings_override  : module-level settings tijdelijk overschrijven
    run_kwargs         : extra kwargs voor run_simulation — OVERSCHRIJVEN BASE_CFG
                         (b.v. min_objective_improvement, trigger_strategy, ...)
    label              : labelkolom in het resultaat; ook sleutel in _objectives_log
    checkpoint_csv     : pad naar sectie-CSV voor resume-support.
                         • Al gedane (label, seed) paren worden overgeslagen.
                         • Elke nieuw voltooide seed wordt meteen toegevoegd.
                         • Bij herstart laadt de caller de CSV via pd.read_csv().
    \"\"\"
    extra = run_kwargs or {}
    _objectives_log.setdefault(label, [])

    # Merge: run_kwargs overschrijven BASE_CFG (zodat b.v. trigger_strategy kan wijzigen)
    cfg = {**BASE_CFG, **extra}

    # ── Checkpoint: welke seeds zijn al gedaan voor dit label? ────────────────
    existing_rows: list[dict] = []
    done_seeds: set[int]      = set()
    if checkpoint_csv is not None and checkpoint_csv.exists():
        try:
            _ck = pd.read_csv(checkpoint_csv)
            _done = _ck[_ck['label'] == label].copy()
            # CSV-roundtrip converteert bool → str; zet terug naar bool.
            if 'deadlock' in _done.columns:
                # astype(bool) werkt niet voor strings: bool("False") == True.
                # Gebruik expliciete string-vergelijking.
                _done['deadlock'] = _done['deadlock'].astype(str).str.lower() == 'true'
            done_seeds = set(_done['seed'].astype(int).tolist())
            existing_rows = _done.to_dict('records')
            if done_seeds:
                remaining = n_seeds - len(done_seeds)
                print(
                    f'  [{label}] checkpoint: {len(done_seeds)}/{n_seeds} seeds '
                    f'al klaar — nog {remaining} te gaan.'
                )
        except Exception as _e:
            print(f'  [{label}] checkpoint lezen mislukt ({_e}) — opnieuw starten.')
            existing_rows, done_seeds = [], set()

    # _wrote_header: True als het bestand al bestaat (ongeacht dit label).
    # Nooit een header schrijven naar een bestaand bestand, ook niet bij het
    # eerste nieuwe label — anders ontstaat een dubbele header midden in de CSV.
    _wrote_header: bool = checkpoint_csv is not None and checkpoint_csv.exists()

    new_rows: list[dict] = []
    for seed in range(n_seeds):
        if seed in done_seeds:
            continue

        patch_settings(**settings_override)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                _, df, meta, _, _ = run_simulation(**cfg, seed=seed)
        finally:
            restore_settings()

        ctrl       = meta['controller_summary']
        m          = compute_metrics(df)
        total      = max(1, ctrl.get('total_steps', 1))
        n_eval     = max(1, ctrl.get('n_evaluated', total))   # MC evaluaties (event-driven)

        # MIP-objectiefwaarden bijhouden voor sectie 7/8
        objs = ctrl.get('solution_objectives', [])
        _objectives_log[label].extend(objs)

        row = {
            'seed':                     seed,
            'label':                    label,
            **settings_override,
            **{k: v for k, v in extra.items()},
            'TED_combined':             m.get('TED_combined'),
            'TED_passenger':            m.get('TED_passenger'),
            'TED_freight':              m.get('TED_freight'),
            'delay_ratio':              m.get('delay_ratio'),
            'n_platform_switches':      ctrl.get('n_platform_switches', 0),
            'n_rescheduled':            ctrl['n_rescheduled'],
            'n_evaluated':              ctrl.get('n_evaluated', 0),
            'n_skipped_no_improvement': ctrl.get('n_skipped_no_improvement', 0),
            'total_steps':              total,
            'total_solve_time_s':       ctrl.get('total_solver_runtime_s', 0),
            'mean_solve_time_s':        ctrl.get('total_solver_runtime_s', 0)
                                        / max(1, ctrl['n_rescheduled']),
            # trigger_rate_pct: voor periodic = n_rescheduled/total_steps
            #                   voor event-driven: gebruik trigger_rate_mc_pct
            'trigger_rate_pct':         ctrl['n_rescheduled'] / total * 100,
            # trigger_rate_mc_pct: n_rescheduled / n_evaluated (MC-specifiek)
            'trigger_rate_mc_pct':      ctrl['n_rescheduled'] / n_eval * 100,
            'deadlock':                 meta.get('deadlock_detected', False),
        }
        new_rows.append(row)

        # ── Per-seed checkpoint: meteen opslaan ──────────────────────────────
        if checkpoint_csv is not None:
            pd.DataFrame([row]).to_csv(
                checkpoint_csv,
                mode='a',
                header=not _wrote_header,
                index=False,
            )
            _wrote_header = True

        if verbose and (seed + 1) % 5 == 0:
            ted = m.get('TED_combined')
            ted_s = f'{ted:.0f}' if ted is not None else '?'
            print(f'  [{label}] seed {seed+1:2d}/{n_seeds}  TED={ted_s}s')

    df_out = pd.DataFrame(existing_rows + new_rows)
    n_dl = df_out['deadlock'].sum() if len(df_out) else 0
    if n_dl:
        print(f'  ⚠  {n_dl}/{len(df_out)} rijen hadden een deadlock!')
    return df_out


print('Helpers geladen.')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTIE 1
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Sectie 1 — Rescheduling Window × Conflict Window (3×3)

**Doel:** interactie-effect tussen de twee vensterparameters kwantificeren.

| Parameter | Waarden | Eenheid |
|---|---|---|
| `RESCHEDULING_HORIZON` (RW) | 1800, 3600, 5400 | s |
| `CONFLICT_WINDOW` (CW) | 600, 1200, 1800 | s |

**Metrics:** TED_combined (kwaliteit) + mean solve time per reschedule (complexiteit)
"""))

cells.append(new_code_cell("""\
print('=' * 60)
print('SECTIE 1: Rescheduling Window × Conflict Window')
print('=' * 60)

RW_VALUES = [1800, 3600, 5400]
CW_VALUES = [600, 1200, 1800]

_s1_csv  = CALIB_DIR / 's1_window_grid.csv'
s1_parts = []
for rw in RW_VALUES:
    for cw in CW_VALUES:
        lbl = f'RW{rw}_CW{cw}'
        print(f'\\n--- RW={rw}s  CW={cw}s ---')
        batch = run_batch(
            settings_override={'RESCHEDULING_HORIZON': rw, 'CONFLICT_WINDOW': cw},
            label=lbl,
            checkpoint_csv=_s1_csv,
        )
        s1_parts.append(batch)

df_s1 = pd.concat(s1_parts, ignore_index=True)
df_s1.to_csv(_s1_csv, index=False)
print(f'\\nKlaar. {len(df_s1)} rijen opgeslagen in {_s1_csv.name}')
"""))

cells.append(new_code_cell("""\
# ── Visualisatie 1a: 3×3 boxplot-grid ────────────────────────────────────────
# Deadlocked seeds worden geëxcludeerd van de boxplots maar gemarkeerd.
df_clean_s1, dl_s1 = _split_deadlocks(df_s1, ['RESCHEDULING_HORIZON', 'CONFLICT_WINDOW'])

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
fig.suptitle(
    'TED_combined (s) — Rescheduling Window × Conflict Window\\n'
    f'(n={N_SEEDS} seeds per combinatie; ⚠ = deadlocked seeds geëxcludeerd)',
    fontsize=14, fontweight='bold'
)

colors = ['#74B9E0', '#55A3D0', '#3A8EC0']  # donkerder per RW-waarde

for i, rw in enumerate(RW_VALUES):
    for j, cw in enumerate(CW_VALUES):
        ax   = axes[i][j]
        sub  = df_clean_s1[
            (df_clean_s1['RESCHEDULING_HORIZON'] == rw) &
            (df_clean_s1['CONFLICT_WINDOW'] == cw)
        ]
        vals = sub['TED_combined'].dropna().values
        ndl  = int(dl_s1.get((rw, cw), 0))

        bp = ax.boxplot(
            vals, patch_artist=True, widths=0.55,
            medianprops=dict(color='black', linewidth=2.0),
            flierprops=dict(marker='.', markersize=4, alpha=0.5),
        )
        bp['boxes'][0].set_facecolor(colors[i])
        bp['boxes'][0].set_alpha(0.8 if ndl == 0 else 0.45)

        title = f'RW = {rw//60} min\\nCW = {cw//60} min'
        if ndl:
            title += f'\\n⚠ {ndl} DL excl.'
        ax.set_title(title, fontsize=9, fontweight='bold',
                     color='black' if ndl == 0 else '#CC0000')
        ax.set_xticks([])
        if j == 0:
            ax.set_ylabel('TED_combined (s)', fontsize=9)

        if len(vals):
            ax.text(
                0.97, 0.96,
                f'μ={vals.mean():.0f}\\nσ={vals.std():.0f}\\nn={len(vals)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            )
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(CALIB_DIR / 's1_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(new_code_cell("""\
# ── Visualisatie 1b: twee heatmaps (TED + solve-time) ────────────────────────
df_clean_s1, dl_s1 = _split_deadlocks(df_s1, ['RESCHEDULING_HORIZON', 'CONFLICT_WINDOW'])

summary_s1 = df_clean_s1.groupby(['RESCHEDULING_HORIZON', 'CONFLICT_WINDOW']).agg(
    TED_mean    = ('TED_combined',      'mean'),
    TED_std     = ('TED_combined',      'std'),
    solve_mean  = ('mean_solve_time_s', 'mean'),
    solve_std   = ('mean_solve_time_s', 'std'),
).round(2)
summary_s1['n_deadlocks'] = dl_s1

pivot_ted   = summary_s1['TED_mean'].unstack(level='CONFLICT_WINDOW')
pivot_solve = summary_s1['solve_mean'].unstack(level='CONFLICT_WINDOW')
pivot_dl    = summary_s1['n_deadlocks'].unstack(level='CONFLICT_WINDOW').fillna(0).astype(int)

# Labels in minuten voor leesbaarheid
_rlbls = [f'{v//60} min' for v in pivot_ted.index]
_clbls = [f'CW {v//60} min' for v in pivot_ted.columns]
for p in [pivot_ted, pivot_solve, pivot_dl]:
    p.index   = _rlbls
    p.columns = _clbls

# Annotaties: "waarde\\n(N DL)" als er deadlocks zijn in die cel
def _hm_annot(pivot_val, pivot_dl, fmt):
    arr = []
    for i in range(len(pivot_val)):
        row = []
        for j in range(len(pivot_val.columns)):
            v   = pivot_val.iloc[i, j]
            ndl = pivot_dl.iloc[i, j]
            txt = fmt.format(v)
            if ndl > 0:
                txt += f'\\n(⚠{ndl}DL)'
            row.append(txt)
        arr.append(row)
    return arr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Interactie-effect Rescheduling Window × Conflict Window\\n'
    '(⚠N DL = N deadlocked seeds geëxcludeerd)',
    fontsize=13, fontweight='bold'
)

sns.heatmap(
    pivot_ted.round(0).astype(int), ax=ax1,
    annot=_hm_annot(pivot_ted.round(0), pivot_dl, '{:.0f}'), fmt='',
    cmap='RdYlGn_r', cbar_kws={'label': 'Mean TED_combined (s)'},
)
ax1.set_title('Mean TED_combined (s) — schone seeds\\n← lager = beter', fontsize=11)
ax1.set_xlabel('Conflict Window')
ax1.set_ylabel('Rescheduling Window')

sns.heatmap(
    pivot_solve.round(3), ax=ax2,
    annot=_hm_annot(pivot_solve.round(3), pivot_dl, '{:.3f}'), fmt='',
    cmap='YlOrRd', cbar_kws={'label': 'Mean solve time / reschedule (s)'},
)
ax2.set_title('Mean solve time per reschedule (s)\\n← lager = sneller MIP', fontsize=11)
ax2.set_xlabel('Conflict Window')
ax2.set_ylabel('Rescheduling Window')

plt.tight_layout()
plt.savefig(CALIB_DIR / 's1_heatmaps.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\nSamenvatting (mean ± std TED_combined, schone seeds):')
print(summary_s1[['TED_mean', 'TED_std', 'solve_mean', 'n_deadlocks']].to_string())
"""))

cells.append(new_code_cell("""\
# ── Beste combo selectie ──────────────────────────────────────────────────────
idx_best = summary_s1['TED_mean'].idxmin()
BEST_RW, BEST_CW = int(idx_best[0]), int(idx_best[1])

print(f'Auto-geselecteerd op basis van laagste TED_combined:')
print(f'  RESCHEDULING_HORIZON = {BEST_RW}s  ({BEST_RW//60} min)')
print(f'  CONFLICT_WINDOW      = {BEST_CW}s  ({BEST_CW//60} min)')
print(f'  TED_combined         = {summary_s1.loc[idx_best, \"TED_mean\"]:.0f} ± '
      f'{summary_s1.loc[idx_best, \"TED_std\"]:.0f}s')
print()
print('Pas eventueel handmatig aan (trade-off TED vs solve-time):')
print('  # BEST_RW = 3600')
print('  # BEST_CW = 1200')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTIE 2  —  RETRACK_CONFLICT_WINDOW sweep
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""---
## Sectie 2 — `RETRACK_CONFLICT_WINDOW` (RCW) sweep

**Doel:** kalibreer het tijdvenster waarbinnen treinen in aanmerking komen voor een platformwissel.

> **`RETRACK_CONFLICT_WINDOW` (RCW):** twee treinen zijn potentieel retracking-conflicterend
> als hun verwachte exittijden op het station maximaal RCW seconden uit elkaar liggen.
> Een groter venster geeft de solver meer alternatieven, maar vergroot ook het MIP.

> **Vaste parameters:** `SWITCH_PENALTY = 0` (neutraal), `min_objective_improvement = −10000`
> (geen filter) — zo is het retracking-gedrag zichtbaar zonder invloed van de filter.
> SP en MOI worden pas in § 3 samen gekalibreerd.

| Parameter | Waarden |
|---|---|
| `RETRACK_CONFLICT_WINDOW` | 300, 600, 900, 1200, 1500 s |

**Metrics:** TED_combined (kwaliteit) + mean solve time per reschedule (complexiteit) + deadlock-rate
"""))

cells.append(new_code_cell("""print('=' * 60)
print('SECTIE 2: RETRACK_CONFLICT_WINDOW sweep')
print(f'  Vaste instellingen: RW={BEST_RW}s, CW={BEST_CW}s')
print('  SWITCH_PENALTY=0, MOI=-10000 (defaults vóór § 3 kalibratie)')
print('=' * 60)

RCW_VALUES = [300, 600, 900, 1200, 1500]

# SP en MOI worden pas in § 3 gekalibreerd; gebruik neutrale defaults hier.
_DEFAULT_SP  = 0.0
_DEFAULT_MOI = -10000.0

_s2_csv  = CALIB_DIR / 's2_rcw_sweep.csv'
s2_parts = []
for rcw in RCW_VALUES:
    lbl = f'RCW{rcw}'
    print(f'\\n--- RCW={rcw}s ---')
    batch = run_batch(
        settings_override={
            'RESCHEDULING_HORIZON':    BEST_RW,
            'CONFLICT_WINDOW':         BEST_CW,
            'SWITCH_PENALTY':          _DEFAULT_SP,
            'RETRACK_CONFLICT_WINDOW': rcw,
        },
        run_kwargs={'min_objective_improvement': _DEFAULT_MOI},
        label=lbl,
        checkpoint_csv=_s2_csv,
    )
    batch['retrack_conflict_window'] = rcw
    s2_parts.append(batch)

df_s2 = pd.concat(s2_parts, ignore_index=True)
df_s2.to_csv(_s2_csv, index=False)
print(f'\\nKlaar. {len(df_s2)} rijen opgeslagen in {_s2_csv.name}')
"""))

cells.append(new_code_cell("""# ── Visualisatie 2: lijnplot (TED + solve-time, dual-axis) ───────────────────
df_clean_s2, dl_s2 = _split_deadlocks(df_s2, 'retrack_conflict_window')

summary_s2 = df_clean_s2.groupby('retrack_conflict_window').agg(
    TED_mean   = ('TED_combined',     'mean'),
    TED_std    = ('TED_combined',     'std'),
    solve_mean = ('mean_solve_time_s','mean'),
    solve_std  = ('mean_solve_time_s','std'),
).reset_index()

dl_s2_counts = df_s2.groupby('retrack_conflict_window')['deadlock'].sum().astype(int).reset_index()
dl_s2_counts.columns = ['retrack_conflict_window', 'n_deadlocks']
summary_s2 = summary_s2.merge(dl_s2_counts, on='retrack_conflict_window', how='left')
summary_s2['n_deadlocks'] = summary_s2['n_deadlocks'].fillna(0).astype(int)

rcw_vals = summary_s2['retrack_conflict_window'].values
rcw_pos  = list(range(len(rcw_vals)))
rcw_lbls = [f'{v}s' for v in rcw_vals]

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.suptitle(
    f'RETRACK_CONFLICT_WINDOW sweep\\n'
    f'(RW={BEST_RW}s, CW={BEST_CW}s, SP=0, MOI=−10000, {N_SEEDS} seeds | excl. deadlocked seeds)',
    fontsize=12, fontweight='bold',
)

c_ted   = '#4C72B0'
c_solve = '#DD8452'

ax1.plot(rcw_pos, summary_s2['TED_mean'], 'o-', color=c_ted, lw=2.5, ms=8,
         label='TED_combined (μ, excl. DL)')
ax1.fill_between(
    rcw_pos,
    summary_s2['TED_mean'] - summary_s2['TED_std'],
    summary_s2['TED_mean'] + summary_s2['TED_std'],
    alpha=0.18, color=c_ted, label='±1 std TED',
)
ax1.set_xticks(rcw_pos)
ax1.set_xticklabels(rcw_lbls, fontsize=10)
ax1.set_xlabel('RETRACK_CONFLICT_WINDOW (s)', fontsize=11)
ax1.set_ylabel('TED_combined (s)', color=c_ted, fontsize=11)
ax1.tick_params(axis='y', labelcolor=c_ted)

ax2 = ax1.twinx()
ax2.plot(rcw_pos, summary_s2['solve_mean'], 's--', color=c_solve, lw=1.8, ms=7, alpha=0.85,
         label='Mean solve time/reschedule (s)')
ax2.fill_between(rcw_pos,
    summary_s2['solve_mean'] - summary_s2['solve_std'],
    summary_s2['solve_mean'] + summary_s2['solve_std'],
    alpha=0.12, color=c_solve,
)
ax2.set_ylabel('Mean solve time per reschedule (s)', color=c_solve, fontsize=11)
ax2.tick_params(axis='y', labelcolor=c_solve)

_annotate_deadlocks(ax1, rcw_pos, rcw_vals, dl_s2.reindex(rcw_vals, fill_value=0))

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(CALIB_DIR / 's2_rcw_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\nNumeriek overzicht (excl. deadlocked seeds):')
print(summary_s2[['retrack_conflict_window','TED_mean','TED_std','solve_mean','n_deadlocks']].to_string(index=False))
"""))

cells.append(new_code_cell("""# ── Beste waarde ───────────────────────────────────────────────────────────────────
safe_s2 = summary_s2[summary_s2['n_deadlocks'] == 0].copy()
if safe_s2.empty:
    safe_s2 = summary_s2.copy()
    print('⚠ Geen deadlock-vrije waarde — selectie op laagste TED.')

best2    = safe_s2.sort_values(['TED_mean', 'retrack_conflict_window']).iloc[0]
BEST_RCW = int(best2['retrack_conflict_window'])

print(f'Auto-geselecteerd (laagste TED, deadlock-vrij):')
print(f'  RETRACK_CONFLICT_WINDOW = {BEST_RCW}s')
print()
print('Pas handmatig aan indien gewenst:')
print('  # BEST_RCW = 900')
"""))

# SECTIE 3  —  SWITCH_PENALTY × min_objective_improvement (2D grid)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Sectie 3 — `SWITCH_PENALTY` × `min_objective_improvement` (5×5 interactiegrid)

**Doel:** gezamenlijke kalibratie van beide parameters, die op dezelfde MIP-objectiefschaal
opereren en daardoor een structureel interactie-effect hebben.

> **Waarom samen calibreren?**
> `SWITCH_PENALTY` wordt opgeteld bij het MIP-objectief per platformwissel.
> `min_objective_improvement` is een drempel op datzelfde objectief.
> Bij hoge `SWITCH_PENALTY` zijn objectiefverbeteringen groter in absolute waarde,
> waardoor dezelfde drempel meer of minder oplossingen doorlaat dan bij lage penalty.
> Sequentiële kalibratie mist dit effect volledig.

| Parameter | Waarden |
|---|---|
| `SWITCH_PENALTY` | −10000, 60, 120, 180, 300 |
| `min_objective_improvement` | −10000, 60, 120, 180, 300 |

**Visualisatie:** twee heatmaps naast elkaar
- Links: TED_combined (μ, excl. deadlocked seeds)
- Rechts: n_deadlocks per cel
→ Kies de cel met laagste TED én nul deadlocks.
"""))

cells.append(new_code_cell("""\
print('=' * 60)
print('SECTIE 3: SWITCH_PENALTY × min_objective_improvement grid')
print(f'  Vaste instellingen: RW={BEST_RW}s, CW={BEST_CW}s')
print('=' * 60)

SP_GRID_VALUES  = [-10000, 60, 120, 180, 300]
MOI_GRID_VALUES = [-10000, 60, 120, 180, 300]

_s3_csv   = CALIB_DIR / 's3_sp_moi_grid.csv'
s3_parts  = []
for sp in SP_GRID_VALUES:
    for moi in MOI_GRID_VALUES:
        lbl = f'SP{sp}_MOI{moi}'
        print(f'\\n--- SP={sp}, MOI={moi} ---')
        batch = run_batch(
            settings_override={
                'RESCHEDULING_HORIZON': BEST_RW,
                'CONFLICT_WINDOW':      BEST_CW,
                'RETRACK_CONFLICT_WINDOW': BEST_RCW,
                'SWITCH_PENALTY':         float(sp),
            },
            run_kwargs={'min_objective_improvement': float(moi)},
            label=lbl,
            checkpoint_csv=_s3_csv,
        )
        batch['switch_penalty']       = sp
        batch['min_obj_improvement']  = moi
        s3_parts.append(batch)

df_s3 = pd.concat(s3_parts, ignore_index=True)
df_s3.to_csv(_s3_csv, index=False)
print(f'\\nKlaar. {len(df_s3)} rijen opgeslagen in {_s3_csv.name}')
"""))

cells.append(new_code_cell("""\
# ── Visualisatie 3: dubbele heatmap (TED | deadlocks) ───────────────────────
df_clean_s3, dl_s3 = _split_deadlocks(df_s3, ['switch_penalty', 'min_obj_improvement'])

summary_s3 = df_clean_s3.groupby(['switch_penalty', 'min_obj_improvement']).agg(
    TED_mean = ('TED_combined', 'mean'),
    TED_std  = ('TED_combined', 'std'),
).reset_index()

dl_s3_df = df_s3.groupby(['switch_penalty', 'min_obj_improvement'])['deadlock'].sum().astype(int).reset_index()
dl_s3_df.columns = ['switch_penalty', 'min_obj_improvement', 'n_deadlocks']

summary_s3 = summary_s3.merge(dl_s3_df, on=['switch_penalty', 'min_obj_improvement'], how='left')
summary_s3['n_deadlocks'] = summary_s3['n_deadlocks'].fillna(0).astype(int)

SP_LBLS  = ['incentive\\n(−10k)' if v == -10000 else str(v) for v in SP_GRID_VALUES]
MOI_LBLS = ['geen\\n(−10k)'     if v == -10000 else str(v) for v in MOI_GRID_VALUES]

pivot_ted = summary_s3.pivot(index='switch_penalty', columns='min_obj_improvement', values='TED_mean')
pivot_dl  = summary_s3.pivot(index='switch_penalty', columns='min_obj_improvement', values='n_deadlocks').fillna(0).astype(int)

annot_arr = _hm_annot(pivot_ted, pivot_dl, '{:.0f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f'SWITCH_PENALTY × min_objective_improvement kalibratie\\n'
    f'(RW={BEST_RW}s, CW={BEST_CW}s, {N_SEEDS} seeds | ⚠NDL = deadlocked seeds geëxcludeerd)',
    fontsize=12, fontweight='bold',
)

import seaborn as sns
sns.heatmap(
    pivot_ted.round(0).astype(int), ax=ax1,
    annot=annot_arr, fmt='',
    cmap='RdYlGn_r', linewidths=0.5,
    xticklabels=MOI_LBLS, yticklabels=SP_LBLS,
    cbar_kws={'label': 'TED_combined (s)'},
)
ax1.set_title('TED_combined (μ, excl. deadlocked seeds)', fontsize=11)
ax1.set_xlabel('min_objective_improvement', fontsize=10)
ax1.set_ylabel('SWITCH_PENALTY', fontsize=10)

sns.heatmap(
    pivot_dl, ax=ax2,
    annot=True, fmt='d',
    cmap='Reds', linewidths=0.5,
    xticklabels=MOI_LBLS, yticklabels=SP_LBLS,
    cbar_kws={'label': 'n_deadlocks'},
)
ax2.set_title('Aantal deadlocks per cel', fontsize=11)
ax2.set_xlabel('min_objective_improvement', fontsize=10)
ax2.set_ylabel('SWITCH_PENALTY', fontsize=10)

plt.tight_layout()
plt.savefig(CALIB_DIR / 's3_sp_moi_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\nNumeriek overzicht (excl. deadlocked seeds):')
print(summary_s3[['switch_penalty','min_obj_improvement','TED_mean','TED_std','n_deadlocks']].to_string(index=False))
"""))

cells.append(new_code_cell("""\
# ── Beste combinatie ──────────────────────────────────────────────────────────
# Stap 1: exclusief deadlock-vrije cellen.
# Stap 2: laagste TED_mean; bij gelijkspel kleinste SP dan kleinste MOI.
safe_s3 = summary_s3[summary_s3['n_deadlocks'] == 0].copy()
if safe_s3.empty:
    safe_s3 = summary_s3.copy()
    print('⚠ Geen deadlock-vrije combinatie gevonden — selectie op laagste TED.')

best_row = safe_s3.sort_values(['TED_mean', 'switch_penalty', 'min_obj_improvement']).iloc[0]
BEST_SP  = float(best_row['switch_penalty'])
BEST_MOI = float(best_row['min_obj_improvement'])

print(f'Auto-geselecteerd (laagste TED, deadlock-vrij):')
print(f'  SWITCH_PENALTY          = {BEST_SP}')
print(f'  min_objective_improvement = {BEST_MOI}')
print()
print('Pas handmatig aan indien gewenst:')
print('  # BEST_SP  = 120.0')
print('  # BEST_MOI = 120.0')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTIE 4 (verwijderd) — DISPATCHER_PRIORITY_TTL
# De TTL is verwijderd uit de codebase: de dispatcher gebruikt altijd MIP-prioriteit.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SECTIE 5
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Sectie 5 — `mc_delay_per_train` (MC_threshold) kalibratie

**Doel:** kalibreer de drempel voor de Monte Carlo trigger in event-driven/hybrid strategieën.

> **Hoe werkt het?**
> Bij elke evaluatie berekent de event-driven trigger:
> ```
> threshold = avg_delay_per_train(state) + mc_delay_per_train
> ```
> De solver vuurt als ≥ `threshold_confidence` van de MC rollouts de `threshold` overschrijden.
> Dit betekent: "verwachten we meer dan `mc_delay_per_train` seconden EXTRA vertraging
> per trein t.o.v. de huidige toestand?"
>
> **Waarden 20–240s:** laag = solver vuurt snel bij kleine verwachte verslechtering;
> hoog = solver wacht tot duidelijke verslechtering voorspeld wordt.

> ⚠ **Trigger-strategie hier: event-driven** (niet periodic).
> Vaste parameters: `event_driven_freq=900s`, `controller_freq=900s`,
> `threshold_confidence=0.6`, `mc_iterations=5`.
>
> **Waarom `controller_freq = event_driven_freq`?**
> Met `controller_freq < event_driven_freq` zou MC tussen twee reschedules meerdere
> keren evalueren (elke 300s) vóór hij vuurt. Een hoge `mc_delay_per_train` vuurt
> minder snel → meer evaluaties per cyclus → de noemer `n_evaluated` stijgt, niet
> omdat het systeem vaker gecontroleerd wordt maar omdat de drempel moeilijker
> te bereiken is. Dit vertekent `trigger_rate_mc_pct`.
> Met `controller_freq = event_driven_freq = 900s` draait MC precies één keer per
> 900s-venster, ongeacht of hij vuurt. De trigger rate meet dan zuiver:
> *"welke fractie van 900s-vensters resulteerde in een reschedule?"*

| Parameter | Waarden |
|---|---|
| `mc_delay_per_train` | 20, 60, 120, 180, 240 s |

**Metrics:**
- **trigger rate % (MC)** = `n_rescheduled / n_evaluated × 100`
  (n_evaluated = aantal keer dat MC effectief gedraaid heeft)
- TED_combined (kwaliteit vs baseline)

**Visualisatie:** heatmap (seed × mc_delay_per_train, kleur = MC trigger rate %)
"""))

cells.append(new_code_cell("""\
print('=' * 60)
print('SECTIE 5: mc_delay_per_train sweep (event-driven trigger)')
print(f'  Vaste vensters: RW={BEST_RW}s, CW={BEST_CW}s')
print('  Trigger: event_driven | event_driven_freq=controller_freq=900s')
print('  threshold_confidence=0.6 | mc_iterations=5')
print('=' * 60)

MC_VALUES = [20, 60, 120, 180, 240]

# Event-driven trigger config (gemeenschappelijk voor alle mc_delay_per_train waarden).
# controller_freq == event_driven_freq zodat MC precies 1× per 900s-venster draait.
# Dit maakt trigger_rate_mc_pct = n_rescheduled / n_evaluated vergelijkbaar
# over alle mc_delay_per_train waarden (vaste noemer ≈ sim_duration / 900).
ED_CFG = dict(
    trigger_strategy     = 'event_driven',
    event_driven_freq    = 900,
    controller_freq      = 900,   # == event_driven_freq: één MC run per venster
    threshold_confidence = 0.6,
)

_s5_csv  = CALIB_DIR / 's5_mc_delay_per_train.csv'
s5_parts = []
for mc in MC_VALUES:
    lbl = f'MC{mc}'
    print(f'\\n--- mc_delay_per_train={mc}s ---')
    batch = run_batch(
        settings_override={'RESCHEDULING_HORIZON': BEST_RW, 'CONFLICT_WINDOW': BEST_CW},
        run_kwargs={**ED_CFG, 'mc_delay_per_train': float(mc)},
        label=lbl,
        checkpoint_csv=_s5_csv,
    )
    batch['mc_delay_per_train'] = mc
    s5_parts.append(batch)

df_s5 = pd.concat(s5_parts, ignore_index=True)
df_s5.to_csv(_s5_csv, index=False)
print(f'\\nKlaar. {len(df_s5)} rijen opgeslagen in {_s5_csv.name}')
"""))

cells.append(new_code_cell("""\
# ── Visualisatie 2: heatmap (seed × mc_delay_per_train) + lijnplot ───────────
# Deadlocked seeds: zichtbaar in de heatmap (⚠DL-label), geëxcludeerd uit lijnplot.
df_clean_s5, dl_s5 = _split_deadlocks(df_s5, 'mc_delay_per_train')

# Pivot trigger rate — alle seeds (ook deadlocked), zodat ze zichtbaar blijven
pivot_trig = df_s5.pivot_table(
    index='seed', columns='mc_delay_per_train',
    values='trigger_rate_mc_pct', aggfunc='first',
)
pivot_trig.columns = [f'{int(c)}s' for c in pivot_trig.columns]

# Deadlock-masker voor heatmap annotaties
pivot_dl2 = df_s5.pivot_table(
    index='seed', columns='mc_delay_per_train',
    values='deadlock', aggfunc='first',
).fillna(False).astype(bool)
pivot_dl2.columns = [f'{int(c)}s' for c in pivot_dl2.columns]

# Bouw aangepaste annotatiematrix: "waarde" of "waarde\\n⚠DL"
annot_trig = []
for seed in pivot_trig.index:
    row = []
    for col in pivot_trig.columns:
        v   = pivot_trig.loc[seed, col]
        is_dl = bool(pivot_dl2.loc[seed, col]) if seed in pivot_dl2.index else False
        row.append(f'{v:.1f}\\n⚠DL' if is_dl else f'{v:.1f}')
    annot_trig.append(row)

# Lijnplot: metrics enkel op schone seeds
summary_s5 = df_clean_s5.groupby('mc_delay_per_train').agg(
    TED_mean      = ('TED_combined',        'mean'),
    TED_std       = ('TED_combined',        'std'),
    trig_mc_mean  = ('trigger_rate_mc_pct', 'mean'),
    trig_mc_std   = ('trigger_rate_mc_pct', 'std'),
    n_eval_mean   = ('n_evaluated',         'mean'),
    n_resc_mean   = ('n_rescheduled',       'mean'),
).reset_index()
summary_s5['n_deadlocks'] = dl_s5.values

fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.suptitle(
    f'mc_delay_per_train kalibratie (event-driven, controller_freq=event_driven_freq=900s)\\n'
    f'(RW={BEST_RW}s, CW={BEST_CW}s, threshold_confidence=0.6, {N_SEEDS} seeds | ⚠DL = deadlock)',
    fontsize=13, fontweight='bold',
)

# Links: heatmap seed × mc_delay_per_train — deadlocked seeds zichtbaar met ⚠DL
sns.heatmap(
    pivot_trig, ax=axes[0],
    annot=annot_trig, fmt='',
    cmap='RdYlGn', vmin=0, vmax=100,
    cbar_kws={'label': 'MC trigger rate % (n_rescheduled / n_evaluated)'},
    linewidths=0.3,
)
axes[0].set_title(
    'MC trigger rate % per (seed × mc_delay_per_train)\\n'
    'Groen = vuurt vaak | Rood = zelden | ⚠DL = deadlock',
    fontsize=10,
)
axes[0].set_xlabel('mc_delay_per_train (s)')
axes[0].set_ylabel('Seed')

# Rechts: TED_combined + MC trigger rate — alleen schone seeds
ax_l = axes[1]
ax_r = ax_l.twinx()

ax_l.errorbar(
    summary_s5['mc_delay_per_train'], summary_s5['TED_mean'],
    yerr=summary_s5['TED_std'],
    fmt='o-', color='#4C72B0', lw=2.5, ms=8, capsize=5,
    label='TED_combined (μ ± σ, excl. DL)',
)
ax_l.set_xlabel('mc_delay_per_train (s)', fontsize=11)
ax_l.set_ylabel('TED_combined (s)', color='#4C72B0', fontsize=11)
ax_l.tick_params(axis='y', labelcolor='#4C72B0')
ax_l.set_title(
    'TED_combined & MC trigger rate vs mc_delay_per_train\\n'
    'Lage threshold → solver vuurt vaker | metrics excl. deadlocked seeds',
    fontsize=10,
)

ax_r.plot(
    summary_s5['mc_delay_per_train'], summary_s5['trig_mc_mean'],
    's--', color='#D62728', lw=1.8, ms=7, alpha=0.85,
    label='Mean MC trigger rate %',
)
ax_r.fill_between(
    summary_s5['mc_delay_per_train'],
    summary_s5['trig_mc_mean'] - summary_s5['trig_mc_std'],
    summary_s5['trig_mc_mean'] + summary_s5['trig_mc_std'],
    alpha=0.12, color='#D62728',
)
ax_r.set_ylabel('Mean MC trigger rate %', color='#D62728', fontsize=11)
ax_r.tick_params(axis='y', labelcolor='#D62728')
ax_r.set_ylim(0, 105)

# Deadlock-annotaties onder x-ticks lijnplot
_annotate_deadlocks(ax_l, summary_s5['mc_delay_per_train'],
                    summary_s5['mc_delay_per_train'], dl_s2)

h1, l1 = ax_l.get_legend_handles_labels()
h2, l2 = ax_r.get_legend_handles_labels()
ax_l.legend(h1 + h2, l1 + l2, fontsize=9, loc='lower left')
ax_l.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(CALIB_DIR / 's5_mc_delay_per_train.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\nNumeriek overzicht (excl. deadlocked seeds):')
print(summary_s5[['mc_delay_per_train','TED_mean','TED_std','trig_mc_mean','n_deadlocks']].to_string(index=False))
"""))

cells.append(new_code_cell("""\
# ── Beste waarde ─────────────────────────────────────────────────────────────
# Kies hoogste mc_delay_per_train waarbij TED_combined ≤ 102% van minimum
# (minder onnodige solver-aanroepen zonder significant kwaliteitsverlies)
ted_min    = summary_s5['TED_mean'].min()
candidates = summary_s5.loc[summary_s5['TED_mean'] <= ted_min * 1.02, 'mc_delay_per_train']
BEST_MC    = float(candidates.max())

print(f'Auto-geselecteerd (hoogste mc_delay_per_train met TED ≤ 102% min):')
print(f'  mc_delay_per_train = {BEST_MC}s')
row_best = summary_s5.loc[summary_s5['mc_delay_per_train'] == BEST_MC].iloc[0]
print(f'  TED_combined       = {row_best[\"TED_mean\"]:.0f}s')
print(f'  MC trigger rate    = {row_best[\"trig_mc_mean\"]:.1f}%')
print(f'  Gem. n_evaluated   = {row_best[\"n_eval_mean\"]:.0f}  |  n_rescheduled = {row_best[\"n_resc_mean\"]:.0f}')
print()
print('NB: BEST_MC wordt gebruikt bij hybrid/event-driven triggerconfiguraties.')
print('    Secties 2–4 gebruiken periodic trigger; BEST_MC geldt voor event-driven runs.')
print('Pas handmatig aan op basis van de figuur:')
print('  # BEST_MC = 120.0')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTIE 6
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Sectie 6 — Empirische `min_objective_threshold`

**Doel:** kies een drempelwaarde voor het MIP-objectief op basis van de natuurlijke verdeling.

**Aanpak:**
- Hergebruik MIP-objectiefwaarden uit sectie 2 (label `MOI0`, threshold=0 → altijd toegepast).
- Als sectie 2 niet beschikbaar is: run extra 25 seeds met alle beste instellingen.
- Toon histogram van de MIP-objectiefwaarden over alle reschedule-events.
- Kies threshold op basis van het 10e percentiel (het "knikpunt").

**MIP-objectief (gewogen totale projectievertraging):**
- Laag = situatie niet zo problematisch → rescheduling mogelijk onnodig
- Hoog = veel vertraging in het systeem → rescheduling zinvol

> **⚠ Noot over implementatie:**
> `min_objective_threshold` is hier een NIEUW concept (ruwe MIP-objectiefdrempel),
> te onderscheiden van `min_objective_improvement` (sectie 2, verbeteringsvergelijking vs FCFS).
> In de huidige codebase bestaat enkel `min_objective_improvement`.
> Aanbeveling: voeg een `min_raw_objective_threshold`-parameter toe aan de controller
> als deze analyse aantoont dat een absolute objectiefdrempel zinvol is.

> **Vereiste controller.py aanpassing:**
> `_solution_objectives: list[float]` wordt bijgehouden en teruggegeven via `summary()`.
> Zorg dat je de meest recente versie van `controller.py` gebruikt.
"""))

cells.append(new_code_cell("""\
# ── Objectives verzamelen ─────────────────────────────────────────────────────
# Sectie 2 gebruikt event-driven trigger (mc_delay_per_train sweep).
# Voor de empirische objectiefverdeling willen we periodic + threshold=0
# zodat elke trigger de MIP oplost en de objectiefwaarde logt.
# Hergebruik label 'MC20' (laagste threshold → meeste reschedules) als proxy,
# of run extra dedicated periodic runs hieronder.
s2_labels_all = [f'MC{mc}' for mc in [20, 60, 120, 180, 240]]
pooled = []
for lbl in s2_labels_all:
    pooled.extend(_objectives_log.get(lbl, []))

if len(pooled) > 0:
    all_objectives = np.array(pooled, dtype=float)
    print(f'Hergebruik sectie 2 (event-driven runs): {len(all_objectives)} MIP-objectiefwaarden.')
    print('NB: mix van mc_delay_per_train waarden — verdeling is representatief voor event-driven runs.')
else:
    print('Geen objectives gevonden uit sectie 2. Extra periodic runs uitvoeren...')
    print(f'Instellingen: RW={BEST_RW}s, CW={BEST_CW}s, SP={BEST_SP}s,')
    print(f'              RCW={BEST_RCW}s, trigger=periodic')
    print()

    lbl_s6 = 'S6_empirical'
    _objectives_log.setdefault(lbl_s6, [])

    for seed in range(N_SEEDS):
        patch_settings(
            RESCHEDULING_HORIZON    = BEST_RW,
            CONFLICT_WINDOW         = BEST_CW,
            SWITCH_PENALTY          = BEST_SP,
            RETRACK_CONFLICT_WINDOW = BEST_RCW,
        )
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                _, _, meta_s6, _, _ = run_simulation(
                    **BASE_CFG, min_objective_improvement=0.0, seed=seed
                )
        finally:
            restore_settings()

        objs = meta_s6['controller_summary'].get('solution_objectives', [])
        _objectives_log[lbl_s6].extend(objs)

        if (seed + 1) % 5 == 0:
            print(f'  seed {seed+1}/{N_SEEDS} klaar  ({len(objs)} objectives)')

    all_objectives = np.array(_objectives_log[lbl_s6], dtype=float)

if len(all_objectives) == 0:
    print()
    print('⚠  Geen objectives beschikbaar.')
    print('   Controleer of controller.py de _solution_objectives bijhoudt.')
    print('   (Zie aanpassing in controller/controller.py)')
else:
    np.save(CALIB_DIR / 's6_mip_objectives.npy', all_objectives)
    print(f'Totaal: {len(all_objectives)} MIP-objectiefwaarden')
    print(f'  min={all_objectives.min():.0f}s  mediaan={np.median(all_objectives):.0f}s'
          f'  mean={all_objectives.mean():.0f}s  max={all_objectives.max():.0f}s')
"""))

cells.append(new_code_cell("""\
# ── Visualisatie 6: histogram + percentiel-analyse ───────────────────────────
if len(all_objectives) == 0:
    print('Geen objectives beschikbaar — voer de cel hierboven eerst uit.')
else:
    pct_candidates = [5, 10, 15, 20, 25, 33]
    pct_values     = {p: float(np.percentile(all_objectives, p)) for p in pct_candidates}
    cmap_pct       = plt.cm.Reds(np.linspace(0.35, 0.90, len(pct_candidates)))

    p95 = np.percentile(all_objectives, 95)
    objs_clip = all_objectives[all_objectives <= p95 * 1.15]  # clip staart voor leesbaarheid

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Verdeling MIP-objectiefwaarden (gewogen totale projectievertraging)\\n'
        f'(n={len(all_objectives):,} rescheduling-events, {N_SEEDS} seeds)',
        fontsize=12, fontweight='bold',
    )

    # Links: lineair histogram
    ax = axes[0]
    ax.hist(objs_clip, bins=60, density=True, alpha=0.55, color='#4C72B0',
            edgecolor='white', linewidth=0.4, label='MIP objective (gekniptd bij p95)')
    for (p, v), col in zip(pct_values.items(), cmap_pct):
        ax.axvline(v, color=col, ls='--', lw=1.8, label=f'p{p} = {v:.0f}s')
    ax.set_xlabel('MIP objective (s gewogen vertraging)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Histogram MIP-objectiefwaarden (lineaire schaal)', fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # Rechts: log-schaal (staart zichtbaar)
    ax2 = axes[1]
    ax2.hist(all_objectives[all_objectives > 0], bins=80, density=True,
             alpha=0.55, color='#DD8452', edgecolor='white', linewidth=0.4,
             label='MIP objective (log-x)')
    ax2.set_xscale('log')
    for (p, v), col in zip(pct_values.items(), cmap_pct):
        ax2.axvline(v, color=col, ls='--', lw=1.8, label=f'p{p} = {v:.0f}s')
    ax2.set_xlabel('MIP objective (s) [log-schaal]', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Histogram MIP-objectiefwaarden (log-schaal)\\n← staart & laag-objectief events zichtbaar',
                  fontsize=10)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(CALIB_DIR / 's6_mip_objective_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Percentielentabel
    print('\\nPercentielentabel:')
    print(f'  {\"Percentiel\":>12}  {\"Threshold (s)\":>15}  {\"% events overgeslagen\":>22}')
    print('  ' + '-' * 52)
    for p, v in pct_values.items():
        skip_pct = 100 * (all_objectives <= v).mean()
        print(f'  p{p:>2}          {v:>12.0f}s   {skip_pct:>20.1f}%')

    # Aanbeveling op basis van p10
    knee = float(np.percentile(all_objectives, 10))
    print(f'\\n→ Aanbevolen threshold (p10): {knee:.0f}s')
    print(f'  Interpretatie: {10:.0f}% van de reschedule-events heeft een MIP-objectief')
    print(f'  onder {knee:.0f}s — de situatie is dan weinig problematisch.')
    print()
    print('NB: Dit is een RUWE OBJECTIEFDREMPEL (min_raw_objective_threshold),')
    print('    te onderscheiden van min_objective_improvement (sectie 2).')
    print('    min_objective_improvement vergelijkt MIP vs FCFS (relatief).')
    print('    min_raw_objective_threshold vergelijkt MIP vs 0 (absoluut).')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SAMENVATTING
# ─────────────────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
---
## Samenvatting aanbevolen parameters

Voer de onderstaande cel uit na alle secties om de kalibratie-resultaten samen te vatten.
"""))

cells.append(new_code_cell("""\
print('=' * 65)
print('KALIBRATIE-SAMENVATTING  —  Rescheduling systeem')
print('=' * 65)

params = {
    'RESCHEDULING_HORIZON':             BEST_RW,
    'CONFLICT_WINDOW':                  BEST_CW,
    'min_objective_improvement':        BEST_MOI,
    'SWITCH_PENALTY':                   BEST_SP,
    'RETRACK_CONFLICT_WINDOW': BEST_RCW,
}

for k, v in params.items():
    unit = 's' if isinstance(v, (int, float)) else ''
    print(f'  {k:<40} = {v}{unit}')

print()
print('Triggerconfig (vast):')
print('  trigger_strategy   = periodic')
print('  periodic_freq      = 900s')
print('  objective_strategy = static (no_priority)')
print('  weight_passenger   = 1')
print('  weight_freight     = 1')
print('  use_retracking     = True')
print()
print(f'Output bestanden: {CALIB_DIR}')
import os
for f in sorted(CALIB_DIR.glob('*.csv')):
    size = os.path.getsize(f)
    print(f'  {f.name:<45} ({size/1024:.1f} kB)')
for f in sorted(CALIB_DIR.glob('*.png')):
    size = os.path.getsize(f)
    print(f'  {f.name:<45} ({size/1024:.1f} kB)')
"""))

# ─────────────────────────────────────────────────────────────────────────────
# NOTEBOOK AANMAKEN
# ─────────────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3 (ipykernel)',
    'language': 'python',
    'name': 'python3',
}
nb.metadata['language_info'] = {
    'codemirror_mode': {'name': 'ipython', 'version': 3},
    'file_extension': '.py',
    'mimetype': 'text/x-python',
    'name': 'python',
    'nbformat': 4,
    'pygments_lexer': 'ipython3',
    'version': '3.11.0',
}

out_path = Path(__file__).parent / 'Calibrate_all.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'Notebook geschreven: {out_path}')
