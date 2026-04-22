# reality/sampling.py
#
# Sampling van werkelijke rijtijden voor de simulatie.
#
# Deze module zit tussen de simulator en de ruwe data-laag in:
#
#   simulator.py
#       ↓  roept aan
#   reality/sampling.py     ← fallback-logica, freight-scaling, interface voor simulator
#       ↓  roept aan
#   data/running_distributions.py  ← laadt JSON, geeft ruwe samples terug
#
# Fallback-hiërarchie bij ontbrekende data (passenger):
#   1. Exacte match:           (section, train_type, dynamics, period)
#   2. Andere period:          (section, train_type, dynamics, *)
#   3. Andere dynamics:        (section, train_type, *, period)
#   4. Andere dynamics+period: (section, train_type, *, *)
#   5. None → simulator gebruikt geplande rijtijd uit timetable
#
# Freight treinen:
#   Geen eigen verdelingen — alle treintypes op het segment worden gecombineerd
#   en geschaald met FREIGHT_RUNNING_TIME_SCALE uit config/settings.py.
#
# Gebruik:
#   from reality.sampling import sample_running_time
#   duration = sample_running_time("36N:SCHAARBEEK-BRUSSEL-NOORD", "IC", "ACC-0", "DAYTIME", rng)

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config.settings import FREIGHT_RUNNING_TIME_SCALE
from data.running_distributions import _load

logger = logging.getLogger(__name__)

_FREIGHT_TYPE = "freight"


def _get_samples_freight(
    data:     dict,
    section:  str,
    dynamics: str,
    period:   str,
) -> list[float] | None:
    """
    Combineert rijtijden van alle treintypes op een segment voor freight.

    Fallback-hiërarchie:
      1. Alle treintypes, exacte dynamics + period
      2. Alle treintypes, exacte dynamics, alle periodes
      3. Alle treintypes, alle dynamics + alle periodes
    """
    section_data = data.get(section)
    if section_data is None:
        return None

    # 1. Exacte dynamics + period over alle treintypes
    all_samples = []
    for type_data in section_data.values():
        dyn_data = type_data.get(dynamics)
        if dyn_data is not None:
            period_data = dyn_data.get(period)
            if period_data is not None:
                all_samples.extend(period_data["real"])
    if all_samples:
        return all_samples

    # 2. Exacte dynamics, alle periodes, alle treintypes
    for type_data in section_data.values():
        dyn_data = type_data.get(dynamics)
        if dyn_data is not None:
            for p_data in dyn_data.values():
                all_samples.extend(p_data["real"])
    if all_samples:
        logger.debug(
            f"Freight fallback niveau 2: ({section}, {dynamics}, {period}) "
            f"→ alle periodes gecombineerd"
        )
        return all_samples

    # 3. Alles gecombineerd
    for type_data in section_data.values():
        for dyn_data in type_data.values():
            for p_data in dyn_data.values():
                all_samples.extend(p_data["real"])
    if all_samples:
        logger.debug(
            f"Freight fallback niveau 3: ({section}, {dynamics}, {period}) "
            f"→ alle dynamics en periodes gecombineerd"
        )
        return all_samples

    return None


def _get_samples_passenger(
    data:       dict,
    section:    str,
    train_type: str,
    dynamics:   str,
    period:     str,
) -> list[float] | None:
    """
    Zoekt de beste beschikbare lijst van rijtijden op via de fallback-hiërarchie
    voor passagierstreinen.
    """
    section_data = data.get(section)
    if section_data is None:
        return None

    type_data = section_data.get(train_type)
    if type_data is None:
        return None

    # 1. Exacte match
    dyn_data = type_data.get(dynamics)
    if dyn_data is not None:
        period_data = dyn_data.get(period)
        if period_data is not None:
            return period_data["real"]

    # 2. Andere period — zelfde dynamics
    if dyn_data is not None:
        all_samples = []
        for p_data in dyn_data.values():
            all_samples.extend(p_data["real"])
        if all_samples:
            logger.debug(
                f"Fallback niveau 2: ({section}, {train_type}, {dynamics}, {period}) "
                f"→ alle periodes gecombineerd ({len(all_samples)} samples)"
            )
            return all_samples

    # 3. Andere dynamics — zelfde period
    all_samples = []
    for dyn, d_data in type_data.items():
        period_data = d_data.get(period)
        if period_data is not None:
            all_samples.extend(period_data["real"])
    if all_samples:
        logger.debug(
            f"Fallback niveau 3: ({section}, {train_type}, {dynamics}, {period}) "
            f"→ alle dynamics voor period={period} ({len(all_samples)} samples)"
        )
        return all_samples

    # 4. Alle dynamics + alle periodes
    all_samples = []
    for d_data in type_data.values():
        for p_data in d_data.values():
            all_samples.extend(p_data["real"])
    if all_samples:
        logger.debug(
            f"Fallback niveau 4: ({section}, {train_type}, {dynamics}, {period}) "
            f"→ alle dynamics en periodes gecombineerd ({len(all_samples)} samples)"
        )
        return all_samples

    return None


def sample_running_time(
    section:    str,
    train_type: str,
    dynamics:   str,
    period:     str,
    rng:        np.random.Generator,
) -> Optional[float]:
    """
    Samplet een werkelijke rijtijd uit de empirische verdeling.

    Voor freight treinen worden alle treintypes op het segment gecombineerd
    en geschaald met FREIGHT_RUNNING_TIME_SCALE (config/settings.py).
    Voor passagierstreinen wordt de fallback-hiërarchie toegepast.

    Parameters
    ----------
    section    : str  — segment-id, consistent met SECTION in de gold timetable
    train_type : str  — treinsoort via TrainSubtype.value
                        ("IC", "S", "L", "EURST", "ICE", "INT", "freight")
    dynamics   : str  — rijdynamiek via Train.dynamics_at()
                        ("ACC-0", "0-BR", "ACC-BR", "0-0")
    period     : str  — dagperiode via _seconds_to_period() in simulator.py
                        ("DAYTIME", "MORNING PEAK", "EVENING PEAK", "EVENING", "NIGHT")
    rng        : np.random.Generator — random generator voor reproduceerbaarheid

    Returns
    -------
    float | None
        Gesamplede rijtijd in seconden (al geschaald voor freight).
        None als geen data beschikbaar — simulator valt terug op geplande rijtijd.
    """
    data = _load()
    if not data:
        return None

    is_freight = (train_type == _FREIGHT_TYPE)

    if is_freight:
        samples = _get_samples_freight(data, section, dynamics, period)
        scale   = FREIGHT_RUNNING_TIME_SCALE
    else:
        samples = _get_samples_passenger(data, section, train_type, dynamics, period)
        scale   = 1.0

    if samples is None:
        logger.debug(
            f"Geen data voor ({section}, {train_type}, {dynamics}, {period}) "
            f"— simulator gebruikt geplande rijtijd"
        )
        return None

    idx = rng.integers(0, len(samples))
    return float(samples[idx]) * scale