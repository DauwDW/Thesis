# reality/sampling.py
#
# Sampling van werkelijke rijtijden voor lijnsegmenten (BETWEEN-STATION).
#
# Stationssegmenten worden niet via deze module bediend:
#   - WITHIN-STATION-DWELL:   geplande dwell uit timetable (sample_duration in simulator)
#   - WITHIN-STATION-PASSING: 1s constante (via is_passing=True)
#
# Verantwoordelijkheden:
#   - Passenger lijnsegmenten: empirische sample per (section, train_type, dynamics, period)
#   - Freight lijnsegmenten:   passenger-pool (IC + L) × FREIGHT_RUNNING_TIME_SCALE
#
# Fallback-hiërarchie:
#   1. (section, train_type, dynamics, period)   — exacte match
#   2. (section, train_type, dynamics, *)        — alle periodes gepoold
#   3. (section, train_type, *, *)               — alle dynamics + periodes gepoold
#   4. None                                       — caller gebruikt geplande rijtijd

from __future__ import annotations

import logging

import numpy as np

from config.settings import FREIGHT_RUNNING_TIME_SCALE, FREIGHT_POOL_TYPES
from data.running_distributions import _load

logger = logging.getLogger(__name__)

PASSING_DURATION = 1.0


def _pool_samples(
    data:        dict,
    section:     str,
    train_types: tuple[str, ...],
    dynamics:    str,
    period:      str,
) -> list[float] | None:
    """
    Verzamelt samples voor de gegeven train_types op (section, dynamics, period),
    met fallback over period → dynamics. Section en train_types blijven vereist.

    Returns None als er voor géén van de train_types data is op deze sectie.
    """
    section_data = data.get(section)
    if section_data is None:
        return None

    # Filter train_types tot wat effectief op deze sectie bestaat
    type_buckets = [section_data[t] for t in train_types if t in section_data]
    if not type_buckets:
        return None

    # 1. Exacte match: dynamics + period
    samples: list[float] = []
    for type_data in type_buckets:
        period_data = type_data.get(dynamics, {}).get(period)
        if period_data is not None:
            samples.extend(period_data["real"])
    if samples:
        return samples

    # 2. Drop period: zelfde dynamics, alle periodes
    for type_data in type_buckets:
        dyn_data = type_data.get(dynamics)
        if dyn_data is not None:
            for p_data in dyn_data.values():
                samples.extend(p_data["real"])
    if samples:
        # logger.debug(
        #     f"Fallback (drop period): ({section}, {train_types}, {dynamics}, {period})"
        # )
        return samples

    # 3. Drop dynamics: alle dynamics en periodes
    for type_data in type_buckets:
        for dyn_data in type_data.values():
            for p_data in dyn_data.values():
                samples.extend(p_data["real"])
    if samples:
    #     logger.debug(
    #         f"Fallback (drop dynamics): ({section}, {train_types}, {dynamics}, {period})"
    #     )
        return samples

    return None


def sample_running_time(
    section:    str,
    train_type: str,
    dynamics:   str,
    period:     str,
    rng:        np.random.Generator,
    is_passing: bool = False,
) -> float | None:
    """
    Samplet een werkelijke rijtijd (seconden) voor de simulatie.

    Returns
    -------
    float | None
        - 1.0 voor WITHIN-STATION-PASSING segmenten (is_passing=True)
        - Gesamplede rijtijd uit empirische verdeling (geschaald voor freight)
        - None als geen (section, train_type) data beschikbaar
          — caller valt terug op geplande rijtijd uit de timetable
    """
    if is_passing:
        return PASSING_DURATION

    data = _load()
    if not data:
        return None

    if train_type == "freight":
        pool_types = FREIGHT_POOL_TYPES
        scale      = FREIGHT_RUNNING_TIME_SCALE
    else:
        pool_types = (train_type,)
        scale      = 1.0

    samples = _pool_samples(data, section, pool_types, dynamics, period)

    if samples is None:
        # logger.debug(
        #     f"Geen data voor ({section}, {train_type}, {dynamics}, {period}) "
        #     f"— caller gebruikt geplande rijtijd"
        # )
        return None

    return float(rng.choice(samples)) * scale

#!!! deterministic times: 
def seconds_to_period(seconds: float) -> str:
    hour = (seconds % 86400) / 3600
    if hour < 6:    return "NIGHT"
    if hour < 9:    return "MORNING PEAK"
    if hour < 16:   return "DAYTIME"
    if hour < 19:   return "EVENING PEAK"
    return "EVENING"


def running_time_statistic(
    section:    str,
    train_type: str,
    dynamics:   str,
    period:     str,
    statistic:  str = "median",
) -> float | None:
    """
    Berekent een aggregaatstatistiek (mediaan/mean/P75) van de empirische
    rijtijdverdeling. Gebruikt dezelfde fallback-hiërarchie als sample_running_time.

    Returns None als geen data beschikbaar — caller valt terug op geplande rijtijd.
    """
    data = _load()
    if not data:
        return None

    if train_type == "freight":
        pool_types = FREIGHT_POOL_TYPES
        scale      = FREIGHT_RUNNING_TIME_SCALE
    else:
        pool_types = (train_type,)
        scale      = 1.0

    samples = _pool_samples(data, section, pool_types, dynamics, period)
    if samples is None:
        return None

    if statistic == "median":
        value = float(np.median(samples))
    elif statistic == "mean":
        value = float(np.mean(samples))
    elif statistic == "p25":
        value = float(np.percentile(samples, 25))
    elif statistic == "p40":
        value = float(np.percentile(samples, 40))
    # sampling.py — running_time_statistic()
    elif statistic == "p60":
        value = float(np.percentile(samples, 60))
    elif statistic == "p65":
        value = float(np.percentile(samples, 65))
    elif statistic == "p70":
        value = float(np.percentile(samples, 70))
    elif statistic == "p75":
        value = float(np.percentile(samples, 75))
    elif statistic == "p80":
        value = float(np.percentile(samples, 80))

    else:
        raise ValueError(
            f"Onbekende statistic: '{statistic}'. "
            f"Kies uit: 'median', 'mean', 'p75'."
        )

    return value * scale

