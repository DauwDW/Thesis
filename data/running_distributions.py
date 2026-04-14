# data/running_distributions.py
#
# Laadt en beheert empirische rijtijdverdelingen voor de simulatie.
#
# Structuur van running_distributions.json:
#   distributions[section][train_type][dynamics][period]
#     → { real: [float, ...], planned: float, n: int }

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DISTRIBUTIONS_PATH = Path(__file__).parent / 'distributions' / 'running_distributions.json'

_distributions: dict | None = None


def _load() -> dict:
    """Laadt running_distributions.json in de globale cache (lazy loading)."""
    global _distributions
    if _distributions is None:
        if not DISTRIBUTIONS_PATH.exists():
            raise FileNotFoundError(
                f"running_distributions.json niet gevonden op {DISTRIBUTIONS_PATH}. "
                "Run Running_Distributions.ipynb eerst."
            )
        with open(DISTRIBUTIONS_PATH) as f:
            _distributions = json.load(f)
        logger.info(f"Rijtijdverdelingen geladen: {len(_distributions)} secties")
    return _distributions


def sample_running_time(
    section:    str,
    train_type: str,
    dynamics:   str,
    period:     str,
    rng:        np.random.Generator | None = None,
) -> float | None:
    """
    Samplet een werkelijke rijtijd (in seconden) via np.random.choice.

    Geeft None terug als geen distributie gevonden — de aanroeper
    gebruikt dan de geplande rijtijd uit de gold timetable als fallback.

    Args:
        section:    sectiecode (bv. '25:BRUSSEL-NOORD-BRUSSEL-CENTRAAL')
        train_type: treintype ('IC', 'S', 'L', ...)
        dynamics:   rijdynamiek ('ACC-BR', 'ACC-0', '0-BR', '0-0')
        period:     dagperiode ('MORNING PEAK', 'DAYTIME', ...)
        rng:        numpy random generator (voor reproduceerbaarheid)

    Returns:
        Werkelijke rijtijd in seconden, of None
    """
    if rng is None:
        rng = np.random.default_rng()

    dist = _load()
    try:
        cell = dist[section][train_type][dynamics][period]
    except KeyError:
        logger.warning(
            f"Geen rijtijdverdeling voor {section} | {train_type} | {dynamics} | {period}"
        )
        return None

    return float(rng.choice(cell['real']))