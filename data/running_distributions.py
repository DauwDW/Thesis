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

from config.settings import DISTRIBUTIONS_DIR

logger = logging.getLogger(__name__)
DISTRIBUTIONS_PATH = DISTRIBUTIONS_DIR / 'running_distributions.json'

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