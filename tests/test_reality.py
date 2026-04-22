# tests/test_reality.py
#
# Rigoureuze tests voor reality/sampling.py en de onderliggende
# data/running_distributions.py.
#
# Testklassen:
#   TestGetSamplesPassenger  — _get_samples_passenger: fallback-hiërarchie
#   TestGetSamplesFreight    — _get_samples_freight: combinatie + fallback
#   TestSampleRunningTime    — sample_running_time: integratie + scaling
#   TestSampleRunningTimeWithRealData — tests op echte JSON (indien beschikbaar)
#
# Strategie:
#   - Alle tests gebruiken een gemockte _load() via monkeypatch
#   - Echte data-tests worden overgeslagen als JSON niet beschikbaar is
#   - Elke fallback-niveau wordt apart getest
#   - Statistische properties worden geverifieerd over vele samples

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from reality.sampling import (
    _get_samples_freight,
    _get_samples_passenger,
    sample_running_time,
)
from config.settings import FREIGHT_RUNNING_TIME_SCALE

# =============================================================================
# Constanten
# =============================================================================

SECTION   = "TEST:A-B"
IC        = "IC"
S         = "S"
L         = "L"
FREIGHT   = "freight"
DYN_ACC0  = "ACC-0"
DYN_0BR   = "0-BR"
DYN_ACCBR = "ACC-BR"
DYN_00    = "0-0"
DAYTIME   = "DAYTIME"
MORNING   = "MORNING PEAK"
EVENING   = "EVENING"
NIGHT     = "NIGHT"
EVE_PEAK  = "EVENING PEAK"

# =============================================================================
# Helper: bouw een mock distributie-dict
# =============================================================================

def make_data(
    section:    str = SECTION,
    train_type: str = IC,
    dynamics:   str = DYN_ACC0,
    period:     str = DAYTIME,
    samples:    list[float] | None = None,
) -> dict:
    """Bouwt een minimale distributie-dict voor tests."""
    if samples is None:
        samples = [80.0, 85.0, 90.0, 95.0, 100.0]
    return {
        section: {
            train_type: {
                dynamics: {
                    period: {"real": samples, "n": len(samples)}
                }
            }
        }
    }


def make_multi_data() -> dict:
    """
    Bouwt een rijke distributie-dict met meerdere treintypes, dynamics en periodes.
    Gebruikt voor fallback- en combinatietests.
    """
    return {
        SECTION: {
            IC: {
                DYN_ACC0: {
                    DAYTIME:  {"real": [80.0, 81.0, 82.0], "n": 3},
                    MORNING:  {"real": [83.0, 84.0],       "n": 2},
                    EVENING:  {"real": [85.0],              "n": 1},
                },
                DYN_0BR: {
                    DAYTIME:  {"real": [70.0, 71.0],        "n": 2},
                    MORNING:  {"real": [72.0],              "n": 1},
                },
            },
            S: {
                DYN_ACC0: {
                    DAYTIME:  {"real": [90.0, 91.0, 92.0], "n": 3},
                    MORNING:  {"real": [93.0],              "n": 1},
                },
                DYN_ACCBR: {
                    DAYTIME:  {"real": [95.0, 96.0],        "n": 2},
                },
            },
            L: {
                DYN_00: {
                    NIGHT:    {"real": [100.0, 101.0],      "n": 2},
                },
            },
        },
        "OTHER:X-Y": {
            IC: {
                DYN_ACC0: {
                    DAYTIME: {"real": [50.0], "n": 1},
                }
            }
        }
    }


# =============================================================================
# TestGetSamplesPassenger
# =============================================================================

class TestGetSamplesPassenger:

    # --- Niveau 1: exacte match ---

    def test_exacte_match_geeft_correcte_samples(self):
        data = make_data(samples=[80.0, 85.0, 90.0])
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, DAYTIME)
        assert result == [80.0, 85.0, 90.0]

    def test_exacte_match_geeft_originele_lijst_terug(self):
        """Exacte match geeft de originele lijst terug, geen kopie."""
        data    = make_data(samples=[80.0, 85.0])
        result  = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, DAYTIME)
        assert result is data[SECTION][IC][DYN_ACC0][DAYTIME]["real"]

    def test_exacte_match_enkele_waarde(self):
        data = make_data(samples=[80.0])
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, DAYTIME)
        assert result == [80.0]

    # --- Niveau 2: andere period, zelfde dynamics ---

    def test_fallback_niveau2_andere_period(self):
        """NIGHT niet beschikbaar → combineert alle periodes van ACC-0."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, NIGHT)
        assert result is not None
        # Moet DAYTIME + MORNING + EVENING van IC/ACC-0 bevatten
        assert set(result) == {80.0, 81.0, 82.0, 83.0, 84.0, 85.0}

    def test_fallback_niveau2_bevat_alle_periodes(self):
        """Fallback niveau 2 combineert alle periodes van de exacte dynamics."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, EVE_PEAK)
        assert result is not None
        assert len(result) == 6  # 3 + 2 + 1 samples

    # --- Niveau 3: andere dynamics, zelfde period ---

    def test_fallback_niveau3_andere_dynamics(self):
        """ACC-BR niet beschikbaar voor IC → combineert alle dynamics voor DAYTIME."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACCBR, DAYTIME)
        assert result is not None
        # Moet DAYTIME van IC/ACC-0 en IC/0-BR bevatten
        assert set(result) == {80.0, 81.0, 82.0, 70.0, 71.0}

    def test_fallback_niveau3_combineert_correct(self):
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_00, MORNING)
        assert result is not None
        # 0-0 bestaat niet voor IC → alle dynamics die MORNING hebben
        assert 83.0 in result  # IC/ACC-0/MORNING
        assert 72.0 in result  # IC/0-BR/MORNING

    # --- Niveau 4: alles gecombineerd ---

    def test_fallback_niveau4_alles_gecombineerd(self):
        """Geen enkele match → combineert alle dynamics en periodes."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_00, NIGHT)
        assert result is not None
        # Alle IC samples: ACC-0 (3+2+1) + 0-BR (2+1) = 9
        assert len(result) == 9

    def test_fallback_niveau4_bevat_alle_ic_samples(self):
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_00, NIGHT)
        expected = {80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 70.0, 71.0, 72.0}
        assert set(result) == expected

    # --- None gevallen ---

    def test_onbekende_section_geeft_none(self):
        data = make_data()
        assert _get_samples_passenger(data, "ONBEKEND:X-Y", IC, DYN_ACC0, DAYTIME) is None

    def test_onbekend_treintype_geeft_none(self):
        data = make_data()
        assert _get_samples_passenger(data, SECTION, "TGV", DYN_ACC0, DAYTIME) is None

    def test_lege_data_geeft_none(self):
        assert _get_samples_passenger({}, SECTION, IC, DYN_ACC0, DAYTIME) is None

    def test_section_zonder_treintype_data_geeft_none(self):
        data = {SECTION: {}}
        assert _get_samples_passenger(data, SECTION, IC, DYN_ACC0, DAYTIME) is None

    def test_treintype_zonder_dynamics_data_geeft_none(self):
        data = {SECTION: {IC: {}}}
        assert _get_samples_passenger(data, SECTION, IC, DYN_ACC0, DAYTIME) is None

    # --- Prioriteit van fallback-niveaus ---

    def test_niveau1_heeft_prioriteit_over_niveau2(self):
        """Exacte match wint altijd van fallback niveau 2."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, DAYTIME)
        # Exacte match: alleen DAYTIME samples
        assert result == [80.0, 81.0, 82.0]

    def test_niveau2_heeft_prioriteit_over_niveau3(self):
        """Niveau 2 (andere period, zelfde dynamics) wint van niveau 3."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACC0, NIGHT)
        # Niveau 2: alleen ACC-0 periodes, niet 0-BR
        assert 70.0 not in result
        assert 71.0 not in result

    def test_niveau3_heeft_prioriteit_over_niveau4(self):
        """Niveau 3 (andere dynamics, zelfde period) wint van niveau 4."""
        data   = make_multi_data()
        result = _get_samples_passenger(data, SECTION, IC, DYN_ACCBR, DAYTIME)
        # Niveau 3: alleen DAYTIME samples, niet MORNING/EVENING
        assert 83.0 not in result  # MORNING
        assert 85.0 not in result  # EVENING


# =============================================================================
# TestGetSamplesFreight
# =============================================================================

class TestGetSamplesFreight:

    # --- Niveau 1: exacte dynamics + period over alle treintypes ---

    def test_exacte_match_combineert_alle_treintypes(self):
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, DYN_ACC0, DAYTIME)
        assert result is not None
        # IC/ACC-0/DAYTIME + S/ACC-0/DAYTIME
        assert set(result) == {80.0, 81.0, 82.0, 90.0, 91.0, 92.0}

    def test_exacte_match_bevat_alle_treintypes(self):
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, DYN_ACC0, DAYTIME)
        assert len(result) == 6  # 3 van IC + 3 van S

    def test_enkele_treintype_match(self):
        """Slechts één treintype heeft de gevraagde dynamics+period."""
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, DYN_ACCBR, DAYTIME)
        # Enkel S/ACC-BR/DAYTIME
        assert set(result) == {95.0, 96.0}

    # --- Niveau 2: exacte dynamics, alle periodes ---

    def test_fallback_niveau2_combineert_alle_periodes(self):
        """NIGHT niet beschikbaar voor ACC-0 → alle periodes van ACC-0."""
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, DYN_ACC0, NIGHT)
        assert result is not None
        # IC/ACC-0: DAYTIME(3) + MORNING(2) + EVENING(1) = 6
        # S/ACC-0:  DAYTIME(3) + MORNING(1) = 4
        # Totaal: 10
        assert len(result) == 10

    def test_fallback_niveau2_bevat_alle_periodes_van_dynamics(self):
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, DYN_0BR, EVE_PEAK)
        assert result is not None
        # IC/0-BR heeft DAYTIME en MORNING, EVE_PEAK bestaat niet
        assert 70.0 in result
        assert 71.0 in result
        assert 72.0 in result

    # --- Niveau 3: alles gecombineerd ---

    def test_fallback_niveau3_alles_gecombineerd(self):
        """Onbekende dynamics → combineert alles."""
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, "ONBEKEND", NIGHT)
        assert result is not None
        assert len(result) > 0

    def test_fallback_niveau3_bevat_alle_samples(self):
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, "ONBEKEND", "ONBEKEND")
        # Alle samples van alle treintypes
        assert 80.0 in result   # IC
        assert 90.0 in result   # S
        assert 100.0 in result  # L

    # --- None gevallen ---

    def test_onbekende_section_geeft_none(self):
        data = make_multi_data()
        assert _get_samples_freight(data, "ONBEKEND:X-Y", DYN_ACC0, DAYTIME) is None

    def test_lege_data_geeft_none(self):
        assert _get_samples_freight({}, SECTION, DYN_ACC0, DAYTIME) is None

    def test_lege_section_data_geeft_none(self):
        data = {SECTION: {}}
        assert _get_samples_freight(data, SECTION, DYN_ACC0, DAYTIME) is None

    def test_section_zonder_dynamics_data_geeft_none(self):
        data = {SECTION: {IC: {}}}
        assert _get_samples_freight(data, SECTION, DYN_ACC0, DAYTIME) is None

    # --- Isolatie van andere secties ---

    def test_andere_sectie_niet_meegenomen(self):
        """Samples van OTHER:X-Y mogen niet in resultaat zitten."""
        data   = make_multi_data()
        result = _get_samples_freight(data, SECTION, DYN_ACC0, DAYTIME)
        assert 50.0 not in result  # OTHER:X-Y/IC/ACC-0/DAYTIME


# =============================================================================
# TestSampleRunningTime
# =============================================================================

class TestSampleRunningTime:

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _patch(self, data: dict):
        """Context manager die _load() mockt met de opgegeven data."""
        return patch("reality.sampling._load", return_value=data)

    # --- Basis ---

    def test_passenger_exacte_match_geeft_waarde_uit_samples(self):
        data = make_data(samples=[80.0, 85.0, 90.0])
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng())
        assert result in [80.0, 85.0, 90.0]

    def test_passenger_geeft_float_terug(self):
        data = make_data(samples=[80.0])
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng())
        assert isinstance(result, float)

    def test_passenger_enkele_sample_geeft_altijd_die_waarde(self):
        data = make_data(samples=[80.0])
        with self._patch(data):
            for seed in range(20):
                result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng(seed))
                assert result == 80.0

    # --- Freight scaling ---

    def test_freight_schaalt_met_factor(self):
        """Freight samples worden geschaald met FREIGHT_RUNNING_TIME_SCALE."""
        samples = [100.0]
        data    = make_data(samples=samples)
        with self._patch(data):
            result = sample_running_time(SECTION, FREIGHT, DYN_ACC0, DAYTIME, self._rng())
        assert result == pytest.approx(100.0 * FREIGHT_RUNNING_TIME_SCALE)

    def test_freight_schaalt_alle_samples(self):
        """Alle mogelijke freight samples zijn geschaald."""
        samples = [80.0, 90.0, 100.0]
        data    = make_data(samples=samples)
        scaled  = {s * FREIGHT_RUNNING_TIME_SCALE for s in samples}
        with self._patch(data):
            results = {
                sample_running_time(SECTION, FREIGHT, DYN_ACC0, DAYTIME, self._rng(seed))
                for seed in range(100)
            }
        assert results.issubset(scaled)

    def test_freight_scale_groter_dan_passenger(self):
        """Freight rijtijd moet gemiddeld groter zijn dan passenger."""
        samples = [80.0, 85.0, 90.0, 95.0, 100.0]
        data    = make_data(samples=samples)
        n       = 200
        rng     = self._rng()

        with self._patch(data):
            passenger_times = [
                sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, rng)
                for _ in range(n)
            ]
            freight_times = [
                sample_running_time(SECTION, FREIGHT, DYN_ACC0, DAYTIME, rng)
                for _ in range(n)
            ]

        assert np.mean(freight_times) == pytest.approx(
            np.mean(passenger_times) * FREIGHT_RUNNING_TIME_SCALE, rel=0.05
        )

    def test_passenger_scale_is_één(self):
        """Passenger samples worden niet geschaald."""
        samples = [100.0]
        data    = make_data(samples=samples)
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng())
        assert result == 100.0

    # --- None gevallen ---

    def test_onbekende_section_geeft_none(self):
        data = make_data()
        with self._patch(data):
            result = sample_running_time("ONBEKEND:X-Y", IC, DYN_ACC0, DAYTIME, self._rng())
        assert result is None

    def test_onbekend_treintype_geeft_none(self):
        data = make_data()
        with self._patch(data):
            result = sample_running_time(SECTION, "TGV", DYN_ACC0, DAYTIME, self._rng())
        assert result is None

    def test_lege_data_geeft_none(self):
        with self._patch({}):
            result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng())
        assert result is None

    def test_freight_onbekende_section_geeft_none(self):
        data = make_data()
        with self._patch(data):
            result = sample_running_time("ONBEKEND:X-Y", FREIGHT, DYN_ACC0, DAYTIME, self._rng())
        assert result is None

    # --- Reproduceerbaarheid ---

    def test_zelfde_seed_geeft_zelfde_resultaat(self):
        data = make_data(samples=[80.0, 85.0, 90.0, 95.0, 100.0])
        with self._patch(data):
            r1 = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng(42))
            r2 = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng(42))
        assert r1 == r2

    def test_verschillende_seeds_kunnen_verschillen(self):
        data    = make_data(samples=[80.0, 85.0, 90.0, 95.0, 100.0])
        results = set()
        with self._patch(data):
            for seed in range(50):
                r = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, self._rng(seed))
                results.add(r)
        assert len(results) > 1

    # --- Statistische eigenschappen ---

    def test_samples_liggen_binnen_distributie(self):
        """Gesamplede waarden liggen altijd binnen de originele distributie."""
        samples = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
        data    = make_data(samples=samples)
        rng     = self._rng()
        with self._patch(data):
            for _ in range(500):
                result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, rng)
                assert result in samples

    def test_alle_waarden_worden_gesampeld(self):
        """Over voldoende samples worden alle distributiewaarden getrokken."""
        samples = [80.0, 85.0, 90.0, 95.0, 100.0]
        data    = make_data(samples=samples)
        rng     = self._rng()
        results = set()
        with self._patch(data):
            for _ in range(500):
                r = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, rng)
                results.add(r)
        assert results == set(samples)

    def test_gemiddelde_benadert_distributiegemiddelde(self):
        """Steekproefgemiddelde benadert populatiegemiddelde bij groot n."""
        samples = list(range(60, 121))  # 60 t/m 120
        data    = make_data(samples=[float(s) for s in samples])
        rng     = self._rng()
        with self._patch(data):
            drawn = [
                sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, rng)
                for _ in range(2000)
            ]
        assert np.mean(drawn) == pytest.approx(np.mean(samples), rel=0.05)

    def test_freight_gemiddelde_benadert_geschaald_gemiddelde(self):
        samples     = [float(s) for s in range(60, 121)]
        data        = make_data(samples=samples)
        rng         = self._rng()
        expected    = np.mean(samples) * FREIGHT_RUNNING_TIME_SCALE
        with self._patch(data):
            drawn = [
                sample_running_time(SECTION, FREIGHT, DYN_ACC0, DAYTIME, rng)
                for _ in range(2000)
            ]
        assert np.mean(drawn) == pytest.approx(expected, rel=0.05)

    # --- Fallback via sample_running_time ---

    def test_fallback_niveau2_via_sample_running_time(self):
        """sample_running_time gebruikt fallback bij ontbrekende period."""
        data = make_multi_data()
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_ACC0, NIGHT, self._rng())
        assert result is not None
        assert result in {80.0, 81.0, 82.0, 83.0, 84.0, 85.0}

    def test_fallback_niveau3_via_sample_running_time(self):
        """sample_running_time gebruikt fallback bij ontbrekende dynamics."""
        data = make_multi_data()
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_ACCBR, DAYTIME, self._rng())
        assert result is not None
        assert result in {80.0, 81.0, 82.0, 70.0, 71.0}

    def test_fallback_niveau4_via_sample_running_time(self):
        """sample_running_time gebruikt fallback niveau 4."""
        data = make_multi_data()
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_00, NIGHT, self._rng())
        assert result is not None

    def test_freight_fallback_via_sample_running_time(self):
        """Freight gebruikt freight-fallback bij ontbrekende period."""
        data = make_multi_data()
        with self._patch(data):
            result = sample_running_time(SECTION, FREIGHT, DYN_ACC0, NIGHT, self._rng())
        assert result is not None
        # Geschaald resultaat
        raw_samples = {80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 90.0, 91.0, 92.0, 93.0}
        expected    = {s * FREIGHT_RUNNING_TIME_SCALE for s in raw_samples}
        assert result in expected

    # --- Positieve waarden ---

    def test_resultaat_altijd_positief(self):
        """Rijtijden zijn altijd positief."""
        samples = [1.0, 50.0, 100.0, 300.0]
        data    = make_data(samples=samples)
        rng     = self._rng()
        with self._patch(data):
            for _ in range(100):
                result = sample_running_time(SECTION, IC, DYN_ACC0, DAYTIME, rng)
                assert result > 0

    def test_freight_resultaat_altijd_positief(self):
        samples = [1.0, 50.0, 100.0]
        data    = make_data(samples=samples)
        rng     = self._rng()
        with self._patch(data):
            for _ in range(100):
                result = sample_running_time(SECTION, FREIGHT, DYN_ACC0, DAYTIME, rng)
                assert result > 0

    # --- Alle train_types ---

    @pytest.mark.parametrize("train_type", ["IC", "S", "L", "EURST", "ICE", "INT"])
    def test_alle_passenger_types_werken(self, train_type):
        data = make_data(train_type=train_type, samples=[80.0, 85.0])
        with self._patch(data):
            result = sample_running_time(SECTION, train_type, DYN_ACC0, DAYTIME, self._rng())
        assert result in [80.0, 85.0]

    @pytest.mark.parametrize("dynamics", ["ACC-0", "0-BR", "ACC-BR", "0-0"])
    def test_alle_dynamics_werken(self, dynamics):
        data = make_data(dynamics=dynamics, samples=[80.0, 85.0])
        with self._patch(data):
            result = sample_running_time(SECTION, IC, dynamics, DAYTIME, self._rng())
        assert result in [80.0, 85.0]

    @pytest.mark.parametrize("period", ["DAYTIME", "MORNING PEAK", "EVENING PEAK", "EVENING", "NIGHT"])
    def test_alle_periodes_werken(self, period):
        data = make_data(period=period, samples=[80.0, 85.0])
        with self._patch(data):
            result = sample_running_time(SECTION, IC, DYN_ACC0, period, self._rng())
        assert result in [80.0, 85.0]


# =============================================================================
# TestSampleRunningTimeWithRealData
# =============================================================================

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "distributions" / "running_distributions.json"

@pytest.mark.skipif(
    not REAL_DATA_PATH.exists(),
    reason="running_distributions.json niet beschikbaar"
)
class TestSampleRunningTimeWithRealData:
    """
    Tests op de echte JSON — worden overgeslagen als het bestand niet bestaat.
    Verifieert dat de sampling correct werkt met echte data.
    """

    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _load_real(self) -> dict:
        with open(REAL_DATA_PATH) as f:
            return json.load(f)

    def test_json_heeft_verwachte_structuur(self):
        data = self._load_real()
        assert isinstance(data, dict)
        assert len(data) > 0
        # Check eerste entry
        section = next(iter(data))
        assert isinstance(data[section], dict)

    def test_elke_sectie_heeft_treintypes(self):
        data = self._load_real()
        for section, section_data in data.items():
            assert isinstance(section_data, dict), f"Sectie {section} is geen dict"
            assert len(section_data) > 0, f"Sectie {section} is leeg"

    def test_elke_cel_heeft_real_en_n(self):
        data = self._load_real()
        for section, section_data in data.items():
            for train_type, type_data in section_data.items():
                for dynamics, dyn_data in type_data.items():
                    for period, cell in dyn_data.items():
                        assert "real" in cell, (
                            f"Geen 'real' in {section}/{train_type}/{dynamics}/{period}"
                        )
                        assert "n" in cell, (
                            f"Geen 'n' in {section}/{train_type}/{dynamics}/{period}"
                        )
                        assert len(cell["real"]) > 0, (
                            f"Lege 'real' in {section}/{train_type}/{dynamics}/{period}"
                        )

    def test_n_consistent_met_real(self):
        """n moet overeenkomen met len(real)."""
        data = self._load_real()
        for section, section_data in data.items():
            for train_type, type_data in section_data.items():
                for dynamics, dyn_data in type_data.items():
                    for period, cell in dyn_data.items():
                        assert cell["n"] == len(cell["real"]), (
                            f"n={cell['n']} ≠ len(real)={len(cell['real'])} "
                            f"in {section}/{train_type}/{dynamics}/{period}"
                        )

    def test_alle_rijtijden_positief(self):
        """Alle rijtijden in de data zijn positief."""
        data = self._load_real()
        for section, section_data in data.items():
            for train_type, type_data in section_data.items():
                for dynamics, dyn_data in type_data.items():
                    for period, cell in dyn_data.items():
                        for t in cell["real"]:
                            assert t > 0, (
                                f"Negatieve rijtijd {t} in "
                                f"{section}/{train_type}/{dynamics}/{period}"
                            )

    def test_sample_op_eerste_sectie(self):
        """Sample op de eerste beschikbare sectie werkt."""
        data    = self._load_real()
        section = next(iter(data))
        type_   = next(iter(data[section]))
        dyn     = next(iter(data[section][type_]))
        period  = next(iter(data[section][type_][dyn]))

        result = sample_running_time(section, type_, dyn, period, self._rng())
        assert result is not None
        assert result > 0

    def test_sample_freight_op_echte_data(self):
        """Freight sampling werkt op echte data."""
        data    = self._load_real()
        section = next(iter(data))
        type_   = next(iter(data[section]))
        dyn     = next(iter(data[section][type_]))
        period  = next(iter(data[section][type_][dyn]))

        result = sample_running_time(section, FREIGHT, dyn, period, self._rng())
        assert result is not None
        assert result > 0

    def test_freight_groter_dan_passenger_op_echte_data(self):
        """Freight rijtijden zijn gemiddeld groter dan passenger op echte data."""
        data    = self._load_real()
        section = next(iter(data))
        type_   = next(iter(data[section]))
        dyn     = next(iter(data[section][type_]))
        period  = next(iter(data[section][type_][dyn]))

        rng = self._rng()
        n   = 500
        passenger = [sample_running_time(section, type_, dyn, period, rng) for _ in range(n)]
        freight   = [sample_running_time(section, FREIGHT, dyn, period, rng) for _ in range(n)]

        passenger = [x for x in passenger if x is not None]
        freight   = [x for x in freight   if x is not None]

        if passenger and freight:
            assert np.mean(freight) > np.mean(passenger)

    def test_onbekende_section_geeft_none_op_echte_data(self):
        result = sample_running_time("ONBEKEND:X-Y", IC, DYN_ACC0, DAYTIME, self._rng())
        assert result is None

    def test_sample_is_reproduceerbaar_op_echte_data(self):
        data    = self._load_real()
        section = next(iter(data))
        type_   = next(iter(data[section]))
        dyn     = next(iter(data[section][type_]))
        period  = next(iter(data[section][type_][dyn]))

        r1 = sample_running_time(section, type_, dyn, period, self._rng(42))
        r2 = sample_running_time(section, type_, dyn, period, self._rng(42))
        assert r1 == r2