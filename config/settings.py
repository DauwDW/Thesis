from pathlib import Path

# =============================================================================
# Paden — pas aan naar jouw lokale situatie
# =============================================================================
# Dauw:
RESULTS_DIR       = Path("/Users/ddw/Desktop/Rescheduling/results")
DISTRIBUTIONS_DIR = Path("/Users/ddw/Desktop/Rescheduling/data/distributions")
RAW_DATA_DIR      = Path("/Users/ddw/Desktop/Rescheduling/data/raw")
BRONZE_DIR        = Path("/Users/ddw/Desktop/Rescheduling/data/bronze")
SILVER_DIR        = Path("/Users/ddw/Desktop/Rescheduling/data/silver")
GOLD_DIR          = Path("/Users/ddw/Desktop/Rescheduling/data/gold")

# # Alice:
# RESULTS_DIR       = Path("C:\\Users\\Allice\\Thesis\\results")
# DISTRIBUTIONS_DIR = Path("C:\\Users\\Allice\\Thesis\\data\\distributions")
# RAW_DATA_DIR      = Path("C:\\Users\\Allice\\Thesis\\data\\raw")
# BRONZE_DIR        = Path("C:\\Users\\Allice\\Thesis\\data\\bronze")
# SILVER_DIR        = Path("C:\\Users\\Allice\\Thesis\\data\\silver")
# GOLD_DIR          = Path("C:\\Users\\Allice\\Thesis\\data\\gold")

# =============================================================================
# Simulatie
# =============================================================================
SIMULATION_SEED = 42

# =============================================================================
# Freight sampling
# =============================================================================
FREIGHT_RUNNING_TIME_SCALE = 1.3  # normaal 1.3

# Treintypen die als referentie dienen voor freight rijtijden.
# Consistent gebruikt in data/freight.py (timetable-generatie) én
# reality/sampling.py (simulatie) — wijzig op één plek, geldt overal.
FREIGHT_POOL_TYPES = ("IC", "L", "S")

PASSING_DURATION_PASSENGER = 1  # seconden
PASSING_DURATION_FREIGHT   = 1  # seconden

# =============================================================================
# MIP model
# =============================================================================
SOLVER_DURATION_STATISTIC = "scheduled"  # "scheduled", "median", "mean", "p75"


GAMMA     = 120    # vertragingsdrempel γ (s)
EPSILON   = 1      # kleine constante ε
DELTA_MAX = 86400  # maximale vertraging (s)
L         = 86400  # big-M

SOLVER_TIMEOUT_SECONDS = 60
SOLVER_MIP_GAP         = 0.00000001

RESCHEDULING_HORIZON = 3600 # normaal 7200
CONFLICT_WINDOW = 1200 # normaal 1800

# Tijdvenster voor retracking-conflicten (z_alt/y_alt variabelen).
# Kleiner dan CONFLICT_WINDOW om model-explosie te voorkomen bij drukke periodes.
# Twee treinen worden als potentieel retracking-conflicterend beschouwd als hun
# verwachte exittijden op het station maximaal RETRACK_CONFLICT_WINDOW uit elkaar liggen.
RETRACK_CONFLICT_WINDOW = 900  # 15 minuten

# Stations waarvoor retracking actief is.
# Pilotrun (seed=42, periodic 1800s) gaf volgende verdeling:
#   BRUSSEL-ZUID        : 194 switches  (top 1)
#   BRUSSEL-CENTRAAL    : 174 switches  (top 2)
#   BRUSSEL-CONGRES     : 167 switches  (top 3)
#   BRUSSEL-KAPELLEKERK : 166 switches  (top 4)
#
# Alle 4 stations meenemen: MIP lost alle 47 reschedules optimaal op in 3.2s totaal.
# Alleen top-3 (zonder Kapellekerk): deadlock bij t≈71 000s doordat Kapellekerk
# zonder retracking congestie opbouwt die de solver niet kan oplossen.
# Conclusie: gebruik alle 4 stations (None = alles toelaten).
RETRACK_STATIONS: set[str] | None = None

# =============================================================================
# Retracking — platform-switch penalty
# =============================================================================
# Penalty (in seconden) per platform-switch in het MIP-objectief.
# De solver wijkt alleen af van het geplande platform als dat minstens
# SWITCH_PENALTY seconden totale vertraging bespaart.
# Hogere waarde → minder switches, stabieler maar minder flexibel.
# Lagere waarde → solver switcht vrij, risico op onnodige wisselingen.
SWITCH_PENALTY: float = 120.0

# =============================================================================
# Monte Carlo trigger
# =============================================================================
MC_DELAY_PER_TRAIN = 120
MC_ITERATIONS        = 5
THRESHOLD_CONFIDENCE = 0.6