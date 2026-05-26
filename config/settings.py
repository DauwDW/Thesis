from pathlib import Path

# =============================================================================
# Paden — pas aan naar jouw lokale situatie
# =============================================================================
# Dauw:
# RESULTS_DIR       = Path("/Users/ddw/Desktop/Rescheduling/results")
# DISTRIBUTIONS_DIR = Path("/Users/ddw/Desktop/Rescheduling/data/distributions")
# RAW_DATA_DIR      = Path("/Users/ddw/Desktop/Rescheduling/data/raw")
# BRONZE_DIR        = Path("/Users/ddw/Desktop/Rescheduling/data/bronze")
# SILVER_DIR        = Path("/Users/ddw/Desktop/Rescheduling/data/silver")
# GOLD_DIR          = Path("/Users/ddw/Desktop/Rescheduling/data/gold")

# # Alice:
RESULTS_DIR       = Path("C:\\Users\\Allice\\Thesis\\results")
DISTRIBUTIONS_DIR = Path("C:\\Users\\Allice\\Thesis\\data\\distributions")
RAW_DATA_DIR      = Path("C:\\Users\\Allice\\Thesis\\data\\raw")
BRONZE_DIR        = Path("C:\\Users\\Allice\\Thesis\\data\\bronze")
SILVER_DIR        = Path("C:\\Users\\Allice\\Thesis\\data\\silver")
GOLD_DIR          = Path("C:\\Users\\Allice\\Thesis\\data\\gold")

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
CONFLICT_WINDOW = 1800 # normaal 1200

# =============================================================================
# Dispatcher prioriteitsveroudering
# =============================================================================
# Na zoveel seconden zonder nieuwe reschedule vervalt de MIP-prioriteit en
# valt de dispatcher terug op scheduled_entry (timetable-volgorde).
# Kies een waarde ≥ controller_freq zodat de prioriteit niet veroudert
# vóór de volgende geplande reschedule.
DISPATCHER_PRIORITY_TTL: float = 900.0

# =============================================================================
# Monte Carlo trigger
# =============================================================================
MC_DELAY_PER_TRAIN = 120
MC_ITERATIONS        = 5
THRESHOLD_CONFIDENCE = 0.6