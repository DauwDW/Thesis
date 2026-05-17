from pathlib import Path

# =============================================================================
# Paden — pas aan naar jouw lokale situatie
# =============================================================================
# Dauw:
#RESULTS_DIR       = Path("/Users/ddw/Desktop/Rescheduling/results")
#DISTRIBUTIONS_DIR = Path("/Users/ddw/Desktop/Rescheduling/data/distributions")
#RAW_DATA_DIR      = Path("/Users/ddw/Desktop/Rescheduling/data/raw")
#BRONZE_DIR        = Path("/Users/ddw/Desktop/Rescheduling/data/bronze")
#SILVER_DIR        = Path("/Users/ddw/Desktop/Rescheduling/data/silver")
#GOLD_DIR          = Path("/Users/ddw/Desktop/Rescheduling/data/gold")

# Alice:
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
DISPATCHER_POLL_INTERVAL = 10.0

# =============================================================================
# Freight sampling
# =============================================================================
FREIGHT_RUNNING_TIME_SCALE = 1.3

PASSING_DURATION_PASSENGER = 30  # seconden
PASSING_DURATION_FREIGHT   = 45  # seconden

# =============================================================================
# MIP model
# =============================================================================
WEIGHT_PASSENGER = 2
WEIGHT_FREIGHT   = 1

PSL_PASSENGER = 1
PSL_FREIGHT   = 0

GAMMA     = 120    # vertragingsdrempel γ (s)
EPSILON   = 1      # kleine constante ε
DELTA_MAX = 86400  # maximale vertraging (s)
L         = 86400  # big-M

SOLVER_TIMEOUT_SECONDS = 60
SOLVER_MIP_GAP         = 0.01

RESCHEDULING_HORIZON = 1800
CONFLICT_WINDOW      = 600

# =============================================================================
# Monte Carlo trigger
# =============================================================================
MC_DELAY_PER_TRAIN = 180
MC_ITERATIONS        = 5
THRESHOLD_CONFIDENCE = 0.6