from pathlib import Path

# Pas aan naar jouw lokale situatie
# Dauw:
RAW_DATA_DIR = Path("/Users/ddw/Desktop/Rescheduling/data/raw")
BRONZE_DIR   = Path("/Users/ddw/Desktop/Rescheduling/data/bronze")
SILVER_DIR   = Path("/Users/ddw/Desktop/Rescheduling/data/silver")
GOLD_DIR     = Path("/Users/ddw/Desktop/Rescheduling/data/gold")

# Alice:
# RAW_DATA_DIR = Path("")
# BRONZE_DIR   = Path("")
# SILVER_DIR   = Path("")
# GOLD_DIR     = Path("")

# MIP parameters — gedeeld, staan wel gewoon hier
M          = 86400   # τ_max in seconden (24 uur)
GAMMA      = 300     # vertragingsdrempel γ in seconden (5 min)
EPSILON    = 1       # kleine constante ε
DELTA_MAX  = 86400   # maximale vertraging (vooraleer er cancellation plaatsvindt?)
#TAU_MAX   = 5400 # Planning horizon (s) 90 min, treinen en segmenten die verder liggen dan 90 min zitten niet in active trains TO DO:NOG TOEVOEGEN IN instance.py

# Solver settings
SOLVER_TIMEOUT_SECONDS = 60     # Max solve time per Gurobi call
SOLVER_MIP_GAP         = 0.01   # Relative MIP gap tolerance (1 %)
 
 
# Cancellation penalties
CANCELLATION_PENALTY = {
    "P": 200,
    "F": 100,
}
 
 
# Priority weights
PRIORITY_WEIGHT = {
    "P": 2,
    "F": 1,
}
 
 
# Monte Carlo constants — TO DO: calibrate from baseline simulations
THRESHOLD_MULTIPLIER = 1.5   # performance_threshold = THRESHOLD_MULTIPLIER * current delay
MC_DELAY_PROBABILITY = 0.3   # Probability a train picks up extra delay per MC step
MC_DELAY_MAX_SECONDS = 300   # Max extra delay per MC step (s) = 5 min
MC_ITERATIONS        = 5     # Number of MC roll-outs per evaluation
 
 
# Default trigger parameters
PERIODIC_FREQ        = 900   # Periodic: interval between solver calls (s)
EVENT_DRIVEN_FREQ    = 900   # Event-driven: min interval between solver calls (s)
CONTROLLER_FREQ      = 300   # How often controller evaluates triggers (s)
THRESHOLD_CONFIDENCE = 0.6   # Min P(metric > threshold) to fire solver