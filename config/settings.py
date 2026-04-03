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
DELTA_MAX  = 86400   # maximale vertraging
