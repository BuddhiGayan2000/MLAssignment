from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Artifacts
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Reproducibility
RANDOM_STATE = 42

# Train/val/test split (fractions)
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Target column (binary: extreme rain tomorrow)
TARGET_COL = "extreme_rain_tomorrow"

# Ensure dirs exist when imported
for d in (FIGURES_DIR, TABLES_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR):
    d.mkdir(parents=True, exist_ok=True)
