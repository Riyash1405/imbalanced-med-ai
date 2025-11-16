# configuration values and global constants
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure directories exist at import
for p in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NUM_WORKERS = 4
DEVICE = "cuda"  # change to "cpu" if no GPU
