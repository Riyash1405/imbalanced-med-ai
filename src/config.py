from pathlib import Path
import torch

# Project root is assumed to be project folder where you run python -m src.main
DATA_DIR = Path("data")
CT_DIR = DATA_DIR / "ct_raw"
GENOMICS_DIR = DATA_DIR / "genomics"
TABULAR_DIR = DATA_DIR / "tabular"

# Output directories
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Other settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1  # for scikit-learn / joblib parallelism

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


RANDOM_STATE = 42
REPORT_ROOT = "reports"
DEVICE = "cpu"  # change to "cuda" if using GPU and torch installed with cuda
N_JOBS = 4