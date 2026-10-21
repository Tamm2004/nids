"""Central settings so you only change things in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # put the CIC-IDS2017 CSV files here
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optional: limit rows per CSV file for quick experiments (None = use all)
MAX_ROWS_PER_FILE = 50000
