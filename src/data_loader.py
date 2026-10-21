"""Load and clean CIC-IDS2017 (or any CSV with a 'Label' column)."""
import glob
import numpy as np
import pandas as pd
from config import DATA_DIR, LABEL_COL, BENIGN_LABEL, MAX_ROWS_PER_FILE


def load_raw(data_dir=DATA_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Download CIC-IDS2017 "
            "(MachineLearningCSV) or run: python src/make_sample_data.py"
        )
    frames = []
    for f in files:
        print(f"Loading {f}")
        frames.append(pd.read_csv(f, nrows=MAX_ROWS_PER_FILE, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    # CIC-IDS2017 column names contain leading/trailing spaces
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, inf/NaN values; tidy labels."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    # Some labels contain odd characters (e.g. 'Web Attack \x96 Brute Force')
    df[LABEL_COL] = df[LABEL_COL].str.replace(r"[^\x20-\x7E]", "-", regex=True)
    print(f"Cleaned: {before:,} -> {len(df):,} rows")
    return df


def make_binary(df: pd.DataFrame):
    """Return X (features) and y (0 = benign, 1 = attack)."""
    y = (df[LABEL_COL] != BENIGN_LABEL).astype(int)
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    return X, y


def load_dataset():
    df = clean(load_raw())
    print("\nOriginal label counts:\n", df[LABEL_COL].value_counts(), "\n")
    return make_binary(df)
