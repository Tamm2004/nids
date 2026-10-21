"""Score new flow records with the saved model.
Usage: python src/predict.py path/to/new_flows.csv"""
import sys
import joblib
import numpy as np
import pandas as pd
from config import MODEL_DIR


def main(path):
    bundle = joblib.load(MODEL_DIR / "nids_model.joblib")
    model, features = bundle["model"], bundle["features"]

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)

    proba = model.predict_proba(df[features])[:, 1]
    out = df.copy()
    out["attack_probability"] = proba.round(4)
    out["prediction"] = np.where(proba >= 0.5, "ATTACK", "BENIGN")
    print(out["prediction"].value_counts())
    out.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/predict.py <csv_file>")
    main(sys.argv[1])
