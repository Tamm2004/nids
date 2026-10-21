"""Train, evaluate and save a binary NIDS (BENIGN vs ATTACK)."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, TEST_SIZE
from data_loader import load_dataset


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    m = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),   # attacks caught
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["BENIGN", "ATTACK"]))
    return m, pred


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    print(f"Features: {X.shape[1]} | Attack ratio: {y.mean():.2%}")

    # stratify keeps the same attack ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    models = {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100, n_jobs=-1, class_weight="balanced",
                random_state=RANDOM_STATE)),
        ]),
    }

    results, fitted, preds = [], {}, {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        m, p = evaluate(name, model, X_test, y_test)
        results.append(m); fitted[name] = model; preds[name] = p

    summary = pd.DataFrame(results).set_index("model").round(4)
    print("\n", summary)
    summary.to_csv(REPORT_DIR / "model_comparison.csv")

    best_name = summary["f1"].idxmax()
    best = fitted[best_name]
    print(f"\nBest model by F1: {best_name}")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds[best_name])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    RocCurveDisplay.from_estimator(best, X_test, y_test)
    plt.title(f"ROC Curve - {best_name}")
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=150); plt.close()

    # Feature importance (tree models only)
    clf = best.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = (pd.Series(clf.feature_importances_, index=X.columns)
               .sort_values(ascending=False).head(15)[::-1])
        plt.figure(figsize=(7, 5))
        imp.plot(kind="barh", color="teal")
        plt.title("Top 15 Features"); plt.tight_layout()
        plt.savefig(REPORT_DIR / "feature_importance.png", dpi=150); plt.close()

    # Save model + feature list (needed for prediction later)
    joblib.dump({"model": best, "features": list(X.columns)},
                MODEL_DIR / "nids_model.joblib")
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(summary.loc[best_name].to_dict(), indent=2))
    print(f"\nSaved model to {MODEL_DIR/'nids_model.joblib'} and reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
