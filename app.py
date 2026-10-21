"""Streamlit dashboard for the ML Network Intrusion Detector.
Run:  streamlit run app.py   ->  opens http://localhost:8501"""
import sys
import json
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_DIR, MODEL_DIR, REPORT_DIR, LABEL_COL, BENIGN_LABEL  # noqa: E402

st.set_page_config(page_title="ML Network Intrusion Detector",
                   page_icon="🛡️", layout="wide")


# ---------- helpers ----------
@st.cache_resource
def load_model():
    path = MODEL_DIR / "nids_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner="Reading CSV files...")
def load_data(rows_per_file: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    frames = []
    for f in files:
        # random-ish sample across the file instead of only the first rows
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        if len(df) > rows_per_file:
            df = df.sample(rows_per_file, random_state=42)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    df[LABEL_COL] = (df[LABEL_COL].astype(str).str.strip()
                     .str.replace(r"[^\x20-\x7E]", "-", regex=True))
    return df


def score(df: pd.DataFrame, bundle, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    feats = bundle["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} required columns, e.g. {missing[:5]}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
    proba = bundle["model"].predict_proba(df[feats])[:, 1]
    df["attack_probability"] = proba.round(4)
    df["prediction"] = np.where(proba >= threshold, "ATTACK", "BENIGN")
    return df


def need_model():
    st.error("No trained model found. Run `python src/train.py` first, then refresh.")
    st.stop()


bundle = load_model()

# ---------- sidebar ----------
st.sidebar.title("🛡️ NIDS Dashboard")
page = st.sidebar.radio("Go to", ["Overview", "Model Results",
                                  "Data Explorer", "Try It Live"])
st.sidebar.caption("Binary classifier: BENIGN vs ATTACK\nDataset: CIC-IDS2017")

# ---------- Overview ----------
if page == "Overview":
    st.title("🛡️ ML Network Intrusion Detector")
    st.write("A machine learning model that looks at network flow records "
             "and predicts whether traffic is **normal** or **potentially malicious**.")

    metrics_file = REPORT_DIR / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        st.subheader("Best model performance (test set)")
        c = st.columns(5)
        c[0].metric("Accuracy", f"{m['accuracy']:.2%}")
        c[1].metric("Precision", f"{m['precision']:.2%}")
        c[2].metric("Recall", f"{m['recall']:.2%}")
        c[3].metric("F1 score", f"{m['f1']:.2%}")
        c[4].metric("ROC-AUC", f"{m['roc_auc']:.4f}")
    else:
        st.info("Run `python src/train.py` to generate results.")

    st.subheader("How it works")
    st.markdown("""
1. **Load** CIC-IDS2017 flow records (78 numeric features per flow).
2. **Clean** duplicates, infinite and missing values.
3. **Label** `BENIGN` = 0, any attack = 1.
4. **Train** Logistic Regression and Random Forest, compare them.
5. **Predict** on new traffic and show attack probabilities.
""")
    st.subheader("What the metrics mean")
    st.markdown("""
- **Precision** – of the flows flagged as attacks, how many really were.
- **Recall** – of all real attacks, how many were caught.
- **F1** – balance of precision and recall.
- Accuracy alone can mislead because attacks are only a small share of the traffic.
""")
    st.warning("Scores above 99% are common on CIC-IDS2017 because the data is "
               "lab-generated. Real networks are harder.")

# ---------- Model Results ----------
elif page == "Model Results":
    st.title("📊 Model Results")
    comp = REPORT_DIR / "model_comparison.csv"
    if not comp.exists():
        need_model()
    df = pd.read_csv(comp, index_col=0)
    st.subheader("Model comparison")
    st.dataframe(df.style.format("{:.4f}").highlight_max(axis=0, color="#14532d"),
                 use_container_width=True)
    st.bar_chart(df[["precision", "recall", "f1"]])

    c1, c2 = st.columns(2)
    for col, name, title in [(c1, "confusion_matrix.png", "Confusion matrix"),
                             (c2, "roc_curve.png", "ROC curve")]:
        p = REPORT_DIR / name
        if p.exists():
            col.subheader(title)
            col.image(str(p), use_container_width=True)
    fi = REPORT_DIR / "feature_importance.png"
    if fi.exists():
        st.subheader("Most important features")
        st.image(str(fi), width=700)
        st.caption("Features the Random Forest relied on most to separate attacks from normal traffic.")

# ---------- Data Explorer ----------
elif page == "Data Explorer":
    st.title("🔎 Data Explorer")
    if not glob.glob(str(DATA_DIR / "*.csv")):
        st.error(f"No CSV files in {DATA_DIR}")
        st.stop()
    rows = st.slider("Rows to sample per file (lower = faster)",
                     5_000, 100_000, 20_000, step=5_000)
    data = load_data(rows)
    st.write(f"Loaded **{len(data):,}** flows, **{data.shape[1]-1}** features.")

    counts = data[LABEL_COL].value_counts()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Traffic labels")
        st.bar_chart(counts)
    with c2:
        st.subheader("Benign vs attack")
        binary = (data[LABEL_COL] != BENIGN_LABEL).map({False: "BENIGN", True: "ATTACK"})
        fig, ax = plt.subplots(figsize=(4, 4))
        binary.value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax,
                                   colors=["#2a9d8f", "#e63946"], ylabel="")
        st.pyplot(fig)

    st.subheader("Compare a feature: benign vs attack")
    num_cols = [c for c in data.select_dtypes(include=[np.number]).columns]
    default = num_cols.index("Flow Duration") if "Flow Duration" in num_cols else 0
    feat = st.selectbox("Feature", num_cols, index=default)
    use_log = st.checkbox("Log scale (helps with skewed values)", value=True)
    vals = data[[feat]].copy()
    vals["Type"] = binary
    if use_log:
        vals[feat] = np.log1p(vals[feat].clip(lower=0))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(data=vals, x=feat, hue="Type", bins=50, stat="density",
                 common_norm=False, palette={"BENIGN": "#2a9d8f", "ATTACK": "#e63946"}, ax=ax)
    ax.set_xlabel(f"log(1 + {feat})" if use_log else feat)
    st.pyplot(fig)

    with st.expander("Raw data preview"):
        st.dataframe(data.head(200))

# ---------- Try It Live ----------
else:
    st.title("🧪 Try It Live")
    if bundle is None:
        need_model()
    threshold = st.slider("Alert threshold (lower = catches more attacks, more false alarms)",
                          0.05, 0.95, 0.50, 0.05)

    source = st.radio("Input", ["Random sample from my data folder", "Upload a CSV"],
                      horizontal=True)
    df_in = None
    if source == "Upload a CSV":
        up = st.file_uploader("CIC-IDS2017 style CSV", type="csv")
        if up is not None:
            df_in = pd.read_csv(up, low_memory=False)
    else:
        n = st.slider("Number of flows", 100, 5000, 1000, step=100)
        if st.button("Draw sample and scan"):
            pool = load_data(20_000)
            if pool.empty:
                st.error("No data found in the data folder.")
            else:
                st.session_state["sample"] = pool.sample(min(n, len(pool)))
        df_in = st.session_state.get("sample")

    if df_in is not None:
        try:
            res = score(df_in, bundle, threshold)
        except ValueError as e:
            st.error(str(e)); st.stop()

        n_att = int((res["prediction"] == "ATTACK").sum())
        c = st.columns(3)
        c[0].metric("Flows scanned", f"{len(res):,}")
        c[1].metric("Flagged as ATTACK", f"{n_att:,}")
        c[2].metric("Attack share", f"{n_att/len(res):.1%}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Attack probability distribution")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.hist(res["attack_probability"], bins=30, color="#e63946")
            ax.axvline(threshold, color="black", ls="--", label="threshold")
            ax.set_xlabel("Predicted attack probability"); ax.set_ylabel("Flows")
            ax.legend()
            st.pyplot(fig)
        with c2:
            if LABEL_COL in res.columns:
                st.subheader("Check against true labels")
                truth = (res[LABEL_COL].astype(str).str.strip() != BENIGN_LABEL).astype(int)
                pred = (res["prediction"] == "ATTACK").astype(int)
                cm = confusion_matrix(truth, pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(4, 3.2))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                            xticklabels=["BENIGN", "ATTACK"], yticklabels=["BENIGN", "ATTACK"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                st.info("No `Label` column in this file, so no accuracy check is shown.")

        st.subheader("Most suspicious flows")
        show = ["attack_probability", "prediction"] + \
               ([LABEL_COL] if LABEL_COL in res.columns else []) + \
               [c for c in ["Destination Port", "Flow Duration", "Total Fwd Packets"]
                if c in res.columns]
        st.dataframe(res.sort_values("attack_probability", ascending=False)[show].head(50),
                     use_container_width=True)
        st.download_button("Download all predictions (CSV)",
                           res.to_csv(index=False).encode(), "predictions.csv", "text/csv")
