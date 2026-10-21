# ML Network Intrusion Detection System (NIDS)

A beginner-friendly machine learning project that classifies network flows as
**BENIGN** or **ATTACK** using Python and scikit-learn, trained on the public
**CIC-IDS2017** dataset.

## Project structure
```
nids/
├── data/                  # put CIC-IDS2017 CSV files here
├── models/                # saved model (created by train.py)
├── reports/               # charts + metrics (created by train.py)
├── src/
│   ├── config.py          # paths and settings
│   ├── data_loader.py     # load + clean data, build binary labels
│   ├── train.py           # train, compare, evaluate, save
│   ├── predict.py         # score new CSV files
│   └── make_sample_data.py# synthetic data for a quick smoke test
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the dataset
1. Download **CIC-IDS2017** from the Canadian Institute for Cybersecurity
   (UNB) website. Use the **MachineLearningCSV** version (8 CSV files).
2. Copy the CSV files into `data/`.
3. Tip: the full set is ~2.8M rows. For a first run, set
   `MAX_ROWS_PER_FILE = 50000` in `src/config.py`.

No dataset yet? Test the pipeline with fake data: `python src/make_sample_data.py`
(results on it are meaningless - delete `data/sample_synthetic.csv` before using real data).

## Run
```bash
python src/train.py
python src/predict.py data/some_flows.csv   # writes predictions.csv
```

## What it does
1. **Load and clean**: strips column-name spaces, removes duplicates, `inf` and `NaN`.
2. **Binary labels**: `BENIGN` -> 0, everything else -> 1.
3. **Stratified 80/20 split** so train/test keep the same attack ratio.
4. **Trains** Logistic Regression (scaled baseline) and Random Forest.
5. **Evaluates** with precision, recall, F1, ROC-AUC, and a confusion matrix.
6. **Saves** the best model, charts and metrics.

## Why these metrics?
Accuracy alone is misleading because benign traffic dominates. **Recall** shows
how many attacks you catch; **precision** shows how many alerts are real.

## Limitations (mention these in your report)
- CIC-IDS2017 is lab-generated, 2017-era traffic; real networks differ.
- Random train/test split can be optimistic; a time-based split is stricter.
- Duplicate-ish flows and very high scores (>99%) are common on this dataset.
  Be honest about that.

## Next steps
- Multi-class: predict the attack type (use `Label` directly instead of binary).
- Tune hyperparameters with `GridSearchCV`/`RandomizedSearchCV`.
- Try Gradient Boosting / XGBoost; handle imbalance with SMOTE.
- Train on some days, test on others to check generalization.
- Wrap `predict.py` in a Flask/Streamlit dashboard.

## Dashboard (localhost)
```bash
pip install -r requirements.txt
python src/train.py          # once, creates the model and charts
streamlit run app.py         # opens http://localhost:8501
```
Pages: Overview, Model Results, Data Explorer, Try It Live.
