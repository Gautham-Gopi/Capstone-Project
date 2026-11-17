"""Train the fake-job-posting classifiers and save an inference bundle.

Usage:  python train.py [path/to/fake_job_postings.csv]

Trains one calibrated logistic-regression model on the posting text:
  word TF-IDF of the cleaned text + character n-gram TF-IDF of the raw text
  (title + company + description + requirements + benefits).

The split is group-aware: postings with identical cleaned text always land on the same side,
and duplicates are dropped from the training set, so the test metrics are not inflated by
postings the model has effectively already seen.

Outputs (next to this file):
  model.joblib  - {"model", "threshold", "term_weights"}  (see scoring.py)
  metrics.json  - hold-out metrics at the tuned threshold
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from text_utils import clean_text

HERE = Path(__file__).parent
SEED = 42
MAX_FEATURES = 20000
CHAR_FEATURES = 60000
C = 30
BETA = 2.0  # >1 favours recall (catching fakes) over precision when choosing the threshold


def load(csv_path):
    df = pd.read_csv(csv_path).fillna("")
    raw = (df["title"] + " " + df["company_profile"] + " " + df["description"] + " "
           + df["requirements"] + " " + df["benefits"]).tolist()
    with Pool(4) as pool:
        cleaned = pool.map(clean_text, raw, chunksize=200)
    X = pd.DataFrame({"text": cleaned, "raw": raw})
    return X, df["fraudulent"].astype(int)


def make_pipeline():
    transformers = [
        # word n-grams on the cleaned text (first, so term_weights indices stay aligned) ...
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2),
                                  min_df=3, sublinear_tf=True), "text"),
        # ... plus character n-grams on the raw text (catch odd spellings, emails, symbols)
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  max_features=CHAR_FEATURES, min_df=3, sublinear_tf=True), "raw")]
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    return Pipeline([("ct", ColumnTransformer(transformers)),
                     # calibrated so the percentage shown in the UI is a real probability
                     ("clf", CalibratedClassifierCV(lr, method="isotonic", cv=5))])


def best_fbeta_threshold(y_true, score, beta=BETA):
    p, r, t = precision_recall_curve(y_true, score)
    f = (1 + beta**2) * p * r / np.clip(beta**2 * p + r, 1e-9, None)
    return float(t[np.argmax(f[:-1])])


def sweep(y_true, proba):
    """Test-set trade-off table (for information only; the threshold is NOT chosen from it)."""
    rows = []
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        pred = proba >= thr
        tp = int((pred & (y_true == 1)).sum()); fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        rows.append({"threshold": thr, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "false_alarms": fp, "missed_fakes": fn})
    return rows


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fake_job_postings.csv"
    X, y = load(csv_path)
    # Group-aware ~75/25 split: identical cleaned texts never straddle train and test.
    groups = pd.factorize(X["text"].str.strip())[0]
    sgkf = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    keep = ~X_train["text"].str.strip().duplicated().values   # one copy of each training text
    print(f"train {len(X_train)} -> {int(keep.sum())} after de-duplication; test {len(X_test)} "
          f"({int(y_test.sum())} fake)", flush=True)
    X_train, y_train = X_train[keep], y_train[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    cols = ["text", "raw"]
    # Threshold comes from out-of-fold *training* scores, so the test set stays untouched.
    oof = cross_val_predict(make_pipeline(), X_train[cols], y_train, cv=cv, method="predict_proba")[:, 1]
    thr = best_fbeta_threshold(y_train, oof)
    pipe = make_pipeline().fit(X_train[cols], y_train)
    proba = pipe.predict_proba(X_test[cols])[:, 1]
    pred = (proba >= thr).astype(int)
    (tn, fp), (fn, tp) = confusion_matrix(y_test, pred)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    metrics = {
        "threshold": thr,
        "beta": BETA,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": 2 * precision * recall / max(precision + recall, 1e-9),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "sweep": sweep(y_test.values, proba),
    }
    print(", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k not in ("confusion_matrix", "sweep")), flush=True)
    print(f"  confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    for r in metrics["sweep"]:
        print(f"    thr {r['threshold']:.2f}: precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"false alarms={r['false_alarms']} missed={r['missed_fakes']}")

    # Per-term weights (log-odds) of the word features, for the UI's word explanations.
    vocab = pipe.named_steps["ct"].named_transformers_["tfidf"].vocabulary_
    coef = np.mean([c.estimator.coef_[0] for c in pipe.named_steps["clf"].calibrated_classifiers_], axis=0)
    term_weights = {w: float(coef[i]) for w, i in vocab.items()}

    joblib.dump({"model": pipe, "threshold": thr, "term_weights": term_weights},
                HERE / "model.joblib", compress=3)
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
