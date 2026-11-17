"""Train the fake-job-posting classifier and save an inference bundle.

Usage:  python train.py [path/to/fake_job_postings.csv]

Outputs (next to this file):
  model.joblib  - dict: {"pipeline", "threshold", "term_weights"}
                  pipeline.predict_proba(cleaned_text) gives calibrated P(fake);
                  threshold is the F1-optimal cut-off chosen from out-of-fold training scores.
  metrics.json  - hold-out metrics (at the tuned threshold) for every model compared
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from text_utils import clean_text

HERE = Path(__file__).parent
SEED = 42
MAX_FEATURES = 20000


def load(csv_path):
    df = pd.read_csv(csv_path).fillna("")
    raw = (df["company_profile"] + " " + df["description"] + " " + df["requirements"]).tolist()
    with Pool(4) as pool:
        cleaned = pool.map(clean_text, raw, chunksize=200)
    return pd.Series(cleaned), df["fraudulent"].astype(int)


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
        ("clf", clf),
    ])


def candidates():
    # Class imbalance is handled with class weights instead of resampling.
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=2000)
    return {
        # calibrated so the probability shown in the UI is a real probability
        "logistic_regression": CalibratedClassifierCV(lr, method="isotonic", cv=5),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2,
            n_jobs=4, random_state=SEED),
    }


def best_f1_threshold(y_true, score):
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fake_job_postings.csv"
    X, y = load(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    metrics, fitted = {}, {}
    for name, clf in candidates().items():
        # Threshold comes from out-of-fold *training* scores, so the test set stays untouched.
        oof = cross_val_predict(make_pipeline(clf), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_train, oof)

        pipe = make_pipeline(clf).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)
        (tn, fp), (fn, tp) = confusion_matrix(y_test, pred)
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        metrics[name] = {
            "threshold": thr,
            "precision_fake": precision,
            "recall_fake": recall,
            "f1_fake": 2 * precision * recall / max(precision + recall, 1e-9),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
        fitted[name] = (pipe, thr)
        print(f"{name}: " + ", ".join(
            f"{k}={v:.3f}" for k, v in metrics[name].items() if k != "confusion_matrix"), flush=True)

    best = max(metrics, key=lambda n: metrics[n]["pr_auc"])
    metrics["selected"] = best
    print("selected:", best)

    pipe, thr = fitted[best]
    # Per-term weights (log-odds contribution) for the UI's explanation; linear models only.
    term_weights = {}
    final = pipe.named_steps["clf"]
    if hasattr(final, "calibrated_classifiers_"):
        coef = np.mean([c.estimator.coef_[0] for c in final.calibrated_classifiers_], axis=0)
        vocab = pipe.named_steps["tfidf"].vocabulary_
        term_weights = {w: float(coef[i]) for w, i in vocab.items()}

    joblib.dump({"pipeline": pipe, "threshold": thr, "term_weights": term_weights},
                HERE / "model.joblib", compress=3)
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
