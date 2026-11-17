"""Shared scoring logic for the app (app.py, app_legacy.py) and training (train.py).

The saved bundle (model.joblib) looks like:
    {"model": Pipeline, "threshold": float, "term_weights": {stem: log-odds weight}}

The model reads the posting text only (title + company + description + requirements + benefits):
word n-grams of the cleaned text plus character n-grams of the raw text.
"""
import pandas as pd

from text_utils import clean_text


def build_text(title="", company="", description="", requirements="", benefits=""):
    """Raw sections -> (raw_text, cleaned_text). Order matches training."""
    raw = " ".join(p for p in (title, company, description, requirements, benefits) if p)
    return raw, clean_text(raw)


def frame(cleaned, raw=""):
    return pd.DataFrame([{"text": cleaned, "raw": raw}])


def score(bundle, cleaned, raw=""):
    """Return (probability_of_fake, decision_threshold)."""
    prob = float(bundle["model"].predict_proba(frame(cleaned, raw))[0][1])
    return prob, bundle["threshold"]
