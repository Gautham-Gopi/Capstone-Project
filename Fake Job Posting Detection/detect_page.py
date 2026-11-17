from pathlib import Path

import joblib
import streamlit as st

from scoring import build_text, score

MODEL_PATH = Path(__file__).parent / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def show_detect_page():
    st.title("Detection of Fake Job Postings")

    if not MODEL_PATH.exists():
        st.error("model.joblib not found. Run `python train.py <path to fake_job_postings.csv>` first.")
        return

    company_description = st.text_area(label="Enter Company Description:")
    job_description = st.text_area(label="Enter Job Description:")
    job_requirements = st.text_area(label="Enter Job Requirements:")

    if not st.button("Detect"):
        return

    raw, text = build_text(company=company_description, description=job_description,
                         requirements=job_requirements)
    if not text.strip():
        st.warning("Please enter some text so there is something to classify.")
        return

    bundle = load_model()
    fake_prob, threshold, _ = score(bundle, text, raw)

    if fake_prob >= threshold:
        st.error("Fake Job Posting")
    else:
        st.success("Genuine Job Posting")
    st.caption(f"Estimated probability of being fake: {fake_prob:.1%}")
