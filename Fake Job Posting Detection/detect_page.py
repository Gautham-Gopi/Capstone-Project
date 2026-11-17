from pathlib import Path

import joblib
import streamlit as st

from text_utils import clean_text

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

    text = clean_text(" ".join([company_description, job_description, job_requirements]))
    if not text.strip():
        st.warning("Please enter some text so there is something to classify.")
        return

    bundle = load_model()
    fake_prob = float(bundle["pipeline"].predict_proba([text])[0][1])

    if fake_prob >= bundle["threshold"]:
        st.error("Fake Job Posting")
    else:
        st.success("Genuine Job Posting")
    st.caption(f"Estimated probability of being fake: {fake_prob:.1%}")
