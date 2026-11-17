"""Fake Job Posting Detector - Streamlit frontend.

The original multi-page report UI is kept in app_legacy.py (run: streamlit run app_legacy.py).
"""
from pathlib import Path

import joblib
import streamlit as st

from text_utils import clean_text

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"

st.set_page_config(page_title="Job Guard · Is this job real?", page_icon="📰",
                   layout="centered", initial_sidebar_state="collapsed")

SAMPLES = {
    "Suspicious offer": {
        "company": "",
        "desc": "WORK FROM HOME! Earn $5000 weekly with no experience needed. Start immediately, "
                "no interview. Send your bank details and a small processing fee to receive your starter kit.",
        "req": "",
    },
    "Legitimate engineering role": {
        "company": "Acme Corp is a 20-year-old software company headquartered in Austin with 500 employees "
                   "and a strong engineering culture.",
        "desc": "We are seeking a Backend Engineer to design and maintain REST services, participate in "
                "code reviews and mentor junior developers.",
        "req": "Bachelor's degree in Computer Science, 3+ years of Python or Java experience, "
               "familiarity with PostgreSQL and AWS.",
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
html, body, [class*="css"], .stMarkdown, label, textarea, button {font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;}
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 760px;}
.masthead {text-align:center; border-bottom: 3px double #2B2118; padding-bottom: 1rem; margin-bottom: 1.8rem;}
.masthead .kicker {font-size:.78rem; letter-spacing:3px; text-transform:uppercase; color:#C2572B; font-weight:600;}
.masthead h1 {font-size:3.1rem; margin:.2rem 0 .3rem 0; font-weight:700; letter-spacing:-.5px; color:#2B2118;}
.masthead p {font-size:1.12rem; font-style:italic; color:#6B5B4B; margin:0;}
h5 {font-weight:600 !important; color:#2B2118; border-bottom:1px solid #E3D8C5; padding-bottom:.3rem;}
.verdict {margin:1.6rem 0 1rem 0; padding:1.4rem 0 1rem 0; border-top:3px solid #2B2118; border-bottom:1px solid #E3D8C5;}
.verdict .label {font-size:.75rem; text-transform:uppercase; letter-spacing:3px; color:#8A7A68;}
.verdict .title {font-size:2.5rem; line-height:1.15; font-weight:700; margin:.3rem 0 .4rem 0;}
.verdict.fake .title {color:#B3261E;} .verdict.real .title {color:#3F7D58;} .verdict.unsure .title {color:#9A6A12;}
.verdict .note {font-size:1.1rem; color:#4A3D30; font-style:italic;}
.meter {position:relative; height:6px; background:#E3D8C5; margin:1.4rem 0 .4rem 0;}
.meter .fill {position:absolute; left:0; top:0; height:6px;}
.verdict.fake .fill {background:#B3261E;} .verdict.real .fill {background:#3F7D58;} .verdict.unsure .fill {background:#9A6A12;}
.meter-scale {display:flex; justify-content:space-between; color:#8A7A68; font-size:.78rem;}
.chip {display:inline-block; padding:.15rem .6rem; margin:.2rem .3rem .2rem 0; border:1px solid #C9BBA3;
  background:#FFFDF8; font-size:.92rem; color:#4A3D30;}
.chip.fake {border-color:#B3261E; color:#B3261E;} .chip.real {border-color:#3F7D58; color:#3F7D58;}
.sidebar-note {border-left:3px solid #C2572B; padding:.2rem 0 .2rem .9rem; margin:.8rem 0; color:#4A3D30;}
.howto {color:#6B5B4B; font-size:1rem; line-height:1.8; margin-top:1.5rem;}
.howto b {color:#2B2118;}
div[data-testid="stButton"] > button {border-radius:2px; font-weight:600; border:1px solid #2B2118;}
button[kind="primary"] {background:#C2572B; border:1px solid #C2572B; color:#fff;}
button[kind="primary"]:hover {background:#A44520; border-color:#A44520; color:#fff;}
textarea {background:#FFFDF8 !important; border-radius:2px !important;}
</style>
"""


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def explain(bundle, text, n=6):
    """Known words in the input, split by whether they push the score toward fake or genuine."""
    weights = bundle["term_weights"]
    present = {w for w in set(text.split()) if w in weights}
    known = len(present)
    toward_fake = sorted((w for w in present if weights[w] > 0), key=lambda w: -weights[w])[:n]
    toward_real = sorted((w for w in present if weights[w] < 0), key=lambda w: weights[w])[:n]
    return toward_fake, toward_real, known


def set_sample(key):
    s = SAMPLES[key]
    st.session_state.company, st.session_state.desc, st.session_state.req = s["company"], s["desc"], s["req"]


def clear_inputs():
    st.session_state.company = st.session_state.desc = st.session_state.req = ""


def verdict_html(prob, known, threshold):
    if known == 0:
        cls, title = "unsure", "Not enough to go on."
        detail = "None of these words are ones the model recognises, so any score would be a guess."
        fill = ""
    else:
        if prob >= threshold:
            cls, title, note = "fake", "This looks fake.", "The posting resembles known fraudulent listings."
        elif prob >= threshold / 2:
            cls, title, note = "unsure", "Proceed with caution.", "There are some warning signs, but nothing conclusive."
        else:
            cls, title, note = "real", "This looks genuine.", "The posting resembles legitimate listings."
        detail = f"Estimated chance of fraud: <b>{prob:.0%}</b>. {note}"
        fill = f'<div class="fill" style="width:{max(prob * 100, 1):.1f}%"></div>'
    return f"""
    <div class="verdict {cls}">
      <div class="label">Our reading</div>
      <div class="title">{title}</div>
      <div class="note">{detail}</div>
      <div class="meter">{fill}</div>
      <div class="meter-scale"><span>Genuine</span><span>Fraudulent</span></div>
    </div>"""


# ----------------------------------------------------------------------------- layout
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="masthead">
  <div class="kicker">Job Posting Verification</div>
  <h1>Job Guard</h1>
  <p>Is this job posting real? Paste it below and find out.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Run `python train.py <path to fake_job_postings.csv>` first.")
    st.stop()

for k in ("company", "desc", "req"):
    st.session_state.setdefault(k, "")

st.markdown("##### Try an example")
b1, b2, b3 = st.columns(3)
b1.button(list(SAMPLES)[0], on_click=set_sample, args=(list(SAMPLES)[0],), width="stretch")
b2.button(list(SAMPLES)[1], on_click=set_sample, args=(list(SAMPLES)[1],), width="stretch")
b3.button("Clear", on_click=clear_inputs, width="stretch")

st.markdown("##### The posting")
st.text_area("Company description", key="company", height=90,
             placeholder="Who is the employer? Leave blank if the posting doesn't say.")
st.text_area("Job description", key="desc", height=150,
             placeholder="What will the person do? Pay, perks, how to apply…")
st.text_area("Requirements", key="req", height=90,
             placeholder="Skills, education, experience…")
analyze = st.button("Check this posting", type="primary", width="stretch")

if analyze:
    text = clean_text(" ".join([st.session_state.company, st.session_state.desc, st.session_state.req]))
    if not text.strip():
        st.warning("Add some posting text first.")
    else:
        with st.spinner("Reading the posting…"):
            bundle = load_model()
            prob = float(bundle["pipeline"].predict_proba([text])[0][1])
            toward_fake, toward_real, known = explain(bundle, text)
        st.markdown(verdict_html(prob, known, bundle["threshold"]), unsafe_allow_html=True)

        if not st.session_state.company.strip():
            st.markdown("<div class='sidebar-note'><b>Worth noting:</b> no company description was given. "
                        "About 68% of fraudulent postings in the dataset had none, versus 16% of genuine ones.</div>",
                        unsafe_allow_html=True)
        if toward_fake or toward_real:
            st.markdown("##### Words that influenced the score")
            chips = "".join(f"<span class='chip fake'>{w}</span>" for w in toward_fake)
            chips += "".join(f"<span class='chip real'>{w}</span>" for w in toward_real)
            st.markdown(chips, unsafe_allow_html=True)
            st.caption("Red words push the score toward fake, green toward genuine (shown stemmed).")
        st.caption("This is a statistical estimate, not a verdict. Verify employers independently and never pay to get a job.")
else:
    st.markdown("""<div class='howto'><b>How it works.</b> The text is cleaned and converted to word-frequency features,
    then scored by a logistic-regression model trained on 17,880 labelled postings. Use the examples above to see both outcomes.</div>""",
                unsafe_allow_html=True)
