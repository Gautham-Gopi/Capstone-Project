"""Fake Job Posting Detector - Streamlit frontend.

The original multi-page report UI is kept in app_legacy.py (run: streamlit run app_legacy.py).
"""
import json
import random
import re
import time
from pathlib import Path

import joblib
import streamlit as st

from posting_parser import parse_posting
from scoring import build_text, score
from text_utils import clean_text, is_probably_english

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"

st.set_page_config(page_title="Job Guard · Is this job real?", page_icon="📰",
                   layout="centered", initial_sidebar_state="collapsed")

# Try-an-example pools. A button press draws a random posting from the pool (never the same one twice in a row).
# Hand-written pair + postings from the held-out part of the public EMSCAD dataset (samples.json).
POOLS = {
    "suspicious": [{
        "title": "Data Entry Clerk - Work From Home",
        "company": "",
        "desc": "WORK FROM HOME! Earn $5000 weekly with no experience needed. Start immediately, "
                "no interview. Send your bank details and a small processing fee to receive your starter kit.",
        "req": "",
        "benefits": "",
    }],
    "genuine": [{
        "title": "Backend Engineer",
        "company": "Acme Corp is a 20-year-old software company headquartered in Austin with 500 employees "
                   "and a strong engineering culture.",
        "desc": "We are seeking a Backend Engineer to design and maintain REST services, participate in "
                "code reviews and mentor junior developers.",
        "req": "Bachelor's degree in Computer Science, 3+ years of Python or Java experience, "
               "familiarity with PostgreSQL and AWS.",
        "benefits": "Health insurance, 25 days of paid holiday, and a yearly learning budget.",
    }],
}
try:
    for _p in json.loads((Path(__file__).parent / "samples.json").read_text(encoding="utf-8")):
        POOLS["suspicious" if _p["label"] else "genuine"].append(
            {k: _p[k] for k in ("title", "company", "desc", "req", "benefits")})
except (OSError, ValueError, KeyError):
    pass  # the two hand-written examples are still available

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap');
html, body, [class*="css"], .stMarkdown, label, textarea, button {font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif !important;}
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 760px;}
.masthead {text-align:center; border-bottom: 3px double #2B2118; padding-bottom: 1rem; margin-bottom: 1.8rem;}
.masthead .kicker {font-size:.78rem; letter-spacing:3px; text-transform:uppercase; color:#C2572B; font-weight:600;}
.masthead h1 {font-size:3.1rem; margin:.2rem 0 .3rem 0; font-weight:700; letter-spacing:-.5px; color:#2B2118;}
.masthead .langnote {display:inline-block; margin-top:.8rem; font-size:.78rem; letter-spacing:1px; text-transform:uppercase; color:#8A7A68; border:1px solid #C9BBA3; padding:.2rem .7rem;}
.langwarn {border-left:4px solid #9A6A12; background:#FFF6E0; padding:.9rem 1.1rem; margin:1.4rem 0 1rem 0; color:#4A3D30;}
.langwarn b {color:#2B2118; font-size:1.15rem;}
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
.loader {margin:1.6rem 0 1rem 0; padding:1.6rem 0 1.2rem 0; border-top:3px solid #2B2118; border-bottom:1px solid #E3D8C5; text-align:center;}
.loader .label {font-size:.75rem; text-transform:uppercase; letter-spacing:3px; color:#8A7A68;}
.loader .msg {font-size:1.5rem; font-weight:600; font-style:italic; color:#2B2118; margin:.5rem 0 1rem 0;}
.loader .bars {display:inline-flex; gap:6px; align-items:flex-end; height:34px;}
.loader .bars span {width:8px; background:#C2572B; animation: jg-bar 1s ease-in-out infinite;}
.loader .bars span:nth-child(2) {animation-delay:.15s;} .loader .bars span:nth-child(3) {animation-delay:.3s;}
.loader .bars span:nth-child(4) {animation-delay:.45s;} .loader .bars span:nth-child(5) {animation-delay:.6s;}
@keyframes jg-bar {0%,100% {height:8px; opacity:.45;} 50% {height:34px; opacity:1;}}
div[data-testid="stButton"] > button {border-radius:2px; font-weight:600; border:1px solid #2B2118;}
button[kind="primary"] {background:#C2572B; border:1px solid #C2572B; color:#fff;}
button[kind="primary"]:hover {background:#A44520; border-color:#A44520; color:#fff;}
.stTabs [data-baseweb="tab"] {font-size:1.05rem; font-weight:600; padding:.5rem 1.2rem;}
textarea {background:#FFFDF8 !important; border-radius:2px !important;}
</style>
"""


MIN_LOADER_SECONDS = 1.0
LOADER_HTML = """
<div class="loader">
  <div class="label">Please wait</div>
  <div class="msg">Reading the posting…</div>
  <div class="bars"><span></span><span></span><span></span><span></span><span></span></div>
</div>"""


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def original_words(raw):
    """Map each stemmed term back to the first word the user actually typed (e.g. 'servic' -> 'services')."""
    mapping = {}
    for word in re.findall(r"[A-Za-z]+", raw):
        stem = clean_text(word.lower())
        if stem and " " not in stem:
            mapping.setdefault(stem, word.lower())
    return mapping


def explain(bundle, text, raw, n=4):
    """Known words in the input, split by whether they push the score toward fake or genuine."""
    weights = bundle["term_weights"]
    shown = original_words(raw)
    present = {w for w in set(text.split()) if w in weights}
    known = len(present)
    toward_fake = sorted((w for w in present if weights[w] > 0), key=lambda w: -weights[w])[:n]
    toward_real = sorted((w for w in present if weights[w] < 0), key=lambda w: weights[w])[:n]
    label = lambda stems: [shown.get(w, w) for w in stems]
    return label(toward_fake), label(toward_real), known


def pick_sample(kind):
    """Random posting from POOLS[kind], different from the one shown last time for that kind."""
    pool = POOLS[kind]
    last = st.session_state.get(f"last_{kind}")
    choices = [i for i in range(len(pool)) if i != last] or [0]
    i = random.choice(choices)
    st.session_state[f"last_{kind}"] = i
    return pool[i]


def set_sample(kind):
    smp = pick_sample(kind)
    st.session_state.title, st.session_state.company = smp["title"], smp["company"]
    st.session_state.desc, st.session_state.req = smp["desc"], smp["req"]
    st.session_state.benefits = smp["benefits"]


def clear_inputs():
    for k in ("title", "company", "desc", "req", "benefits"):
        st.session_state[k] = ""


def full_text_for(smp):
    """Render an example entry as one pasted posting, with section headings like a real ad."""
    parts = [smp["title"]]
    if smp["company"]:
        parts.append("About Us\n" + smp["company"])
    parts.append("Job Description\n" + smp["desc"])
    if smp["req"]:
        parts.append("Requirements\n" + smp["req"])
    if smp["benefits"]:
        parts.append("Benefits\n" + smp["benefits"])
    return "\n\n".join(parts)


def set_full_sample(kind):
    st.session_state.full = full_text_for(pick_sample(kind))


def clear_full():
    st.session_state.full = ""


def verdict_html(prob, known, threshold):
    if known == 0:
        cls, title = "unsure", "Not enough to go on."
        detail = "None of these words are ones the model recognises, so any score would be a guess."
        fill = ""
    else:
        if prob >= max(0.5, threshold):
            cls, title, note = "fake", "This looks fake.", "The posting resembles known fraudulent listings."
        elif prob >= threshold:
            cls, title, note = "fake", "This looks suspicious.", (
                "It was flagged for a closer look. Most postings with a score like this are genuine, but "
                "enough are fake that it is worth checking the employer before you apply.")
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
  <div class="langnote">English-language postings only &middot; other languages are planned for a future version</div>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("`model.joblib` not found. Run `python train.py <path to fake_job_postings.csv>` first.")
    st.stop()

for k in ("title", "company", "desc", "req", "benefits", "full"):
    st.session_state.setdefault(k, "")


def render_result(title, company, desc, req, benefits, company_known=True):
    """Score a posting and draw the verdict. company_known=False suppresses the missing-company note."""
    raw, text = build_text(title, company, desc, req, benefits)
    if not raw.strip():
        st.warning("Add some posting text first.")
        return

    if not is_probably_english(raw):
        st.markdown("<div class='langwarn'><b>This posting doesn&rsquo;t look like English.</b><br>"
                    "Job Guard was trained on English-language postings only, so a score for this text would be "
                    "unreliable and we haven&rsquo;t given one. Support for other languages is planned for a "
                    "future version. In the meantime, please paste an English version of the posting.</div>",
                    unsafe_allow_html=True)
        return

    loader = st.empty()
    loader.markdown(LOADER_HTML, unsafe_allow_html=True)
    started = time.perf_counter()
    if not text.strip():
        loader.empty()
        st.warning("Add some posting text first.")
        return
    bundle = load_model()  # slow only on the first request after the app wakes up
    prob, threshold = score(bundle, text, raw)
    toward_fake, toward_real, known = explain(bundle, text, raw)
    # Scoring itself takes milliseconds; keep the loader up briefly so it doesn't just flash.
    time.sleep(max(0.0, MIN_LOADER_SECONDS - (time.perf_counter() - started)))
    loader.empty()
    st.markdown(verdict_html(prob, known, threshold), unsafe_allow_html=True)
    if company_known and not company.strip():
        st.markdown("<div class='sidebar-note'><b>Worth noting:</b> no company description was given. "
                    "About 68% of fraudulent postings in the dataset had none, versus 16% of genuine ones.</div>",
                    unsafe_allow_html=True)
    if toward_fake or toward_real:
        st.markdown("##### Words that influenced the score")
        chips = "".join(f"<span class='chip fake'>{w}</span>" for w in toward_fake)
        chips += "".join(f"<span class='chip real'>{w}</span>" for w in toward_real)
        st.markdown(chips, unsafe_allow_html=True)
        st.caption("Red words nudge the score toward fake, green toward genuine. Each word is a weak signal "
                   "on its own; the overall score weighs every word in the posting together.")
    st.caption("This is a statistical estimate, not a verdict. Verify employers independently and never pay to get a job.")


tab_full, tab_sections = st.tabs(["Paste the whole posting", "Fill in sections"])
analyzed = False

with tab_full:
    st.markdown("##### Try an example")
    st.caption("Each press picks a different example, drawn from a public research dataset of 2012–2014 postings "
               "(EMSCAD). Labels are the dataset's; company details may be anonymised or copied.")
    f1, f2, f3 = st.columns(3)
    f1.button("Suspicious example", key="ex_full_1", on_click=set_full_sample, args=("suspicious",), width="stretch")
    f2.button("Genuine example", key="ex_full_2", on_click=set_full_sample, args=("genuine",), width="stretch")
    f3.button("Clear", key="clear_full", on_click=clear_full, width="stretch")

    st.markdown("##### The posting")
    st.text_area("Full job posting", key="full", height=300, label_visibility="collapsed",
                 placeholder="Paste the entire job ad here, headings and all. We'll separate the company "
                             "description, the job description and the requirements for you.")
    go_full = st.button("Check this posting", key="go_full", type="primary", width="stretch")

    if go_full:
        analyzed = True
        parsed = parse_posting(st.session_state.full)
        render_result(parsed.title, parsed.company, parsed.description, parsed.requirements, parsed.benefits,
                      company_known=parsed.structured)
        if st.session_state.full.strip():
            with st.expander("How we split your posting"):
                if parsed.company_inferred:
                    st.caption("No “About us” style heading was found, so sentences that read like a company "
                               "introduction were picked out of the text. The score is the same either way.")
                elif parsed.structured:
                    st.caption("Sections are detected from headings such as “About us”, “Responsibilities” "
                               "and “Requirements”.")
                else:
                    st.caption("No section headings were recognised, so the whole text was treated as the job "
                               "description. The score is the same either way.")
                for label, value in (("Job title", parsed.title),
                                     ("Company description", parsed.company),
                                     ("Job description", parsed.description),
                                     ("Requirements", parsed.requirements),
                                     ("Benefits", parsed.benefits)):
                    st.markdown(f"**{label}**")
                    st.text(value if value else "(none found)")
                if parsed.ignored:
                    st.markdown("**Not used by the model** (location, how to apply…)")
                    st.text(parsed.ignored)

with tab_sections:
    st.markdown("##### Try an example")
    st.caption("Each press picks a different example, drawn from a public research dataset of 2012–2014 postings "
               "(EMSCAD). Labels are the dataset's; company details may be anonymised or copied.")
    b1, b2, b3 = st.columns(3)
    b1.button("Suspicious example", key="ex_sec_1", on_click=set_sample, args=("suspicious",), width="stretch")
    b2.button("Genuine example", key="ex_sec_2", on_click=set_sample, args=("genuine",), width="stretch")
    b3.button("Clear", key="clear_sec", on_click=clear_inputs, width="stretch")

    st.markdown("##### The posting")
    st.text_input("Job title", key="title", placeholder="e.g. Backend Engineer")
    st.text_area("Company description", key="company", height=90,
                 placeholder="Who is the employer? Leave blank if the posting doesn't say.")
    st.text_area("Job description", key="desc", height=150,
                 placeholder="What will the person do? Pay, perks, how to apply…")
    st.text_area("Requirements", key="req", height=90,
                 placeholder="Skills, education, experience…")
    st.text_area("Benefits", key="benefits", height=70,
                 placeholder="Health insurance, holiday, perks… Leave blank if none are listed.")
    go_sections = st.button("Check this posting", key="go_sections", type="primary", width="stretch")

    if go_sections:
        analyzed = True
        render_result(st.session_state.title, st.session_state.company, st.session_state.desc,
                      st.session_state.req, st.session_state.benefits)

if not analyzed:
    st.markdown("""<div class='howto'><b>How it works.</b> The text is cleaned and converted to word-frequency features,
    then scored by a logistic-regression model trained on 17,880 labelled postings. Use the examples above to see both outcomes.</div>""",
                unsafe_allow_html=True)
