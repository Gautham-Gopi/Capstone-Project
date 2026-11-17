# Fake Job Posting Detection

Classifies job postings as fake or genuine from company profile, description and requirements text (TF-IDF + calibrated logistic regression), with a Streamlit UI.

## Setup
```
pip install -r ../requirements.txt   # requirements.txt lives at the repo root
# extract fake_job_postings.csv from csvzip.rar (e.g. `bsdtar -xf csvzip.rar`)
python train.py fake_job_postings.csv   # writes model.joblib and metrics.json
streamlit run app.py          # new UI
# streamlit run app_legacy.py   # original report-style UI
```

## Method
- Text = title + company_profile + description + requirements + benefits, cleaned in `text_utils.py` (shared by training and the app).
- Group-aware 75/25 split: postings with identical cleaned text never straddle train and test, and training duplicates are dropped, so the reported metrics are not inflated by postings the model has already seen.
- Features: word TF-IDF (20,000 uni/bi-grams of the cleaned text) + character TF-IDF (3-5-grams of the raw text) feed a class-weighted (C=30), probability-calibrated logistic regression. Imbalance is handled with class weights, not resampling. The model reads the text only; yes/no posting details (logo, questions, remote, salary) added nothing once character n-grams were used, so they were removed.
- The decision threshold maximises F2 (favouring recall) on out-of-fold *training* scores, so the test set is only used for the reported metrics.
- `scoring.py` holds the scoring logic shared by the apps; `posting_parser.py` splits a pasted posting into title / company / description / requirements / benefits.
- Only English postings are supported; the app warns on other languages.

`project.ipynb` is the original exploratory notebook; `train.py` supersedes its modelling section.
