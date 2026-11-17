# Fake Job Posting Detection

Classifies job postings as fake or genuine from company profile, description and requirements text (TF-IDF + Random Forest), with a Streamlit UI.

## Setup
```
pip install -r requirements.txt
# extract fake_job_postings.csv from csvzip.rar (e.g. `bsdtar -xf csvzip.rar`)
python train.py fake_job_postings.csv   # writes model.joblib and metrics.json
streamlit run app.py          # new UI
# streamlit run app_legacy.py   # original report-style UI
```

## Method
- Text = company_profile + description + requirements, cleaned in `text_utils.py` (shared by training and the app).
- Stratified 75/25 split. TF-IDF (20,000 uni/bi-grams, sublinear tf) feeds a class-weighted, probability-calibrated logistic regression; a Random Forest is trained for comparison. Imbalance is handled with class weights, not resampling.
- The decision threshold maximises F1 on out-of-fold *training* scores, so the test set is only used for the reported metrics.
- The best PR-AUC model is saved to `model.joblib` together with its threshold and per-term weights (used by the UI to show which words pushed the score).

`project.ipynb` is the original exploratory notebook; `train.py` supersedes its modelling section.
