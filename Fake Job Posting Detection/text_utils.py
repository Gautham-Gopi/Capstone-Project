"""Text cleaning shared by training (train.py) and inference (detect_page.py)."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

for _resource in ("wordnet", "omw-1.4", "stopwords"):
    nltk.download(_resource, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()
_TOKENIZER = RegexpTokenizer(r"\w+")
_HTML = re.compile(r"<.*?>")


def clean_text(text):
    """Lowercase, strip noise, stem, lemmatize and drop stopwords."""
    if not isinstance(text, str):
        return ""
    text = _HTML.sub(" ", text)
    text = text.lower()
    text = re.sub(r"^https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S*@\S*\s?", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"'ve", " have ", text)
    text = re.sub(r"'re", " are ", text)
    text = re.sub(r"'d", " would ", text)
    text = re.sub(r"'ll", " will ", text)
    text = re.sub(r"[0-9]+", "", text)
    text = re.sub(r"[-_]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = _TOKENIZER.tokenize(text)
    tokens = [_LEMMATIZER.lemmatize(_STEMMER.stem(t)) for t in tokens]
    return " ".join(t for t in tokens if t not in _STOPWORDS)


def is_probably_english(text, min_words=15, threshold=0.5):
    """False only when the text is clearly another language.

    Language detection is unreliable on very short or jargon-only text, so postings shorter than
    `min_words` words are never rejected, and any detection failure fails open (returns True).
    """
    if len(re.findall(r"[^\W\d_]+", text)) < min_words:
        return True
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0  # deterministic results
        en_prob = next((l.prob for l in detect_langs(text) if l.lang == "en"), 0.0)
        return en_prob >= threshold
    except Exception:
        return True
