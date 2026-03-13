"""
Text cleaning utilities for sentiment analysis.
"""

import re
from typing import List

_NLTK_READY = False


def ensure_nltk_resources() -> None:
    """
    Ensure NLTK resources are available locally (only once per process).
    """
    global _NLTK_READY
    if _NLTK_READY:
        return

    import nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

    _NLTK_READY = True


def basic_clean(text: str) -> str:
    """
    Fast deterministic cleaning for tweets.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """
    Tokenization + optional stopwords removal + lemmatization.
    """
    ensure_nltk_resources()

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    cleaned = basic_clean(text)
    if not cleaned:
        return ""

    tokens: List[str] = word_tokenize(cleaned)

    if remove_stopwords:
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw and len(t) > 1]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)