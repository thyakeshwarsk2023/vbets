"""Shared SLP pipeline: normalization -> tokenization -> lemmatization -> vectorization."""
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

LABELS = ["RED", "YELLOW", "GREEN"]


def ensure_nltk():
    for pkg, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk()
_lem = WordNetLemmatizer()


def normalize(text: str) -> str:
    """Lowercase and expand contractions so 'can't' and 'cannot' look the same."""
    t = text.lower().strip()
    t = re.sub(r"\bwon'?t\b", "will not", t)
    t = re.sub(r"\bcan'?t\b|\bcan not\b", "cannot", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"\bi'm\b", "i am", t)
    return t


def tokenize(text: str):
    return re.findall(r"[a-z]+", normalize(text))


def lemmatize(tokens):
    return [_lem.lemmatize(_lem.lemmatize(w, "v"), "n") for w in tokens]


def preprocess(text: str):
    return lemmatize(tokenize(text))


def build_vocab(texts):
    words = sorted({w for t in texts for w in preprocess(t)})
    return {w: i for i, w in enumerate(words)}


def bow_vector(text: str, vocab: dict) -> np.ndarray:
    v = np.zeros(len(vocab), dtype="float32")
    for w in preprocess(text):
        if w in vocab:
            v[vocab[w]] = 1.0
    return v


def seq_vector(text: str, vocab: dict, max_len: int = 30) -> np.ndarray:
    """Word-index sequence for the Embedding model (0 = padding/unknown)."""
    ids = [vocab[w] + 1 for w in preprocess(text) if w in vocab][:max_len]
    return np.array(ids + [0] * (max_len - len(ids)), dtype="int32")
