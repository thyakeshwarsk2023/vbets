"""Shared SLP pipeline: normalization -> tokenization -> lemmatization -> vectorization."""
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

LABELS = ["RED", "YELLOW", "GREEN"]
USE_NEGATION_FEATURES = True
NEGATION_CUES = {"no", "not", "denies", "without", "never", "cannot"}
NEGATION_WINDOW = 5


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


def preprocess(text: str, use_negation=None):
    """Tokenize and lemmatize, optionally marking tokens in a negation scope.

    Negation features are prefixed with ``NEG_`` and apply to the next five
    tokens after a cue. ``cannot`` is included because normalize() expands
    contractions such as "can't" to that form.
    """
    if use_negation is None:
        use_negation = USE_NEGATION_FEATURES
    words = lemmatize(tokenize(text))
    if not use_negation:
        return words
    cue_positions = [i for i, word in enumerate(words) if word in NEGATION_CUES]
    return [
        f"NEG_{word}" if any(cue < i <= cue + NEGATION_WINDOW for cue in cue_positions) else word
        for i, word in enumerate(words)
    ]


def build_vocab(texts, use_negation=None):
    words = sorted({w for t in texts for w in preprocess(t, use_negation=use_negation)})
    return {w: i for i, w in enumerate(words)}


def bow_vector(text: str, vocab: dict, use_negation=None) -> np.ndarray:
    v = np.zeros(len(vocab), dtype="float32")
    for w in preprocess(text, use_negation=use_negation):
        if w in vocab:
            v[vocab[w]] = 1.0
    return v


def seq_vector(text: str, vocab: dict, max_len: int = 30, use_negation=None) -> np.ndarray:
    """Word-index sequence for the Embedding model (0 = padding/unknown)."""
    ids = [vocab[w] + 1 for w in preprocess(text, use_negation=use_negation) if w in vocab][:max_len]
    return np.array(ids + [0] * (max_len - len(ids)), dtype="int32")
