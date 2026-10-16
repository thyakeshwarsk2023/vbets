"""Compare baseline and current full policy on held-out and unseen datasets."""
import json
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

from preprocess import LABELS, bow_vector, normalize
from safety_rules import matched_red_flag
from triage import classify_text

LEGACY_RED_FLAGS = [
    "cannot breath", "not breath", "unconscious", "unconsious", "unresponsive",
    "not responding", "heart attack", "having a stroke", "seizure", "choking",
    "overdose", "will not stop bleeding", "bleeding heavily", "kill myself",
    "end my life", "suicid", "no pulse", "no heartbeat", "turning blue",
]
SEED = 42


def clopper_pearson(successes, total):
    lo = beta.ppf(0.025, successes, total - successes + 1) if successes else 0.0
    hi = beta.ppf(0.975, successes + 1, total - successes) if successes < total else 1.0
    return lo, hi


def baseline_predictions(texts, model, vocab):
    matrix = np.stack([bow_vector(text, vocab, use_negation=False) for text in texts])
    probs = model.predict(matrix, verbose=0)
    preds = []
    for text, p in zip(texts, probs):
        label = LABELS[int(p.argmax())]
        norm = normalize(text)
        if any(flag in norm for flag in LEGACY_RED_FLAGS):
            label = "RED"
        elif p.max() < 0.50 and label == "GREEN":
            label = "YELLOW"
        preds.append(label)
    return preds


def current_predictions(texts, model, vocab):
    return [classify_text(text, model, vocab) for text in texts]


def report_set(name, frame, before, after, results=None):
    truth = frame["label"].tolist()
    labels = LABELS
    cm_before = confusion_matrix(truth, before, labels=labels)
    cm_after = confusion_matrix(truth, after, labels=labels)
    red_n = sum(label == "RED" for label in truth)
    old_red = int(cm_before[0, 0])
    new_red = int(cm_after[0, 0])
    old_ci = clopper_pearson(old_red, red_n)
    new_ci = clopper_pearson(new_red, red_n)
    print(f"\n=== {name} (N={len(frame)}) ===")
    print("Baseline pipeline confusion matrix (rows=true, cols=pred [RED,YELLOW,GREEN]):")
    print(cm_before)
    print(f"Baseline RED recall: {old_red}/{red_n} = {old_red/red_n:.4f}, 95% CI [{old_ci[0]:.4f}, {old_ci[1]:.4f}]")
    print(f"Baseline accuracy: {accuracy_score(truth, before):.4f}")
    print("Current pipeline confusion matrix (rows=true, cols=pred [RED,YELLOW,GREEN]):")
    print(cm_after)
    print(f"Current RED recall: {new_red}/{red_n} = {new_red/red_n:.4f}, 95% CI [{new_ci[0]:.4f}, {new_ci[1]:.4f}]")
    print(f"Current accuracy: {accuracy_score(truth, after):.4f}")
    if results is not None:
        new_red_rule_fp = sum(
            truth[i] != "RED" and before[i] != "RED" and after[i] == "RED"
            and matched_red_flag(text) is not None
            for i, text in enumerate(frame["text"].astype(str))
        )
        print(f"Current explicit RED-rule matches on non-RED truth: {new_red_rule_fp}")
        yellow_escalations = [r for r in results if r["confidence_escalation"] == "low_confidence_yellow_to_red"]
        print(f"Low-confidence YELLOW->RED escalations: {len(yellow_escalations)}")
    return before, after


def main():
    old_model = keras.models.load_model("model/triage_mlp.keras")
    old_vocab = json.load(open("model/vocab.json"))
    new_model = keras.models.load_model("model/triage_mlp_v2.keras")
    new_vocab = json.load(open("model/vocab_v2.json"))

    full = pd.read_csv("data/triage_dataset.csv").dropna().reset_index(drop=True)
    y = full["label"].map({label: i for i, label in enumerate(LABELS)})
    _, test_idx = train_test_split(np.arange(len(full)), test_size=0.2, stratify=y, random_state=SEED)
    heldout = full.iloc[test_idx].reset_index(drop=True)
    unseen = pd.read_csv("data/unseen_test.csv")

    for name, frame in (("Original held-out", heldout), ("Unseen", unseen)):
        texts = frame["text"].astype(str).tolist()
        before = baseline_predictions(texts, old_model, old_vocab)
        results = current_predictions(texts, new_model, new_vocab)
        after = [result["label"] for result in results]
        report_set(name, frame, before, after, results)
        if name == "Unseen":
            for sample_number in (32, 45):
                row = unseen.iloc[sample_number - 1]
                result = results[sample_number - 1]
                print(f"Sample #{sample_number}: true={row['label']} baseline={before[sample_number-1]} current={result['label']} rule={result['matched_rule']}")


if __name__ == "__main__":
    main()
