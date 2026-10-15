"""Evaluation of saved V-BETS BoW MLP on independent unseen test set (data/unseen_test.csv).
Evaluates both the raw model and the complete production pipeline (model + red-flag rules + low-confidence escalation).
"""
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import LABELS, bow_vector, normalize

# Match app.py rule layer exactly
RED_FLAGS = ["cannot breath", "not breath", "unconscious", "unconsious", "unresponsive",
             "not responding", "heart attack", "having a stroke", "seizure", "choking",
             "overdose", "will not stop bleeding", "bleeding heavily", "kill myself",
             "end my life", "suicid", "no pulse", "no heartbeat", "turning blue"]
CONF_THRESHOLD = 0.50


def classify_pipeline(text: str, model, vocab):
    """Identical logic to app.py classify() function."""
    probs = model.predict(bow_vector(text, vocab)[None, :], verbose=0)[0]
    raw_label = LABELS[int(probs.argmax())]
    label = raw_label
    note = ""
    norm = normalize(text)
    if any(flag in norm for flag in RED_FLAGS):
        if label != "RED":
            note = "Escalated to RED by safety rule (red-flag phrase detected)."
        label = "RED"
    elif probs.max() < CONF_THRESHOLD and label == "GREEN":
        label = "YELLOW"
        note = "Low model confidence – escalated to YELLOW to be safe."
    return label, raw_label, probs, note


def main():
    print("=" * 70)
    print("V-BETS INDEPENDENT UNSEEN TEST EVALUATION")
    print("=" * 70)

    # 1. Load assets
    model = keras.models.load_model("model/triage_mlp.keras")
    vocab = json.load(open("model/vocab.json"))
    df = pd.read_csv("data/unseen_test.csv")
    print(f"Loaded unseen test dataset: {len(df)} samples")
    print(f"Class distribution: {dict(df['label'].value_counts())}")
    print(f"Borderline samples: {df['borderline'].sum()} of {len(df)}\n")

    # 2. Run inference
    records = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        true_label = str(row["label"])
        borderline = bool(row["borderline"])

        pipe_label, raw_label, probs, note = classify_pipeline(text, model, vocab)
        records.append({
            "idx": idx + 1,
            "text": text,
            "true_label": true_label,
            "raw_pred": raw_label,
            "pipe_pred": pipe_label,
            "probs": probs,
            "note": note,
            "borderline": borderline,
            "correct_raw": raw_label == true_label,
            "correct_pipe": pipe_label == true_label,
        })

    res_df = pd.DataFrame(records)
    y_true = res_df["true_label"].values
    y_raw = res_df["raw_pred"].values
    y_pipe = res_df["pipe_pred"].values

    # 3. Model Alone Results
    acc_raw = accuracy_score(y_true, y_raw)
    cm_raw = confusion_matrix(y_true, y_raw, labels=LABELS)
    rep_raw = classification_report(y_true, y_raw, labels=LABELS, zero_division=0)

    print("=" * 70)
    print("1. MODEL ALONE (NO RULE LAYER)")
    print("=" * 70)
    print(f"Accuracy: {acc_raw:.4f} ({acc_raw * 100:.2f}% - {res_df['correct_raw'].sum()}/{len(res_df)})")
    print("\nClassification Report:")
    print(rep_raw)
    print("Confusion Matrix (rows=True [RED, YELLOW, GREEN], cols=Pred [RED, YELLOW, GREEN]):")
    print(cm_raw)

    # 4. Full Pipeline Results
    acc_pipe = accuracy_score(y_true, y_pipe)
    cm_pipe = confusion_matrix(y_true, y_pipe, labels=LABELS)
    rep_pipe = classification_report(y_true, y_pipe, labels=LABELS, zero_division=0)

    print("\n" + "=" * 70)
    print("2. FULL PRODUCTION PIPELINE (MODEL + RED-FLAG RULES + CONFIDENCE ESCALATION)")
    print("=" * 70)
    print(f"Accuracy: {acc_pipe:.4f} ({acc_pipe * 100:.2f}% - {res_df['correct_pipe'].sum()}/{len(res_df)})")
    print("\nClassification Report:")
    print(rep_pipe)
    print("Confusion Matrix (rows=True [RED, YELLOW, GREEN], cols=Pred [RED, YELLOW, GREEN]):")
    print(cm_pipe)

    # 5. Rule Layer Impact Analysis
    escalations = res_df[res_df["raw_pred"] != res_df["pipe_pred"]]
    print("\n" + "=" * 70)
    print(f"3. RULE LAYER ACTIVATIONS ({len(escalations)} samples escalated)")
    print("=" * 70)
    if len(escalations) == 0:
        print("No cases triggered rule escalation.")
    else:
        for _, r in escalations.iterrows():
            print(f"- Sample #{r['idx']}: True={r['true_label']} | Model={r['raw_pred']} -> Pipeline={r['pipe_pred']}")
            print(f"  Text: \"{r['text']}\"")
            print(f"  Note: {r['note']}")
            print(f"  Probs: RED={r['probs'][0]:.4f}, YELLOW={r['probs'][1]:.4f}, GREEN={r['probs'][2]:.4f}")

    # 6. Misclassified Sentences (Full Pipeline)
    mis_pipe = res_df[~res_df["correct_pipe"]]
    print("\n" + "=" * 70)
    print(f"4. MISCLASSIFIED SENTENCES IN FULL PIPELINE ({len(mis_pipe)} total errors)")
    print("=" * 70)
    for _, r in mis_pipe.iterrows():
        b_tag = " [BORDERLINE]" if r["borderline"] else ""
        print(f"\n[Sample #{r['idx']}{b_tag}]")
        print(f"  Text: \"{r['text']}\"")
        print(f"  True Label:      {r['true_label']}")
        print(f"  Pipeline Pred:   {r['pipe_pred']} (Raw Model Pred: {r['raw_pred']})")
        print(f"  Probabilities:   RED={r['probs'][0]:.4f}, YELLOW={r['probs'][1]:.4f}, GREEN={r['probs'][2]:.4f}")
        if r["note"]:
            print(f"  Rule Action:     {r['note']}")

    # 7. Misclassified Sentences (Model Alone)
    mis_raw = res_df[~res_df["correct_raw"]]
    print("\n" + "=" * 70)
    print(f"5. MISCLASSIFIED SENTENCES IN MODEL ALONE ({len(mis_raw)} total errors)")
    print("=" * 70)
    for _, r in mis_raw.iterrows():
        b_tag = " [BORDERLINE]" if r["borderline"] else ""
        print(f"\n[Sample #{r['idx']}{b_tag}]")
        print(f"  Text: \"{r['text']}\"")
        print(f"  True Label:      {r['true_label']}")
        print(f"  Raw Model Pred:  {r['raw_pred']}")
        print(f"  Probabilities:   RED={r['probs'][0]:.4f}, YELLOW={r['probs'][1]:.4f}, GREEN={r['probs'][2]:.4f}")


if __name__ == "__main__":
    main()
