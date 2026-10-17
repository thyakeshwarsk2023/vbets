"""Evaluation of saved V-BETS BoW MLP on independent unseen test set (data/unseen_test.csv)."""
import argparse
import json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import LABELS
from triage import GREEN_CONF_THRESHOLD, YELLOW_CONF_THRESHOLD, classify_text


def classify_pipeline(text: str, model, vocab):
    """Shared production policy from triage.py."""
    return classify_text(text, model, vocab)


def main():
    print("=" * 70)
    print("V-BETS INDEPENDENT UNSEEN TEST EVALUATION")
    print("=" * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/triage_mlp_v4.keras")
    parser.add_argument("--vocab", default="model/vocab_v4.json")
    parser.add_argument("--data", default="data/unseen_test.csv")
    args = parser.parse_args()
    model = keras.models.load_model(args.model)
    vocab = json.load(open(args.vocab))
    df = pd.read_csv(args.data)
    print(f"Loaded unseen test dataset: {len(df)} samples")
    print(f"Class distribution: {dict(df['label'].value_counts())}")
    print(f"Borderline samples: {df['borderline'].sum()} of {len(df)}\n")

    records = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        true_label = str(row["label"])
        borderline = bool(row["borderline"])

        result = classify_pipeline(text, model, vocab)
        pipe_label = result["label"]
        raw_label = result["raw_label"]
        probs = result["probabilities"]
        note = result["note"]
        records.append({
            "idx": idx + 1,
            "text": text,
            "true_label": true_label,
            "raw_pred": raw_label,
            "pipe_pred": pipe_label,
            "probs": probs,
            "note": note,
            "confidence": result["confidence"],
            "matched_rule": result["matched_rule"],
            "confidence_escalation": result["confidence_escalation"],
            "borderline": borderline,
            "correct_raw": raw_label == true_label,
            "correct_pipe": pipe_label == true_label,
        })

    res_df = pd.DataFrame(records)
    y_true = res_df["true_label"].values
    y_raw = res_df["raw_pred"].values
    y_pipe = res_df["pipe_pred"].values

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

    low_yellow = res_df[(res_df["raw_pred"] == "YELLOW") & (res_df["confidence"] < YELLOW_CONF_THRESHOLD)]
    print(f"Low-confidence YELLOW->RED escalations: {len(low_yellow)}")
    print("  True RED among them:", int((low_yellow["true_label"] == "RED").sum()))
    low_green = res_df[(res_df["raw_pred"] == "GREEN") & (res_df["confidence"] < GREEN_CONF_THRESHOLD)]
    print(f"Low-confidence GREEN->YELLOW escalations: {len(low_green)}")
    low_green_red = res_df[(res_df["raw_pred"] == "GREEN") & (res_df["true_label"] == "RED")]
    print("  Prior GREEN-should-be-RED cases caught by this YELLOW rule: 0; such cases do not enter its branch.")
    print(f"  Raw GREEN / true RED cases in this set: {len(low_green_red)}")

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
