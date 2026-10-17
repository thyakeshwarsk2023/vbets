"""Train the fixed seed-42 v4 model without overwriting earlier artifacts."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import LABELS
from train_v3 import train_bow_model


def main():
    original = pd.read_csv("data/triage_dataset.csv").dropna().reset_index(drop=True)
    aug_v3 = pd.read_csv("data/augmentation_v3_new_rows.csv")
    aug_v4 = pd.read_csv("data/augmentation_v4_new_rows.csv")
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    y = original.label.map(label_to_id).to_numpy()
    train_idx, test_idx = train_test_split(np.arange(len(original)), test_size=.2,
                                           stratify=y, random_state=42)
    train = original.iloc[train_idx]
    test = original.iloc[test_idx]
    train_texts = (train.text.astype(str).tolist()
                   + aug_v3.text.astype(str).tolist()
                   + aug_v4.text.astype(str).tolist())
    train_labels = np.concatenate([
        y[train_idx],
        aug_v3.label.map(label_to_id).to_numpy(),
        aug_v4.label.map(label_to_id).to_numpy(),
    ])
    test_texts = test.text.astype(str).tolist()
    test_labels = y[test_idx]

    model, vocab, metrics = train_bow_model(train_texts, train_labels, test_texts, test_labels, seed=42)
    metrics.update({
        "n_original_train": len(train),
        "n_v3_augmentation": len(aug_v3),
        "n_v4_augmentation": len(aug_v4),
        "n_augmentation": len(aug_v3) + len(aug_v4),
        "split_seed": 42,
        "init_seed": 42,
        "training_class_counts_RED_YELLOW_GREEN": {
            label: int((train_labels == label_to_id[label]).sum()) for label in LABELS
        },
    })
    os.makedirs("model", exist_ok=True)
    model.save("model/triage_mlp_v4.keras")
    with open("model/vocab_v4.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open("model/metrics_v4.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved v4 model, vocabulary, metrics")
    print("Train/test/vocab:", len(train_texts), len(test_texts), len(vocab))
    print("Training class counts RED/YELLOW/GREEN:", metrics["training_class_counts_RED_YELLOW_GREEN"])
    print("Raw held-out RED recall / accuracy:", metrics["bow_mlp"]["RED"]["recall"], metrics["bow_mlp"]["accuracy"])
    print("Raw held-out confusion matrix:", metrics["confusion_matrix_rows_true_cols_pred_RED_YELLOW_GREEN"])


if __name__ == "__main__":
    main()
