"""Controlled seed-42 training for the reviewed v3 augmentation experiment."""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from preprocess import LABELS, bow_vector, build_vocab

SEED = 42


def train_bow_model(train_texts, train_labels, test_texts, test_labels, seed=SEED):
    """Fit the same BoW MLP, class weights and early stopping used by train.py."""
    tf.keras.utils.set_random_seed(seed)
    vocab = build_vocab(train_texts, use_negation=True)
    x_train = np.stack([bow_vector(t, vocab, use_negation=True) for t in train_texts])
    x_test = np.stack([bow_vector(t, vocab, use_negation=True) for t in test_texts])
    cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weight = {i: w for i, w in enumerate(cw)}
    callbacks = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")]
    model = keras.Sequential([
        keras.layers.Input(shape=(len(vocab),)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, train_labels, validation_split=0.15, epochs=150, batch_size=16,
              class_weight=class_weight, callbacks=callbacks, verbose=0)
    probs = model.predict(x_test, verbose=0)
    pred = probs.argmax(1)
    report = classification_report(test_labels, pred, labels=range(3), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    metrics = {
        "seed": seed,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "n_augmentation": 36,
        "vocab_size": len(vocab),
        "use_negation": True,
        "bow_mlp": report,
        "confusion_matrix_rows_true_cols_pred_RED_YELLOW_GREEN": confusion_matrix(
            test_labels, pred, labels=range(3)).tolist(),
    }
    return model, vocab, metrics


def main():
    original = pd.read_csv("data/triage_dataset.csv").dropna().reset_index(drop=True)
    augmentation = pd.read_csv("data/augmentation_v3_new_rows.csv")
    labels = {label: i for i, label in enumerate(LABELS)}
    y = original.label.map(labels).to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(original)), test_size=0.2, stratify=y, random_state=SEED)
    train = original.iloc[train_idx]
    test = original.iloc[test_idx]
    train_texts = train.text.astype(str).tolist() + augmentation.text.astype(str).tolist()
    train_labels = np.concatenate([y[train_idx], augmentation.label.map(labels).to_numpy()])
    test_texts = test.text.astype(str).tolist()
    test_labels = y[test_idx]

    model, vocab, metrics = train_bow_model(train_texts, train_labels, test_texts, test_labels)
    os.makedirs("model", exist_ok=True)
    model.save("model/triage_mlp_v3.keras")
    with open("model/vocab_v3.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with open("model/metrics_v3.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model/triage_mlp_v3.keras, model/vocab_v3.json, model/metrics_v3.json")
    print("Train/test:", len(train_texts), len(test_texts), "vocab:", len(vocab))
    print("Raw held-out RED recall:", metrics["bow_mlp"]["RED"]["recall"])
    print("Raw held-out accuracy:", metrics["bow_mlp"]["accuracy"])
    print("Confusion matrix:", metrics["confusion_matrix_rows_true_cols_pred_RED_YELLOW_GREEN"])


if __name__ == "__main__":
    main()
