"""Train + evaluate the triage models. Run once locally:  python train.py
Outputs: model/triage_mlp.keras, model/vocab.json, model/metrics.json, model/confusion_matrix.png
"""
import argparse, json, os
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocess import LABELS, build_vocab, bow_vector, seq_vector

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="", help="Output suffix, e.g. v2; preserves baseline artifacts")
parser.add_argument("--no-negation", action="store_true", help="Disable NEG_ features for A/B comparisons")
parser.add_argument("--split-seed", type=int, default=42, help="Train/test split seed (default preserves the original split)")
parser.add_argument("--init-seed", type=int, default=42, help="Model initialization seed (separate from the split seed)")
args = parser.parse_args()
SPLIT_SEED = args.split_seed
USE_NEGATION = not args.no_negation
suffix = f"_{args.suffix}" if args.suffix else ""
model_path = f"model/triage_mlp{suffix}.keras"
vocab_path = f"model/vocab{suffix}.json"
metrics_path = f"model/metrics{suffix}.json"
matrix_path = f"model/confusion_matrix{suffix}.png"
tf.keras.utils.set_random_seed(args.init_seed)
os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/triage_dataset.csv").dropna()
df["y"] = df["label"].map({l: i for i, l in enumerate(LABELS)})
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=SPLIT_SEED)
print(f"Train: {len(train_df)}  Test: {len(test_df)}")

# Vocabulary is built from TRAINING data only (no leakage)
vocab = build_vocab(train_df["text"], use_negation=USE_NEGATION)
json.dump(vocab, open(vocab_path, "w"))

y_train, y_test = train_df["y"].values, test_df["y"].values
cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw = {i: w for i, w in enumerate(cw)}
cb = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")]


def evaluate(name, model, Xte):
    probs = model.predict(Xte, verbose=0)
    pred = probs.argmax(1)
    rep = classification_report(y_test, pred, target_names=LABELS, output_dict=True, zero_division=0)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=LABELS, zero_division=0))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    return rep, cm


# ---- Model A: Bag-of-Words MLP (deployed) ------------------------------
Xtr = np.stack([bow_vector(t, vocab, use_negation=USE_NEGATION) for t in train_df["text"]])
Xte = np.stack([bow_vector(t, vocab, use_negation=USE_NEGATION) for t in test_df["text"]])
mlp = keras.Sequential([
    keras.layers.Input(shape=(len(vocab),)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(3, activation="softmax"),
])
mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
mlp.fit(Xtr, y_train, validation_split=0.15, epochs=150, batch_size=16,
        class_weight=cw, callbacks=cb, verbose=0)
rep_a, cm_a = evaluate("BoW MLP (deployed)", mlp, Xte)
mlp.save(model_path)

# ---- Model B: Embedding model (comparison for the report) --------------
Str = np.stack([seq_vector(t, vocab, use_negation=USE_NEGATION) for t in train_df["text"]])
Ste = np.stack([seq_vector(t, vocab, use_negation=USE_NEGATION) for t in test_df["text"]])
emb = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Embedding(len(vocab) + 1, 32, mask_zero=False),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(3, activation="softmax"),
])
emb.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
emb.fit(Str, y_train, validation_split=0.15, epochs=200, batch_size=16,
        class_weight=cw, callbacks=cb, verbose=0)
rep_b, cm_b = evaluate("Embedding model (comparison)", emb, Ste)

json.dump({"bow_mlp": rep_a, "embedding": rep_b,
           "n_train": len(train_df), "n_test": len(test_df), "vocab_size": len(vocab)},
          open(metrics_path, "w"), indent=2)

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm_a, cmap="Blues")
ax.set_xticks(range(3), LABELS); ax.set_yticks(range(3), LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("BoW MLP confusion matrix")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm_a[i, j], ha="center", va="center")
plt.tight_layout(); plt.savefig(matrix_path, dpi=150)
print(f"\nSaved {model_path}, {vocab_path}, {metrics_path}. RED recall (BoW):", round(rep_a["RED"]["recall"], 3))
