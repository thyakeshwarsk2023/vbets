"""Diagnose v3 Sample #10 behavior using only existing train/held-out/unseen data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evaluate_v3 import config_predictions
from preprocess import LABELS, bow_vector, preprocess
from train_v3 import train_bow_model

SPLIT_SEED = 42
INIT_SEEDS = (1, 7, 13, 21, 42, 77, 101, 123, 2026, 314)
LOCO_SEEDS = INIT_SEEDS[:5]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def measure(model, vocab, texts, truth):
    probs = model.predict(np.stack([bow_vector(t, vocab, use_negation=True) for t in texts]), verbose=0)
    raw = [LABELS[int(p.argmax())] for p in probs]
    general = config_predictions("B_general_rules", model, vocab, texts, probs)
    red_n = sum(x == "RED" for x in truth)
    return {
        "probs": probs,
        "raw_red_recall": sum(t == p == "RED" for t, p in zip(truth, raw)) / red_n,
        "general_red_recall": sum(t == p == "RED" for t, p in zip(truth, general)) / red_n,
        "general_accuracy": accuracy_score(truth, general),
    }


def main():
    original = pd.read_csv("data/triage_dataset.csv").dropna().reset_index(drop=True)
    aug = pd.read_csv("data/augmentation_v3_new_rows.csv")
    unseen = pd.read_csv("data/unseen_test.csv").reset_index(drop=True)
    y = original.label.map(LABEL_TO_ID).to_numpy()
    train_idx, held_idx = train_test_split(np.arange(len(original)), test_size=.2,
                                           stratify=y, random_state=SPLIT_SEED)
    train = original.iloc[train_idx]
    held = original.iloc[held_idx]
    held_text = held.text.astype(str).tolist()
    held_truth = held.label.astype(str).tolist()
    unseen_text = unseen.text.astype(str).tolist()
    unseen_truth = unseen.label.astype(str).tolist()
    sample10 = unseen_text[9]

    lines = [
        "V-BETS v4 diagnostic: Sample #10 regression",
        "Allowed diagnostic data: original seed-42 held-out rows and contaminated data/unseen_test.csv only.",
        f"Split: {len(train)} original training rows / {len(held)} held-out rows; initialization seeds={list(INIT_SEEDS)}.",
        f"Sample #10: {sample10}",
        "",
        "Initialization-seed runs (each row is a separate retraining; probabilities RED/YELLOW/GREEN):",
        "Model | Init seed | Sample #10 raw probabilities | raw class | general-rule class | held-out RED recall raw/general | unseen RED recall raw/general",
        "---|---:|---|---|---|---|---",
    ]
    init_results = {}
    for version in ("v2", "v3"):
        init_results[version] = []
        for seed in INIT_SEEDS:
            train_text = train.text.astype(str).tolist()
            train_y = y[train_idx]
            if version == "v3":
                train_text += aug.text.astype(str).tolist()
                train_y = np.concatenate([train_y, aug.label.map(LABEL_TO_ID).to_numpy()])
            model, vocab, _ = train_bow_model(train_text, train_y, held_text, y[held_idx], seed=seed)
            held_m = measure(model, vocab, held_text, held_truth)
            unseen_m = measure(model, vocab, unseen_text, unseen_truth)
            p10 = unseen_m["probs"][9]
            raw_class = LABELS[int(p10.argmax())]
            general_class = config_predictions("B_general_rules", model, vocab, [sample10], p10[None, :])[0]
            init_results[version].append({
                "seed": seed, "p_red": float(p10[0]), "raw_class": raw_class,
                "general_class": general_class, "held_raw": held_m["raw_red_recall"],
                "held_general": held_m["general_red_recall"],
                "unseen_raw": unseen_m["raw_red_recall"],
                "unseen_general": unseen_m["general_red_recall"],
            })
            lines.append(f"{version} | {seed} | [{p10[0]:.4f},{p10[1]:.4f},{p10[2]:.4f}] | {raw_class} | {general_class} | {held_m['raw_red_recall']:.3f}/{held_m['general_red_recall']:.3f} | {unseen_m['raw_red_recall']:.3f}/{unseen_m['general_red_recall']:.3f}")
            print(f"init diagnosis finished: {version} seed {seed}", flush=True)
            del model
    for version in ("v2", "v3"):
        recs = init_results[version]
        raw_ok = sum(r["raw_class"] == "RED" for r in recs)
        general_ok = sum(r["general_class"] == "RED" for r in recs)
        lines.append(f"{version} Sample #10 RED correctness: raw {raw_ok}/{len(recs)}; with general rules {general_ok}/{len(recs)}.")

    lines += ["", "Leave-one-category-out v3 runs (five initialization seeds each):",
              "Omitted category | Init seed | Sample #10 raw/general | held-out RED recall raw/general | held-out accuracy general | unseen RED recall raw/general",
              "---|---:|---|---|---:|---"]
    categories = {}
    for category in ("C1_green_admin", "C2_green_negation", "C3_red_lay"):
        categories[category] = []
        subset = aug[aug.category != category]
        for seed in LOCO_SEEDS:
            train_text = train.text.astype(str).tolist() + subset.text.astype(str).tolist()
            train_y = np.concatenate([y[train_idx], subset.label.map(LABEL_TO_ID).to_numpy()])
            model, vocab, _ = train_bow_model(train_text, train_y, held_text, y[held_idx], seed=seed)
            held_m = measure(model, vocab, held_text, held_truth)
            unseen_m = measure(model, vocab, unseen_text, unseen_truth)
            p10 = unseen_m["probs"][9]
            raw_cls = LABELS[int(p10.argmax())]
            gen_cls = config_predictions("B_general_rules", model, vocab, [sample10], p10[None, :])[0]
            gen_acc = accuracy_score(held_truth, config_predictions("B_general_rules", model, vocab, held_text, held_m["probs"]))
            categories[category].append({"seed": seed, "raw": raw_cls, "general": gen_cls,
                "held_red": held_m["general_red_recall"], "held_acc": gen_acc,
                "unseen_red": unseen_m["general_red_recall"]})
            lines.append(f"{category} | {seed} | {raw_cls}/{gen_cls} | {held_m['raw_red_recall']:.3f}/{held_m['general_red_recall']:.3f} | {gen_acc:.3f} | {unseen_m['raw_red_recall']:.3f}/{unseen_m['general_red_recall']:.3f}")
            print(f"category ablation finished: without {category}, seed {seed}", flush=True)
            del model
    for category, recs in categories.items():
        lines.append(f"Without {category}: Sample #10 correct raw {sum(r['raw']=='RED' for r in recs)}/5, general {sum(r['general']=='RED' for r in recs)}/5; mean held-out general RED recall {np.mean([r['held_red'] for r in recs]):.3f}; mean held-out accuracy {np.mean([r['held_acc'] for r in recs]):.3f}; mean unseen general RED recall {np.mean([r['unseen_red'] for r in recs]):.3f}.")

    # Five nearest neighbors from the full v3 training partition (original 296 + all 36 additions).
    v3_train_text = train.text.astype(str).tolist() + aug.text.astype(str).tolist()
    v3_train_labels = train.label.astype(str).tolist() + aug.label.astype(str).tolist()
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(v3_train_text + [sample10])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    nearest = np.argsort(sims)[::-1][:5]
    lines += ["", "Nearest v3 training neighbors to Sample #10 (TF-IDF cosine):", "Rank | Cosine | Label | Text", "---:|---:|---|---"]
    for rank, ix in enumerate(nearest, 1):
        lines.append(f"{rank} | {sims[ix]:.4f} | {v3_train_labels[ix]} | {v3_train_text[ix]}")
    vocab = json.loads(Path("model/vocab_v3.json").read_text(encoding="utf-8"))
    sample_features = preprocess(sample10, use_negation=True)
    present = [token for token in sample_features if token in vocab]
    lines.append("Sample #10 preprocessed tokens present in the v3 vocabulary: " + ", ".join(present))
    lines.append("Tokens absent from the v3 vocabulary: " + ", ".join(token for token in sample_features if token not in vocab))

    # Interpret only what the observed fractions support.
    v3_ok = sum(r["raw_class"] == "RED" for r in init_results["v3"])
    v2_ok = sum(r["raw_class"] == "RED" for r in init_results["v2"])
    loco = {c: sum(r["raw"] == "RED" for r in rs) / 5 for c, rs in categories.items()}
    lines += ["", "Diagnosis:"]
    lines.append(f"Sample #10 was raw-RED in {v2_ok}/10 v2 initializations and {v3_ok}/10 v3 initializations.")
    lines.append("Leave-one-category-out raw correctness fractions: " + "; ".join(f"without {c} {v:.0%}" for c, v in loco.items()) + ".")
    lines.append("Sample #10 flips across initialization seeds in both v2 and v3, so initialization variability is present. However, v3 was correct in 2/10 runs versus v2 in 6/10, an augmentation-associated reduction in recovery frequency across these tested runs.")
    lines.append("On the comparable first five seeds, full v3 was correct in 1/5; removing C1 was also 1/5, while removing C2 or C3 was 0/5. No leave-one-category-out result improves on full v3, so no specific category is implicated by this ablation. The experiment does not identify a causal mechanism.")
    lines.append("The held-out and original unseen metrics are diagnostic only; the original unseen set is contaminated and is not a generalization estimate.")
    Path("reports/v4_diagnosis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote reports/v4_diagnosis.txt", flush=True)


if __name__ == "__main__":
    main()
