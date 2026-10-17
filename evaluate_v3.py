"""Compare v2/v3 under raw, general-policy and full-policy configurations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

import triage
from preprocess import LABELS, bow_vector, normalize, preprocess
from safety_rules import matched_red_flag
from train_v3 import train_bow_model

SEEDS = (1, 7, 42, 123)
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def cp_interval(successes, total):
    low = beta.ppf(0.025, successes, total - successes + 1) if successes else 0.0
    high = beta.ppf(0.975, successes + 1, total - successes) if successes < total else 1.0
    return low, high


def probabilities(model, vocab, texts):
    x = np.stack([bow_vector(t, vocab, use_negation=True) for t in texts])
    return model.predict(x, verbose=0)


def config_predictions(config, model, vocab, texts, probs=None):
    if config == "A_raw_model":
        if probs is None:
            probs = probabilities(model, vocab, texts)
        return [LABELS[int(p.argmax())] for p in probs]
    targeted = config == "C_full_policy"
    outputs = []
    for text, p in zip(texts, probs if probs is not None else probabilities(model, vocab, texts)):
        raw_label = LABELS[int(p.argmax())]
        confidence = float(p.max())
        norm = normalize(text)
        rule = matched_red_flag(text)
        if rule:
            label = "RED"
        elif targeted and any(phrase in norm for phrase in (
            "prescription refill", "medication refilled", "medicine refill",
            "repeat prescription for my regular",
        )):
            label = "GREEN"
        elif targeted and (
            "lower back" in norm and "gardening" in norm
            and bool({"stiff", "sore", "achy"} & set(preprocess(text)))
            and {"NEG_pain", "NEG_numbness", "NEG_weakness"}.issubset(set(preprocess(text)))
        ):
            label = "GREEN"
        elif raw_label == "GREEN" and confidence < triage.GREEN_CONF_THRESHOLD:
            label = "YELLOW"
        elif raw_label == "YELLOW" and confidence < triage.YELLOW_CONF_THRESHOLD:
            label = "RED"
        else:
            label = raw_label
        outputs.append(label)
    return outputs


def metrics_for(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    red_total = sum(t == "RED" for t in truth)
    red_hit = int(matrix[0, 0])
    lo, hi = cp_interval(red_hit, red_total)
    return {
        "matrix": matrix.tolist(), "red_hit": red_hit, "red_total": red_total,
        "red_recall": red_hit / red_total, "ci_low": lo, "ci_high": hi,
        "accuracy": accuracy_score(truth, predicted),
        "undertriage": [i + 1 for i, (t, p) in enumerate(zip(truth, predicted)) if t == "RED" and p != "RED"],
    }


def main():
    print("Loading datasets and v2/v3 artifacts...", flush=True)
    original = pd.read_csv("data/triage_dataset.csv").dropna().reset_index(drop=True)
    augmentation = pd.read_csv("data/augmentation_v3_new_rows.csv")
    unseen = pd.read_csv("data/unseen_test.csv").reset_index(drop=True)
    y = original.label.map(LABEL_TO_ID).to_numpy()
    _, test_idx = train_test_split(np.arange(len(original)), test_size=.2, stratify=y, random_state=42)
    heldout = original.iloc[test_idx].reset_index(drop=True)
    datasets = {"heldout_seed42": heldout, "unseen_test": unseen}

    model_data = {}
    for version in ("v2", "v3"):
        model_data[version] = (
            keras.models.load_model(f"model/triage_mlp_{version}.keras"),
            json.loads(Path(f"model/vocab_{version}.json").read_text(encoding="utf-8")),
        )
        print(f"Loaded {version}", flush=True)

    results = {}
    for version, (model, vocab) in model_data.items():
        print(f"Evaluating fixed-split {version}", flush=True)
        for dataset_name, frame in datasets.items():
            texts = frame.text.astype(str).tolist()
            truth = frame.label.astype(str).tolist()
            raw_probs = probabilities(model, vocab, texts)
            for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
                preds = config_predictions(config, model, vocab, texts, raw_probs)
                results[(version, config, dataset_name)] = {
                    **metrics_for(truth, preds), "predictions": preds,
                    "raw_probs": raw_probs, "truth": truth,
                }
        print(f"Finished fixed-split {version}", flush=True)

    # Config B seed robustness: same architecture, training recipe, and added rows.
    seed_results = {}
    for seed in SEEDS:
        print(f"Training seed robustness seed={seed}", flush=True)
        train_idx, test_idx_seed = train_test_split(
            np.arange(len(original)), test_size=.2, stratify=y, random_state=seed)
        train = original.iloc[train_idx]
        test = original.iloc[test_idx_seed]
        test_texts = test.text.astype(str).tolist()
        test_truth = test.label.astype(str).tolist()
        for version in ("v2", "v3"):
            train_texts = train.text.astype(str).tolist()
            train_labels = y[train_idx]
            if version == "v3":
                train_texts += augmentation.text.astype(str).tolist()
                train_labels = np.concatenate([train_labels, augmentation.label.map(LABEL_TO_ID).to_numpy()])
            model, vocab, _ = train_bow_model(train_texts, train_labels, test_texts,
                                               y[test_idx_seed], seed=seed)
            pred = config_predictions("B_general_rules", model, vocab, test_texts)
            seed_results[(seed, version)] = metrics_for(test_truth, pred)
            del model
            print(f"Finished seed={seed} {version}", flush=True)

    # Format compact, auditable comparison artifact.
    lines = [
        "V-BETS v3 augmentation evaluation",
        "Labels/order: RED, YELLOW, GREEN. Confusion matrices use rows=true and columns=predicted.",
        "Config A: raw model argmax. Config B: general RED_FLAGS + symmetric confidence escalation; targeted fixes disabled.",
        "Config C: full policy with targeted fixes enabled. RED recall CIs are two-sided exact Clopper-Pearson 95% intervals.",
        "No v3_noflag model: similarity audit flagged zero rows, so the requested ablation was skipped.",
        "",
        "Comparison table (matrix cells are compact row arrays [RED,YELLOW,GREEN]):",
        "Model | Config | Dataset | Confusion matrix | RED recall (95% exact CI) | Accuracy | Under-triage RED row numbers",
        "---|---|---|---|---|---:|---",
    ]
    for version in ("v2", "v3"):
        for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
            for dataset_name in datasets:
                r = results[(version, config, dataset_name)]
                matrix = ";".join("[" + ",".join(map(str, row)) + "]" for row in r["matrix"])
                lines.append(f"{version} | {config} | {dataset_name} | {matrix} | {r['red_hit']}/{r['red_total']} ({r['red_recall']:.1%}; [{r['ci_low']:.1%}, {r['ci_high']:.1%}]) | {r['accuracy']:.1%} | {r['undertriage'] or 'none'}")

    lines += ["", "Unseen Samples #32 and #45 (each config prediction; probabilities are raw model probabilities RED/YELLOW/GREEN):"]
    for version in ("v2", "v3"):
        for number in (32, 45):
            row = unseen.iloc[number - 1]
            probabilities_here = results[(version, "A_raw_model", "unseen_test")]["raw_probs"][number - 1]
            lines.append(f"{version} Sample #{number}: true={row.label}; raw probabilities=[{', '.join(f'{p:.4f}' for p in probabilities_here)}]")
            for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
                pred = results[(version, config, "unseen_test")]["predictions"][number - 1]
                lines.append(f"  {config}: {pred}")
    lines.append(f"  Sample #32 text: {unseen.iloc[31].text}")
    lines.append(f"  Sample #45 text: {unseen.iloc[44].text}")

    lines += ["", "RED under-triage details (row numbers are one-based within the named evaluation file):"]
    for version in ("v2", "v3"):
        for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
            for dataset_name, frame in datasets.items():
                r = results[(version, config, dataset_name)]
                for row_no in r["undertriage"]:
                    lines.append(f"{version}/{config}/{dataset_name} row #{row_no}: predicted {r['predictions'][row_no-1]}, text={frame.iloc[row_no-1].text}")
    for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
        for dataset_name in datasets:
            old = set(results[("v2", config, dataset_name)]["undertriage"])
            new = set(results[("v3", config, dataset_name)]["undertriage"])
            lines.append(f"New v3 under-triage vs v2 for {config}/{dataset_name}: {sorted(new-old) or 'none'}; resolved vs v2: {sorted(old-new) or 'none'}.")
    unseen_probs10_v2 = results[("v2", "A_raw_model", "unseen_test")]["raw_probs"][9]
    unseen_probs10_v3 = results[("v3", "A_raw_model", "unseen_test")]["raw_probs"][9]
    lines.append(f"Unseen Sample #10 raw probabilities RED/YELLOW/GREEN: v2=[{', '.join(f'{p:.4f}' for p in unseen_probs10_v2)}], v3=[{', '.join(f'{p:.4f}' for p in unseen_probs10_v3)}].")

    lines += ["", "Seed robustness (Config B, independently retrained per seed):", "Seed | v2 RED recall | v2 accuracy | v3 RED recall | v3 accuracy"]
    for seed in SEEDS:
        a, b = seed_results[(seed, "v2")], seed_results[(seed, "v3")]
        lines.append(f"{seed} | {a['red_hit']}/{a['red_total']} ({a['red_recall']:.1%}) | {a['accuracy']:.1%} | {b['red_hit']}/{b['red_total']} ({b['red_recall']:.1%}) | {b['accuracy']:.1%}")
    for version in ("v2", "v3"):
        rr = [seed_results[(seed, version)]["red_recall"] for seed in SEEDS]
        aa = [seed_results[(seed, version)]["accuracy"] for seed in SEEDS]
        lines.append(f"{version} mean across seeds: RED recall {np.mean(rr):.1%}, range {min(rr):.1%}-{max(rr):.1%}; accuracy {np.mean(aa):.1%}, range {min(aa):.1%}-{max(aa):.1%}.")

    lines += ["", "Honest conclusions:"]
    for number in (32, 45):
        ix = number - 1
        for version in ("v2", "v3"):
            a = results[(version, "A_raw_model", "unseen_test")]["predictions"][ix]
            b = results[(version, "B_general_rules", "unseen_test")]["predictions"][ix]
            lines.append(f"Sample #{number} without targeted rules ({version}): Config A={a}; Config B={b}; no flagged-row ablation exists because no rows were flagged.")
    lines.append("Neither v3 Config A nor Config B classifies unseen Sample #32 (GREEN refill) or Sample #45 (GREEN gardening complaint) correctly; both remain RED/YELLOW respectively, and there is no flagged-row ablation to compare.")
    for version in ("v2", "v3"):
        for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
            for ds in datasets:
                r = results[(version, config, ds)]
                lines.append(f"{version}/{config}/{ds}: RED recall {r['red_hit']}/{r['red_total']} and {len(r['undertriage'])} RED under-triage cases.")
    lines.append("The v3 unseen evaluation has one new RED under-triage versus v2: Sample #10, a complaint of coughing up dark red clotted blood with lightheadedness and breathlessness. This regression is associated with the augmented training run; this experiment does not isolate which individual added sentence caused it. The v3 held-out Config B/C resolve the two v2 misses, but raw v3 has one different held-out miss.")
    lines.append("Seed results should be interpreted as split/training variability on the same small synthetic corpus; compare the per-seed paired differences above rather than treating the mean as a generalization estimate.")
    lines.append("data/unseen_test.csv informed prior rule and data decisions and is no longer a clean holdout. Generalization claims require a fresh untouched test set.")
    if Path("data/unseen_test_v2.csv").exists():
        lines.append("data/unseen_test_v2.csv exists; it requires a separately reported evaluation before making fresh-holdout claims.")
    else:
        lines.append("data/unseen_test_v2.csv does not exist.")

    Path("reports/v3_augmentation_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
