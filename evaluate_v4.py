"""Five-init and four-split comparisons for v2/v3/v4 (original test sets only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import beta

from evaluate_v3 import config_predictions
from preprocess import LABELS
from train_v3 import train_bow_model

INIT_SEEDS = (1, 7, 13, 21, 42)
SPLIT_SEEDS = (1, 7, 42, 123)
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def calculate(truth, predictions):
    cm = confusion_matrix(truth, predictions, labels=LABELS)
    nred, nyellow = int(cm[0].sum()), int(cm[1].sum())
    red_hit = int(cm[0, 0])
    low = beta.ppf(.025, red_hit, nred-red_hit+1) if red_hit else 0.0
    high = beta.ppf(.975, red_hit+1, nred-red_hit) if red_hit < nred else 1.0
    return {
        "confusion_matrix": cm.tolist(), "red_hits": red_hit, "red_total": nred,
        "red_recall": red_hit/nred, "red_ci_low": float(low), "red_ci_high": float(high),
        "yellow_hits": int(cm[1,1]), "yellow_total": nyellow,
        "yellow_recall": float(cm[1,1]/nyellow) if nyellow else 0.0,
        "accuracy": float(accuracy_score(truth,predictions)),
        "undertriage_rows": [i+1 for i,(t,p) in enumerate(zip(truth,predictions)) if t=="RED" and p!="RED"],
    }


def prep_training(version, original, aug3, aug4, train_idx, y):
    texts = original.iloc[train_idx].text.astype(str).tolist()
    labels = y[train_idx]
    if version in ("v3", "v4"):
        texts += aug3.text.astype(str).tolist()
        labels = np.concatenate([labels, aug3.label.map(LABEL_TO_ID).to_numpy()])
    if version == "v4":
        texts += aug4.text.astype(str).tolist()
        labels = np.concatenate([labels, aug4.label.map(LABEL_TO_ID).to_numpy()])
    return texts, labels


def main():
    original = pd.read_csv("data/triage_dataset.csv").dropna().reset_index(drop=True)
    aug3 = pd.read_csv("data/augmentation_v3_new_rows.csv")
    aug4 = pd.read_csv("data/augmentation_v4_new_rows.csv")
    unseen = pd.read_csv("data/unseen_test.csv").reset_index(drop=True)
    y = original.label.map(LABEL_TO_ID).to_numpy()
    train42, held42 = train_test_split(np.arange(len(original)), test_size=.2,
                                       stratify=y, random_state=42)
    held = original.iloc[held42].reset_index(drop=True)
    datasets = {"heldout_seed42": held, "unseen_contaminated": unseen}
    records = []
    seed42_sample_records = {}
    models = ("v2", "v3", "v4")
    for version in models:
        for init_seed in INIT_SEEDS:
            print(f"5-init run: {version}, init seed {init_seed}", flush=True)
            train_texts, train_y = prep_training(version, original, aug3, aug4, train42, y)
            test_texts = held.text.astype(str).tolist()
            model, vocab, _ = train_bow_model(train_texts, train_y, test_texts, y[held42], seed=init_seed)
            for ds_name, frame in datasets.items():
                texts = frame.text.astype(str).tolist()
                truth = frame.label.astype(str).tolist()
                probs = model.predict(np.stack([__import__('preprocess').bow_vector(t,vocab,use_negation=True) for t in texts]), verbose=0)
                for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
                    pred = config_predictions(config, model, vocab, texts, probs)
                    result = calculate(truth, pred)
                    rec = {"model":version,"init_seed":init_seed,"dataset":ds_name,"config":config,
                           **result,"predictions":pred,"probabilities":probs.tolist()}
                    records.append(rec)
                    if ds_name == "unseen_contaminated":
                        for sample_no in (10,32,45):
                            sr = seed42_sample_records.setdefault((version,init_seed,sample_no),{})
                            sr[config] = pred[sample_no-1]
                            sr["truth"] = truth[sample_no-1]
                            sr["probabilities"] = [float(x) for x in probs[sample_no-1]]
            del model

    split_records = []
    for split_seed in SPLIT_SEEDS:
        tr_idx, te_idx = train_test_split(np.arange(len(original)), test_size=.2,
                                           stratify=y, random_state=split_seed)
        frame = original.iloc[te_idx].reset_index(drop=True)
        texts = frame.text.astype(str).tolist()
        truth = frame.label.astype(str).tolist()
        for version in models:
            print(f"split-seed Config B: {version}, split {split_seed}", flush=True)
            train_texts, train_y = prep_training(version, original, aug3, aug4, tr_idx, y)
            model, vocab, _ = train_bow_model(train_texts, train_y, texts, y[te_idx], seed=42)
            probs = model.predict(np.stack([__import__('preprocess').bow_vector(t,vocab,use_negation=True) for t in texts]), verbose=0)
            pred = config_predictions("B_general_rules", model, vocab, texts, probs)
            split_records.append({"model":version,"split_seed":split_seed,**calculate(truth,pred)})
            del model

    lines = [
        "V-BETS v4 evaluation: original held-out and contaminated unseen sets only.",
        "The locked data/unseen_test_v2.csv was not read or evaluated in this step.",
        "Configs: A=raw argmax; B=general RED_FLAGS + symmetric confidence escalation, targeted fixes off; C=full pipeline, targeted fixes on.",
        "Matrices use rows=true and columns=predicted in RED,YELLOW,GREEN order. RED recall uses exact two-sided Clopper-Pearson 95% CI.",
        "No no-flag or category-omission model: v4 similarity audit flagged no rows and Step 2 did not implicate a category.",
        "",
        "Five initialization seeds per model, fixed split seed 42 (each row is raw output for one evaluation):",
        "Model | Init | Dataset | Config | Confusion matrix | RED recall (95% CI) | YELLOW recall | Accuracy | RED under-triage rows",
        "---|---:|---|---|---|---|---|---:|---",
    ]
    for r in records:
        matrix = ";".join("["+",".join(str(v) for v in row)+"]" for row in r["confusion_matrix"])
        lines.append(f"{r['model']} | {r['init_seed']} | {r['dataset']} | {r['config']} | {matrix} | {r['red_hits']}/{r['red_total']} ({r['red_recall']:.1%}; [{r['red_ci_low']:.1%},{r['red_ci_high']:.1%}]) | {r['yellow_recall']:.1%} | {r['accuracy']:.1%} | {r['undertriage_rows'] or 'none'}")

    lines += ["", "Five-init mean and min-max range by model/config/dataset (RED recall and accuracy):",
              "Model | Dataset | Config | mean RED recall (range) | mean accuracy (range)",
              "---|---|---|---|---"]
    for version in models:
        for ds_name in datasets:
            for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
                group=[r for r in records if r["model"]==version and r["dataset"]==ds_name and r["config"]==config]
                reds=[r["red_recall"] for r in group]; accs=[r["accuracy"] for r in group]
                lines.append(f"{version} | {ds_name} | {config} | {np.mean(reds):.1%} ({min(reds):.1%}-{max(reds):.1%}) | {np.mean(accs):.1%} ({min(accs):.1%}-{max(accs):.1%})")

    lines += ["", "Samples #10/#32/#45 on the contaminated original unseen set (raw probabilities RED/YELLOW/GREEN; labels under A/B/C):",
              "Model | Init | Sample | Truth | Probabilities | A | B | C", "---|---:|---:|---|---|---|---|---"]
    for version in models:
        for init_seed in INIT_SEEDS:
            for sample_no in (10,32,45):
                r=seed42_sample_records[(version,init_seed,sample_no)]
                ps=",".join(f"{p:.4f}" for p in r["probabilities"])
                lines.append(f"{version} | {init_seed} | {sample_no} | {r['truth']} | [{ps}] | {r['A_raw_model']} | {r['B_general_rules']} | {r['C_full_policy']}")

    lines += ["", "Config B split-seed runs (init seed fixed at 42; new rows included in each training partition only):",
              "Model | Split seed | Confusion matrix | RED recall | YELLOW recall | Accuracy", "---|---:|---|---:|---:|---:"]
    for r in split_records:
        matrix=";".join("["+",".join(str(v) for v in row)+"]" for row in r["confusion_matrix"])
        lines.append(f"{r['model']} | {r['split_seed']} | {matrix} | {r['red_recall']:.1%} | {r['yellow_recall']:.1%} | {r['accuracy']:.1%}")
    lines += ["", "Config B split-seed mean and min-max range:", "Model | mean RED recall (range) | mean accuracy (range) | mean YELLOW recall (range)", "---|---|---|---"]
    for version in models:
        group=[r for r in split_records if r["model"]==version]
        rr=[r["red_recall"] for r in group]; aa=[r["accuracy"] for r in group]; yy=[r["yellow_recall"] for r in group]
        lines.append(f"{version} | {np.mean(rr):.1%} ({min(rr):.1%}-{max(rr):.1%}) | {np.mean(aa):.1%} ({min(aa):.1%}-{max(aa):.1%}) | {np.mean(yy):.1%} ({min(yy):.1%}-{max(yy):.1%})")

    Path("reports/v4_evaluation.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    Path("reports/v4_evaluation_results.json").write_text(json.dumps({"init_seeds":INIT_SEEDS,"split_seeds":SPLIT_SEEDS,"init_runs":records,"split_runs":split_records},indent=2),encoding="utf-8")
    print("Wrote reports/v4_evaluation.txt and reports/v4_evaluation_results.json",flush=True)


if __name__ == "__main__":
    main()
