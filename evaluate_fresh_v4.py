"""One-shot final evaluation of selected v4 versus v2 on the locked fresh set."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras

from evaluate_v3 import config_predictions
from preprocess import LABELS, bow_vector


def red_ci(hits, total):
    lo = beta.ppf(.025, hits, total-hits+1) if hits else 0.0
    hi = beta.ppf(.975, hits+1, total-hits) if hits < total else 1.0
    return float(lo), float(hi)


def main():
    model_specs = {
        "selected_v4": (keras.models.load_model("model/triage_mlp_v4.keras"),
                        json.loads(Path("model/vocab_v4.json").read_text(encoding="utf-8"))),
        "baseline_v2": (keras.models.load_model("model/triage_mlp_v2.keras"),
                        json.loads(Path("model/vocab_v2.json").read_text(encoding="utf-8"))),
    }
    frame = pd.read_csv("data/unseen_test_v2.csv").reset_index(drop=True)
    texts = frame.text.astype(str).tolist()
    truth = frame.label.astype(str).tolist()
    all_results = {}
    lines = [
        "V-BETS final fresh-set evaluation (single pass)",
        "Set: data/unseen_test_v2.csv; model selected by committed reports/model_selection_rule.txt: v4; baseline: v2.",
        "Each model produced one raw probability matrix; Configs A/B/C were derived from that same matrix.",
        "Config A=raw argmax; B=general RED_FLAGS and symmetric confidence escalation with targeted fixes off; C=full pipeline with targeted fixes on.",
        "Confusion matrices are rows=true and columns=predicted in RED,YELLOW,GREEN order. RED recall CI is exact two-sided Clopper-Pearson 95%.",
        f"N={len(frame)}; rows are synthetic author-written cases.",
        "",
    ]
    for model_name, (model, vocab) in model_specs.items():
        matrix = np.stack([bow_vector(text, vocab, use_negation=True) for text in texts])
        probs = model.predict(matrix, verbose=0)
        all_results[model_name] = {}
        for config in ("A_raw_model", "B_general_rules", "C_full_policy"):
            predicted = config_predictions(config, model, vocab, texts, probs)
            cm = confusion_matrix(truth, predicted, labels=LABELS)
            red_total = int(cm[0].sum())
            red_hits = int(cm[0, 0])
            lo, hi = red_ci(red_hits, red_total)
            per_class = {label: float(cm[i,i]/cm[i].sum()) if cm[i].sum() else 0.0
                         for i,label in enumerate(LABELS)}
            acc = float(accuracy_score(truth, predicted))
            all_results[model_name][config] = {
                "confusion_matrix": cm.tolist(), "red_recall": red_hits/red_total,
                "red_ci_95": [lo,hi], "accuracy": acc,
                "per_class_recall": per_class, "predictions": predicted,
                "probabilities": probs.tolist(),
            }
            matrix_str = ";".join("["+",".join(str(int(v)) for v in row)+"]" for row in cm)
            lines += [
                f"{model_name} / {config}",
                f"Confusion matrix: {matrix_str}",
                f"RED recall: {red_hits}/{red_total} = {red_hits/red_total:.1%}; 95% CI [{lo:.1%}, {hi:.1%}]",
                f"Accuracy: {acc:.1%}",
                "Per-class recall: " + ", ".join(f"{label}={per_class[label]:.1%}" for label in LABELS),
                "Misclassified sentences (raw model probabilities RED/YELLOW/GREEN):",
            ]
            misses = [i for i,(actual,pred) in enumerate(zip(truth,predicted)) if actual != pred]
            if not misses:
                lines.append("  none")
            for i in misses:
                marker = "*** TRUE-RED MISS *** " if truth[i] == "RED" else ""
                ps = ",".join(f"{float(x):.5f}" for x in probs[i])
                lines.append(f"  {marker}row #{i+1:02d}: true={truth[i]}, predicted={predicted[i]}, p=[{ps}], text={texts[i]}")
            lines.append("")
        del model
    lines += [
        "No tuning, rule changes, or data additions were made after this evaluation.",
        "This fresh set remains small, synthetic, and author-written; one evaluation cannot establish clinical generalization.",
    ]
    report = "\n".join(lines)+"\n"
    Path("reports/v4_final_fresh_evaluation.txt").write_text(report,encoding="utf-8")
    Path("reports/v4_final_fresh_evaluation.json").write_text(json.dumps({
        "n":len(frame), "labels":LABELS, "models":all_results,
    },indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
