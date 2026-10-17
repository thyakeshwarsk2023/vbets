"""Apply the user-specified Step 6 selection rule to committed Step 5 results."""
import json
from pathlib import Path

import pandas as pd


def main():
    results = json.loads(Path("reports/v4_evaluation_results.json").read_text(encoding="utf-8"))
    unseen = pd.read_csv("data/unseen_test.csv").reset_index(drop=True)
    true_red = {i+1 for i, label in enumerate(unseen.label.astype(str)) if label == "RED"}
    split = results["split_runs"]
    init = [r for r in results["init_runs"] if r["dataset"] == "unseen_contaminated" and r["config"] == "B_general_rules"]
    candidates = ("v2", "v3", "v4")
    stats = {}
    for model in candidates:
        per_split = [r["red_recall"] for r in split if r["model"] == model]
        per_init = [r["predictions"] for r in init if r["model"] == model]
        miss_counts = {
            row: sum(preds[row-1] != "RED" for preds in per_init)
            for row in sorted(true_red)
        }
        total_misses = sum(miss_counts.values())
        stable = {row: n for row, n in miss_counts.items() if n >= 3}
        stats[model] = {
            "split_mean": sum(per_split)/len(per_split),
            "total_init_red_misses": total_misses,
            "stable_misses": stable,
            "miss_counts": miss_counts,
        }

    eligible = [m for m in candidates if not stats[m]["stable_misses"]]
    highest = max(stats[m]["split_mean"] for m in eligible)
    within_one = [m for m in eligible if highest - stats[m]["split_mean"] <= 0.01]
    # The specified tie breaks: fewer total RED misses across init runs, then v2.
    min_misses = min(stats[m]["total_init_red_misses"] for m in within_one)
    tied = [m for m in within_one if stats[m]["total_init_red_misses"] == min_misses]
    selected = "v2" if "v2" in tied else tied[0]

    lines = [
        "Pre-registered model-selection rule and application",
        "Rule (specified before any fresh-set evaluation): candidates v2/v3/v4 and any Step-4 variants; exclude any candidate with at least one true-RED unseen row predicted non-RED in >=60% of five initialization runs; among remaining candidates choose highest mean Config B RED recall across split seeds; candidates within 1 percentage point of the best are tied and the one with fewer Config B RED misses across five init seeds on the contaminated original unseen set is preferred; if still tied, choose v2.",
        "Only the committed Step 5 metrics and data/unseen_test.csv were used below. data/unseen_test_v2.csv was not read, inspected, or evaluated.",
        "No v4 no-flag or category-omission variant exists: the similarity audit flagged zero rows and the diagnostic did not implicate a category.",
        "",
        "Candidate | mean Config B RED recall over split seeds | Config B RED misses across 5 init runs on contaminated unseen | stable missed true-RED rows (count/5) | eligible",
        "---|---:|---:|---|---",
    ]
    for m in candidates:
        s=stats[m]
        stable="; ".join(f"#{row} ({n}/5)" for row,n in s["stable_misses"].items()) or "none"
        lines.append(f"{m} | {s['split_mean']:.3%} | {s['total_init_red_misses']} | {stable} | {'yes' if m in eligible else 'no'}")
    lines += ["", "Within-one-percentage-point eligible set: " + ", ".join(within_one),
              "Applied miss-count tie break: " + ", ".join(f"{m}={stats[m]['total_init_red_misses']}" for m in within_one),
              f"Selected candidate: {selected}.",
              "Variants not trained: none. The final fresh-set evaluation must compare this selected model with v2 only, exactly once, under Configs A/B/C."]
    Path("reports/model_selection_rule.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
