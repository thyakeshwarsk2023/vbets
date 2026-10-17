"""Validate the v4 additions, build the training corpus, and audit similarity."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import normalize


def token_set(text):
    return set(re.findall(r"[a-z]+", normalize(str(text))))


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def main():
    new = pd.read_csv("data/augmentation_v4_new_rows.csv")
    assert list(new.columns) == ["text", "label", "category"] and len(new) == 24
    assert not new.text.astype(str).str.contains(",", regex=False).any()
    assert set(new.category) == {"C4_red_bleeding", "C5_yellow"}
    assert new.text.map(lambda x: str(x).strip().casefold()).nunique() == len(new)

    new_norm = {str(x).strip().casefold() for x in new.text}
    prior_norm = set()
    for path in Path("data").glob("*.csv"):
        if path.name in {"augmentation_v4_new_rows.csv", "triage_dataset_v4.csv"}:
            continue
        frame = pd.read_csv(path)
        if "text" in frame:
            prior_norm.update(str(x).strip().casefold() for x in frame.text)
    duplicates = sorted(new_norm & prior_norm)
    if duplicates:
        raise ValueError(f"Exact duplicate v4 rows: {duplicates}")

    base = pd.read_csv("data/triage_dataset_v3.csv")
    combined = pd.concat([base, new[["text", "label"]]], ignore_index=True)
    combined.to_csv("data/triage_dataset_v4.csv", index=False)

    unseen = pd.read_csv("data/unseen_test.csv").reset_index(drop=True)
    new_texts = new.text.astype(str).tolist()
    unseen_texts = unseen.text.astype(str).tolist()
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(unseen_texts + new_texts)
    unseen_matrix = matrix[:len(unseen_texts)]
    new_matrix = matrix[len(unseen_texts):]
    cos = cosine_similarity(new_matrix, unseen_matrix)
    unseen_tokens = [token_set(t) for t in unseen_texts]
    new_tokens = [token_set(t) for t in new_texts]
    lines = [
        "V-BETS v4 data validation and similarity audit",
        f"New row count: {len(new)}; exact duplicates against other data CSVs (including locked fresh test): {len(duplicates)}.",
        f"Text rows containing commas: {int(new.text.astype(str).str.contains(',', regex=False).sum())}.",
        f"v3 corpus rows: {len(base)}; v4 corpus rows: {len(combined)}.",
        "Jaccard uses unique lowercase normalized word tokens; cosine uses sklearn TfidfVectorizer defaults. Similarity reference is data/unseen_test.csv only.",
        "Flag rule: any Jaccard > 0.50 or TF-IDF cosine > 0.60 against original unseen rows.",
        "",
        "Row | Category | Label | max Jaccard (unseen row) | max cosine (unseen row) | Jaccard to Sample #10 | cosine to Sample #10 | Flag",
        "---:|---|---|---|---|---:|---:|---",
    ]
    flagged = []
    for i, row in new.iterrows():
        jac = [jaccard(new_tokens[i], tokens) for tokens in unseen_tokens]
        ji, ci = int(np.argmax(jac)), int(np.argmax(cos[i]))
        j10 = jac[9]
        c10 = float(cos[i, 9])
        flag = max(jac) > 0.50 or float(cos[i, ci]) > 0.60
        if flag:
            flagged.append(i + 1)
        lines.append(f"{i+1:02d} | {row.category} | {row.label} | {max(jac):.3f} (#{ji+1:02d}) | {cos[i,ci]:.3f} (#{ci+1:02d}) | {j10:.3f} | {c10:.3f} | {'YES' if flag else 'no'}")
    lines += ["", "Flagged v4 row numbers: " + (", ".join(f"#{i:02d}" for i in flagged) if flagged else "none")]
    Path("reports/v4_similarity_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("reports/v4_data_note.txt").write_text(
        "The 12 C4_red_bleeding sentences were motivated by the Sample #10 miss diagnosed in reports/v4_diagnosis.txt. They are training-only additions and do not quote Sample #10.\n",
        encoding="utf-8")
    print(f"v4 data ready: {len(base)} + {len(new)} = {len(combined)} rows")
    print("new labels:", new.label.value_counts().to_dict())
    print("commas in text: 0; exact duplicates: none; flagged rows:", flagged or "none")
    print("v4 similarity audit: reports/v4_similarity_audit.txt")


if __name__ == "__main__":
    main()
