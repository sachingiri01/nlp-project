"""
build_case_index.py
Builds the searchable index of historical cases from the TRAINING csv.
Run once after training:
    python build_case_index.py --data train.csv --text_col text --label_col label
Creates case_index.joblib
"""
import argparse

import joblib
import pandas as pd

from baseline_tfidf import extract_facts
from bail_engine import label_to_name
from preprocess import clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training CSV (index must not contain test cases)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--model", default="model_tfidf_facts_only.joblib")
    ap.add_argument("--out", default="case_index.joblib")
    ap.add_argument("--max_cases", type=int, default=0, help="0 = use all (index gets big for 300k+ docs)")
    a = ap.parse_args()

    pipe = joblib.load(a.model)
    vec = pipe.named_steps["tfidf"]

    df = pd.read_csv(a.data).dropna(subset=[a.text_col, a.label_col])
    df["raw"] = df[a.text_col].apply(extract_facts)
    df = df[df["raw"].str.strip() != ""]
    if a.max_cases and len(df) > a.max_cases:
        df = df.sample(a.max_cases, random_state=42)
    print(f"Indexing {len(df)} cases ...")

    clean = df["raw"].apply(lambda t: clean_text(t, drop_stopwords=False))
    matrix = vec.transform(clean)          # same vectorizer as the classifier

    joblib.dump({
        "ids": [str(i) for i in df.index],             # = row number in the CSV
        "raw": df["raw"].tolist(),
        "outcomes": [label_to_name(x) for x in df[a.label_col]],
        "matrix": matrix,
    }, a.out)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()