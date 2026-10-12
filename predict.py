"""
predict.py  -  command-line demo
    python predict.py --file case.txt
    python predict.py --text "अभियुक्त ... धारा 302 भा0द0वि0 ..."
    python predict.py --csv test.csv --row 5 --text_col text
"""
import argparse

import pandas as pd

from bail_engine import BailEngine, format_report


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    g.add_argument("--csv")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--model", default="model_tfidf_facts_only.joblib")
    ap.add_argument("--index", default="case_index.joblib")
    ap.add_argument("--top_k", type=int, default=3)
    a = ap.parse_args()

    if a.text:
        text = a.text
    elif a.file:
        text = open(a.file, encoding="utf-8").read()
    else:
        text = str(pd.read_csv(a.csv).iloc[a.row][a.text_col])

    engine = BailEngine(a.model, a.index, a.top_k)
    print(format_report(engine.analyze(text)))


if __name__ == "__main__":
    main()