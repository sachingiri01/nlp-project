import argparse
import re

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV      # NEW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text


def extract_facts(text):
    if not isinstance(text, str):
        return ""

    start_pattern = r"'facts-and-arguments':\s*array\(\["
    start_match = re.search(start_pattern, text)
    if not start_match:
        return ""

    start = start_match.end()
    end_pattern = r"\],\s*dtype=object\),\s*'judge-opinion':"
    end_match = re.search(end_pattern, text[start:], re.DOTALL)

    if end_match:
        end = start + end_match.start()
        facts = text[start:end]
    else:
        facts = text[start:]

    facts = facts.replace("dtype=object", " ")
    facts = facts.replace("'", " ")
    facts = facts.replace('"', " ")
    facts = facts.replace("[", " ")
    facts = facts.replace("]", " ")
    facts = facts.replace("(", " ")
    facts = facts.replace(")", " ")
    facts = re.sub(r"\s+", " ", facts).strip()
    return facts


def load_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    print(f"\nLoading: {path}")
    df = pd.read_csv(path)
    print(f"Raw documents: {len(df)}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].apply(extract_facts)

    empty_count = (df[text_col].str.strip() == "").sum()
    print(f"Empty texts after extraction: {empty_count}")

    if len(df) > 0:
        print("\nExample extracted text:")
        print(df[text_col].iloc[0][:500])
        print()

    df = df[df[text_col].str.strip() != ""].copy()
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, drop_stopwords=False))
    df = df[df[text_col].str.strip() != ""].copy()

    print(f"Usable documents: {len(df)}")
    return df


def build_pipeline(model_name: str = "svm") -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    if model_name == "svm":
        # CHANGED: calibrated SVM -> gives real probabilities for the confidence score
        clf = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=1.0),
            cv=3,
            method="sigmoid",
        )
    elif model_name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser(
        description="Hindi legal bail outcome classification using TF-IDF."
    )
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--test_data", default=None, help="Optional separate test CSV")
    parser.add_argument("--text_col", default="text", help="Name of text column")
    parser.add_argument("--label_col", default="label", help="Name of label column")
    parser.add_argument("--model", default="svm", choices=["svm", "logreg"],
                        help="Classification model")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split size when --test_data is not provided")
    parser.add_argument("--out", default="model_tfidf_facts_only.joblib",
                        help="Output model file")
    args = parser.parse_args()

    if args.test_data:
        train_df = load_data(args.data, args.text_col, args.label_col)
        test_df = load_data(args.test_data, args.text_col, args.label_col)

        print("\n==========================================")
        print("DATASET SUMMARY")
        print("==========================================")
        print(f"Train documents : {len(train_df)}")
        print(f"Test documents : {len(test_df)}")
        print(f"Number of classes: {train_df[args.label_col].nunique()}")
        print("\nTrain label distribution:")
        print(train_df[args.label_col].value_counts())
        print("\nTest label distribution:")
        print(test_df[args.label_col].value_counts())

        X_train = train_df[args.text_col]
        y_train = train_df[args.label_col]
        X_test = test_df[args.text_col]
        y_test = test_df[args.label_col]
    else:
        df = load_data(args.data, args.text_col, args.label_col)

        print("\nDATASET SUMMARY\n")
        print(f"Documents : {len(df)}")
        print(f"Classes : {df[args.label_col].nunique()}")
        print("\nLabel distribution:")
        print(df[args.label_col].value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            df[args.text_col],
            df[args.label_col],
            test_size=args.test_size,
            random_state=42,
            stratify=df[args.label_col],
        )

    print("\n==========================================")
    print("MODEL")
    print("==========================================")
    print(f"Model: {args.model}")
    print("Features: TF-IDF")
    print("N-grams: (1, 2)")
    print("Max features: 50,000")
    print("Input: facts-and-arguments ONLY")
    print("Judge opinion: EXCLUDED")

    pipeline = build_pipeline(args.model)

    print("\n==========================================")
    print("TRAINING")
    print("==========================================")
    pipeline.fit(X_train, y_train)
    print("Training completed.")

    print("\n==========================================")
    print("EVALUATION")
    print("==========================================")
    y_pred = pipeline.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, args.out)
    print(f"\nModel saved to: {args.out}")


if __name__ == "__main__":
    main()