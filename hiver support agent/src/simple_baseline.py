"""
simple_baseline.py — the assignment's required "simple" baseline (as
opposed to prepare_data.py's "trivial" keyword baseline): TF-IDF features
+ a Multinomial Naive Bayes classifier, evaluated with cross-validation on the
golden set. Runs entirely locally — no API key needed.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_SET_PATH = "data/processed/golden_set.csv"


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", MultinomialNB()),
    ])


def evaluate_simple_baseline(golden_set_path=GOLDEN_SET_PATH):
    df = pd.read_csv(golden_set_path)
    pipeline = build_pipeline()

    # The smallest class (payment_billing, n=3) caps how many stratified
    # folds we can use — StratifiedKFold requires n_splits <= smallest
    # class size. Worth a line in the decision log: metrics on very rare
    # classes here are noisy no matter what, given so few examples.
    min_class_size = df["custom_intent"].value_counts().min()
    n_splits = min(5, min_class_size)
    print(f"Using {n_splits}-fold stratified CV (smallest class has only "
          f"{min_class_size} examples, which caps fold count)\n")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(pipeline, df["text"], df["custom_intent"], cv=cv)

    print("=== Simple baseline: TF-IDF + Multinomial Naive Bayes ===")
    print("Accuracy:", round(accuracy_score(df["custom_intent"], preds), 3))
    print("Macro F1:", round(f1_score(df["custom_intent"], preds, average="macro", zero_division=0), 3))
    print(classification_report(df["custom_intent"], preds, zero_division=0))

    return preds


if __name__ == "__main__":
    evaluate_simple_baseline()
