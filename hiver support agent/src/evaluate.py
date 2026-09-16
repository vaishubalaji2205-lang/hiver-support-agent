"""
evaluate.py — evaluation harness

Currently covers:
  - Intent classification: accuracy + macro F1 vs. the trivial keyword
    baseline, on a held-out slice of the golden set (the few-shot
    exemplars used in agent.py's prompt are excluded so the model isn't
    scored on rows it already saw).

TODO (next): LLM-as-judge rubric for reply quality, validated against a
manually-scored subsample (see decision log / report for the plan).
"""

import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from agent import AppleSupportAgent, load_few_shot_examples, GOLDEN_SET_PATH  # noqa: E402
from simple_baseline import evaluate_simple_baseline  # noqa: E402


def trivial_baseline(text):
    """Same rule-based classifier as prepare_data.py — keep in sync by hand."""
    text = str(text).lower()
    if any(w in text for w in ["refund", "money back", "reimburse"]):
        return "refund_request"
    if any(w in text for w in ["charged", "billing", "payment", "invoice", "subscription", "price"]):
        return "payment_billing"
    if any(w in text for w in ["order", "purchase", "bought", "buy", "delivery", "shipping",
                                "arrived", "package", "trade in", "pre-order", "preorder"]):
        return "purchase_order"
    if any(w in text for w in ["password", "login", "log in", "sign in", "locked", "account",
                                "verification", "two-factor", "2fa", "apple id"]):
        return "account_access"
    if any(w in text for w in ["app", "update", "ios", "software", "bug", "crash",
                                "not working", "error", "battery", "freeze", "frozen", "restart"]):
        return "device_software_issue"
    return "general_information"


def build_eval_split():
    df = pd.read_csv(GOLDEN_SET_PATH)
    _, few_shot_ids = load_few_shot_examples(GOLDEN_SET_PATH)
    eval_df = df[~df["tweet_id"].isin(few_shot_ids)].copy()
    return eval_df, len(df) - len(eval_df)


def evaluate_classifier():
    eval_df, n_held_out = build_eval_split()
    print(f"Evaluating on {len(eval_df)} golden-set examples "
          f"({n_held_out} reserved as few-shot exemplars, excluded here)\n")

    eval_df["trivial_pred"] = eval_df["text"].apply(trivial_baseline)

    print("=== Trivial (keyword) baseline ===")
    print("Accuracy:", round(accuracy_score(eval_df["custom_intent"], eval_df["trivial_pred"]), 3))
    print("Macro F1:", round(f1_score(eval_df["custom_intent"], eval_df["trivial_pred"],
                                       average="macro", zero_division=0), 3))
    print(classification_report(eval_df["custom_intent"], eval_df["trivial_pred"], zero_division=0))

    print("\n(Simple baseline runs on the full 250-example golden set via "
          "cross-validation, not this 238-example split — see its own fold "
          "count printed below.)\n")
    evaluate_simple_baseline()

    if os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "YOUR_NEW_GEMINI_API_KEY":
        # Free-tier daily quotas on the newest Gemini models have been cut
        # sharply (as low as 20 requests/day) — cap the LLM eval to a small
        # sample so one run doesn't exhaust the whole day's quota. Bump this
        # up if you're on a paid tier or have quota to spare.
        LLM_EVAL_SAMPLE_SIZE = 15
        llm_eval_df = eval_df.sample(n=min(LLM_EVAL_SAMPLE_SIZE, len(eval_df)), random_state=1)
        print(f"(LLM classifier eval capped to {len(llm_eval_df)} examples — "
              f"free-tier daily quota, not the full {len(eval_df)}-example set)\n")

        agent = AppleSupportAgent()
        llm_eval_df["llm_pred"] = llm_eval_df["text"].apply(lambda t: agent.classifier.classify(t)["intent"])
        eval_df = llm_eval_df

        print("\n=== LLM few-shot classifier ===")
        print("Accuracy:", round(accuracy_score(eval_df["custom_intent"], eval_df["llm_pred"]), 3))
        print("Macro F1:", round(f1_score(eval_df["custom_intent"], eval_df["llm_pred"],
                                           average="macro", zero_division=0), 3))
        print(classification_report(eval_df["custom_intent"], eval_df["llm_pred"], zero_division=0))
    else:
        print("\nGEMINI_API_KEY not set — skipping LLM classifier eval.")
        print("Add your key to .env and re-run to compare it against the baseline above.")

    return eval_df


if __name__ == "__main__":
    evaluate_classifier()
