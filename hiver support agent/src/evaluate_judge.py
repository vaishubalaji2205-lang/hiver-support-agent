"""
evaluate_judge.py — runs the LLM judge on your human-rated sample and
reports how well the judge agrees with you, per the assignment's
requirement for evidence of judge-vs-human agreement.

Requires GEMINI_API_KEY and a data/processed/reply_judge_sample.csv that's
already been rated (run generate_reply_sample.py then label_replies.py
first).

Usage:
    python src/evaluate_judge.py
"""

import os
import sys
import time

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from judge import ReplyJudge, CRITERIA  # noqa: E402
from agent import create_gemini_client  # noqa: E402

FILE = "data/processed/reply_judge_sample.csv"
ALL_CRITERIA = CRITERIA + ["overall"]


def main():
    if create_gemini_client() is None:
        print("GEMINI_API_KEY not set — the judge itself is an LLM call, "
              "so this script can't run without a key. Everything else in "
              "the pipeline works without one; this is the one exception.")
        return

    if not os.path.exists(FILE):
        print(f"{FILE} doesn't exist — run generate_reply_sample.py and "
              "label_replies.py first.")
        return

    df = pd.read_csv(FILE)
    df = df[pd.to_numeric(df["human_overall"], errors="coerce").notna()].copy()

    if df.empty:
        print("No human-labeled rows found. Run label_replies.py first.")
        return

    print(f"Scoring {len(df)} human-rated replies with the LLM judge...\n")

    judge = ReplyJudge()

    for c in ALL_CRITERIA:
        df[f"judge_{c}"] = None

    for i, (idx, row) in enumerate(df.iterrows()):
        result = judge.judge(row["message"], row["draft_reply"])
        for c in ALL_CRITERIA:
            df.at[idx, f"judge_{c}"] = result.get(c)

        if result.get("overall") is None:
            print(f"  Row {i + 1}/{len(df)} FAILED: {result.get('reasoning')}")
        else:
            print(f"  Row {i + 1}/{len(df)} scored: overall={result.get('overall')}")

        if i < len(df) - 1:
            time.sleep(4)  # stay under free-tier per-minute rate limits

    print("=== Judge vs. human agreement ===")
    for c in ALL_CRITERIA:
        human_col = pd.to_numeric(df[f"human_{c}"], errors="coerce")
        judge_col = pd.to_numeric(df[f"judge_{c}"], errors="coerce")
        valid = human_col.notna() & judge_col.notna()

        if valid.sum() < 2:
            print(f"\n{c}: not enough valid judge responses to compare.")
            continue

        diff = (human_col[valid] - judge_col[valid]).abs()
        mae = diff.mean()
        exact_match = (diff == 0).mean()
        within_one = (diff <= 1).mean()
        corr, _ = spearmanr(human_col[valid], judge_col[valid])

        print(f"\n{c}  (n={valid.sum()}):")
        print(f"  Mean absolute error:  {mae:.2f}")
        print(f"  Exact match rate:     {exact_match:.2%}")
        print(f"  Within +/-1 point:    {within_one:.2%}")
        print(f"  Spearman correlation: {corr:.2f}")

    df.to_csv(FILE, index=False)
    print(f"\nSaved judge scores back to {FILE}")


if __name__ == "__main__":
    main()