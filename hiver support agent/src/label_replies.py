"""
label_replies.py — CLI for you to rate drafted replies with a single
1-5 overall score, producing ground truth to check the judge's
agreement against.

Run after src/generate_reply_sample.py, before src/evaluate_judge.py.

Usage:
    python src/label_replies.py
"""

import os

import pandas as pd

FILE = "data/processed/reply_judge_sample.csv"


def prompt_overall():
    while True:
        raw = input("  Overall quality (1=poor, 5=excellent), or q to save and quit\n  > ").strip()
        if raw.lower() == "q":
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter a number from 1 to 5, or q to quit.")


def main():
    if not os.path.exists(FILE):
        print(f"{FILE} doesn't exist yet — run src/generate_reply_sample.py first.")
        return

    df = pd.read_csv(FILE)
    rated = pd.to_numeric(df["human_overall"], errors="coerce").notna()
    remaining = (~rated).sum()

    if remaining == 0:
        print("All replies already rated. Run src/evaluate_judge.py next.")
        return

    print(f"{remaining} replies left to rate — one number each.\n")

    for i in range(len(df)):
        if pd.notna(pd.to_numeric(df.loc[i, "human_overall"], errors="coerce")):
            continue

        print("\n" + "=" * 60)
        print(f"Example {i + 1} of {len(df)}  (intent: {df.loc[i, 'intent']})")
        print(f"\nCustomer message:\n  {df.loc[i, 'message']}")
        print(f"\nDraft reply:\n  {df.loc[i, 'draft_reply']}\n")

        try:
            df.loc[i, "human_overall"] = prompt_overall()
        except KeyboardInterrupt:
            df.to_csv(FILE, index=False)
            print("\nSaved. Run this script again to continue.")
            return

        df.to_csv(FILE, index=False)
        print("Saved.")

    print("\nAll replies rated. Run src/evaluate_judge.py next.")


if __name__ == "__main__":
    main()