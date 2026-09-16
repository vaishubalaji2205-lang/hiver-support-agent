"""
generate_reply_sample.py — builds a sample of (message, draft_reply) pairs
for the judge-vs-human agreement check. Works with or without
GEMINI_API_KEY — falls back to retrieval-only replies if unset, exactly
like agent.py does normally.

Run after this: src/label_replies.py (to add your own ratings), then
src/evaluate_judge.py (to score the same sample with the LLM judge and
compare).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from agent import AppleSupportAgent, GOLDEN_SET_PATH  # noqa: E402

SAMPLE_SIZE = 20
OUTPUT_FILE = "data/processed/reply_judge_sample.csv"


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=7)

    agent = AppleSupportAgent()
    rows = []
    for _, row in sample.iterrows():
        result = agent.handle(row["text"])
        rows.append({
            "tweet_id": row["tweet_id"],
            "message": row["text"],
            "intent": result["intent"],
            "decision": result["decision"],
            "draft_reply": result["draft_reply"],
            "human_groundedness": "",
            "human_helpfulness": "",
            "human_tone": "",
            "human_conciseness": "",
            "human_overall": "",
        })

    out = pd.DataFrame(rows)
    # Escalated messages have no drafted reply — nothing to judge there,
    # so only keep rows that actually got auto-handled.
    before = len(out)
    out = out[out["draft_reply"].notna()].copy()

    os.makedirs("data/processed", exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Sampled {before} golden-set messages, {len(out)} got a drafted "
          f"reply (the rest escalated, so there's nothing to judge there).")
    print(f"Saved to {OUTPUT_FILE}")
    print("Next: run src/label_replies.py to add your own ratings.")


if __name__ == "__main__":
    main()