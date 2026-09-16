import os
import pandas as pd

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/support_data.csv"

SAMPLE_SIZE = 15000

# Same 6 classes used in golden_set.csv — keep these in sync by hand.
INTENTS = [
    "device_software_issue",
    "account_access",
    "payment_billing",
    "refund_request",
    "purchase_order",
    "general_information",
]


def prepare_data():
    print("Reading dataset...")

    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=columns,
        low_memory=False
    )

    print("Total rows:", len(df))

    # --- FIX: restrict to one brand (Apple), matching inspect_conversations.py
    # and create_golden_set.py, instead of sampling across every brand. ---
    inbound_df = df[
        (df["inbound"] == True) &
        (df["text"].str.contains("@AppleSupport", case=False, na=False))
    ].copy()

    print("AppleSupport inbound messages:", len(inbound_df))

    # Remove empty messages
    inbound_df = inbound_df.dropna(subset=["text"])

    # Sample a smaller dataset for faster development
    if len(inbound_df) > SAMPLE_SIZE:
        inbound_df = inbound_df.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    # Trivial rule-based classifier, predicting into the SAME taxonomy as
    # golden_set.csv so it can be scored directly as a baseline against it.
    # Order matters: first match wins, most specific rules go first.
    def assign_intent(text):
        text = str(text).lower()

        if any(w in text for w in [
            "refund", "money back", "reimburse"
        ]):
            return "refund_request"

        if any(w in text for w in [
            "charged", "billing", "payment", "invoice", "subscription", "price"
        ]):
            return "payment_billing"

        if any(w in text for w in [
            "order", "purchase", "bought", "buy", "delivery", "shipping",
            "arrived", "package", "trade in", "pre-order", "preorder"
        ]):
            return "purchase_order"

        if any(w in text for w in [
            "password", "login", "log in", "sign in", "locked", "account",
            "verification", "two-factor", "2fa", "apple id"
        ]):
            return "account_access"

        if any(w in text for w in [
            "app", "update", "ios", "software", "bug", "crash",
            "not working", "error", "battery", "freeze", "frozen", "restart"
        ]):
            return "device_software_issue"

        return "general_information"

    inbound_df["intent"] = inbound_df["text"].apply(assign_intent)

    output_df = inbound_df[[
        "tweet_id",
        "author_id",
        "text",
        "intent",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]].copy()

    os.makedirs("data/processed", exist_ok=True)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nProcessed file created:")
    print(OUTPUT_FILE)

    print("\nIntent distribution:")
    print(output_df["intent"].value_counts())


if __name__ == "__main__":
    prepare_data()
