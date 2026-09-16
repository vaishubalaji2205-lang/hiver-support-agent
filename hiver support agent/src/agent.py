"""
agent.py — Apple Support Agent

Three pieces:
1. IntentClassifier   — classifies a message into one of 6 intents
2. ReplyDrafter       — drafts a reply using similar historical replies
3. EscalationDecider  — decides auto-handle vs escalate

Run:
    python src/agent.py "My iPhone won't update to the latest iOS"
"""

import os
import re
import json
import sys
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from google import genai
from google.genai import types


load_dotenv()


# "gemini-flash-latest" is an ALIAS Google keeps pointed at their current
# recommended Flash model, rather than a dated snapshot like
# "gemini-2.5-flash" (which is already in its deprecation window as of
# mid-2026, and Google's Flash lineup has since moved past the 3.x line).
# Google's own docs note this alias is experimental / can shift without
# much notice, which is fine for coursework but not for anything you'd
# ship. For a pinned, more stable choice, check the current model list at
# https://ai.google.dev/gemini-api/docs/models and set an explicit ID here.
MODEL_NAME = "gemini-flash-lite-latest"

INTENTS = [
    "device_software_issue",
    "account_access",
    "payment_billing",
    "refund_request",
    "purchase_order",
    "general_information",
]

INTENT_DESCRIPTIONS = {
    "device_software_issue": (
        "Problems with iOS, apps, updates, bugs, crashes, "
        "battery, or device performance."
    ),
    "account_access": (
        "Trouble signing in, Apple ID, passwords, "
        "two-factor authentication, or locked accounts."
    ),
    "payment_billing": (
        "Questions or complaints about charges, subscriptions, "
        "invoices, or billing."
    ),
    "refund_request": (
        "Explicit requests for a refund or money back."
    ),
    "purchase_order": (
        "Order status, purchases, trade-ins, delivery, or shipping."
    ),
    "general_information": (
        "Everything else, including general questions, praise, "
        "unclear messages, or off-topic messages."
    ),
}

GOLDEN_SET_PATH = "data/processed/golden_set.csv"
CONVERSATIONS_PATH = "data/processed/apple_conversations.csv"

SENSITIVE_INTENTS = {
    "payment_billing",
    "refund_request",
    "account_access",
}

CONFIDENCE_THRESHOLD = 0.6

# Must match whatever placeholder text sits in .env.example, or the
# "did they actually paste a key" guard below silently does nothing.
PLACEHOLDER_KEY = "YOUR_NEW_GEMINI_API_KEY"


def load_few_shot_examples(
    golden_set_path=GOLDEN_SET_PATH,
    per_intent=2,
):
    """
    Load a small fixed number of examples from each intent.
    """

    df = pd.read_csv(golden_set_path)

    examples = []
    exclude_ids = set()

    for intent, group in df.groupby("custom_intent"):
        picked = group.sort_values("tweet_id").head(per_intent)

        exclude_ids.update(picked["tweet_id"].tolist())

        for _, row in picked.iterrows():
            examples.append(
                {
                    "text": row["text"],
                    "intent": row["custom_intent"],
                }
            )

    return examples, exclude_ids


def _extract_text(response):
    """
    Extract text from a Gemini response.
    """

    return response.text or ""


def _parse_json_response(raw):
    """
    Parse JSON returned by Gemini.
    """

    cleaned = raw.strip()

    cleaned = re.sub(
        r"^```json",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    cleaned = re.sub(
        r"^```",
        "",
        cleaned,
    ).strip()

    cleaned = re.sub(
        r"```$",
        "",
        cleaned,
    ).strip()

    return json.loads(cleaned)


def create_gemini_client():
    """
    Create a Gemini client using GEMINI_API_KEY from .env.
    Returns None if no key is set, or if it's still the placeholder text —
    this is what lets the whole pipeline fall back to local/offline mode
    automatically instead of throwing a confusing auth error.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    if api_key == PLACEHOLDER_KEY:
        return None

    return genai.Client(api_key=api_key)


def _call_with_retries(func, max_retries=3, base_delay=2):
    """
    Calls func() and retries on transient errors (server overload / rate
    limits) with exponential backoff, so a momentary Gemini hiccup doesn't
    quietly degrade a chunk of a batch run (e.g. evaluate.py's 238 calls)
    into fallback results. Re-raises the last error if retries run out —
    callers still need their own except block for genuinely bad requests.
    """
    last_error = None
    transient_markers = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as error:
            last_error = error
            if attempt == max_retries or not any(m in str(error) for m in transient_markers):
                raise
            time.sleep(base_delay * (2 ** attempt))

    raise last_error


class IntentClassifier:
    """
    Gemini-based intent classifier.
    """

    def __init__(
        self,
        client=None,
        few_shot_examples=None,
        model=MODEL_NAME,
    ):
        self.client = client or create_gemini_client()
        self.few_shot_examples = few_shot_examples or []
        self.model = model

    def _system_prompt(self):
        lines = [
            "You are an intent classifier for Apple customer support tweets.",
            "Classify the customer's message into EXACTLY ONE intent.",
            "",
        ]

        for intent in INTENTS:
            lines.append(
                f"- {intent}: {INTENT_DESCRIPTIONS[intent]}"
            )

        if self.few_shot_examples:
            lines.append("\nExamples:")

            for example in self.few_shot_examples:
                lines.append(
                    f'Message: "{example["text"]}"\n'
                    f'Intent: {example["intent"]}\n'
                )

        lines.append(
            "\nRespond with ONLY a JSON object in this format:\n"
            '{"intent": "<one valid intent>", '
            '"confidence": <number from 0.0 to 1.0>, '
            '"reasoning": "<one short sentence>"}'
        )

        return "\n".join(lines)

    def classify(self, text):
        if self.client is None:
            return {
                "intent": "general_information",
                "confidence": 0.0,
                "reasoning": "Gemini API key is not configured.",
            }

        try:
            response = _call_with_retries(lambda: self.client.models.generate_content(
                model=self.model,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt(),
                    temperature=0,
                    response_mime_type="application/json",
                    max_output_tokens=200,
                ),
            ))

            raw = _extract_text(response)
            parsed = _parse_json_response(raw)

            if parsed.get("intent") not in INTENTS:
                parsed["intent"] = "general_information"

            confidence = float(
                parsed.get("confidence", 0.0)
            )

            parsed["confidence"] = max(
                0.0,
                min(1.0, confidence),
            )

            return parsed

        except Exception as error:
            return {
                "intent": "general_information",
                "confidence": 0.0,
                "reasoning": (
                    f"Gemini classification failed: {str(error)[:120]}"
                ),
            }


class LocalIntentClassifier:
    """
    Local TF-IDF + Multinomial Naive Bayes classifier.

    No API key is required.
    """

    def __init__(
        self,
        golden_set_path=GOLDEN_SET_PATH,
    ):
        df = pd.read_csv(golden_set_path)

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        ngram_range=(1, 2),
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    MultinomialNB(),
                ),
            ]
        )

        self.pipeline.fit(
            df["text"],
            df["custom_intent"],
        )

    def classify(self, text):
        prediction = self.pipeline.predict([text])[0]

        confidence = float(
            max(
                self.pipeline.predict_proba([text])[0]
            )
        )

        return {
            "intent": prediction,
            "confidence": confidence,
            "reasoning": (
                "Local TF-IDF + Naive Bayes classifier "
                "(no API used)."
            ),
        }


class ReplyRetriever:
    """
    Retrieve similar historical Apple Support conversations.
    """

    def __init__(
        self,
        conversations_path=CONVERSATIONS_PATH,
    ):
        self.df = pd.read_csv(
            conversations_path
        ).dropna(
            subset=["text", "reply_text"]
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["text"]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        top_indices = similarities.argsort()[::-1][:k]

        results = []

        for index in top_indices:
            row = self.df.iloc[index]

            results.append(
                {
                    "past_message": row["text"],
                    "past_reply": row["reply_text"],
                    "similarity": float(
                        similarities[index]
                    ),
                }
            )

        return results


class ReplyDrafter:
    """
    Draft a reply using Gemini and historical replies.
    """

    def __init__(
        self,
        client=None,
        retriever=None,
        model=MODEL_NAME,
    ):
        self.retriever = retriever or ReplyRetriever()
        self.model = model

        if client is not None:
            self.client = client
        else:
            self.client = create_gemini_client()

    def draft(self, text, intent, k=3):
        examples = self.retriever.retrieve(
            text,
            k=k,
        )

        if self.client is None:
            best_reply = (
                examples[0]["past_reply"]
                if examples
                else None
            )

            return best_reply, examples

        examples_block = "\n\n".join(
            (
                f'Past customer message: "{example["past_message"]}"\n'
                f'Apple reply: "{example["past_reply"]}"'
            )
            for example in examples
        )

        system_prompt = (
            "You are drafting a reply as @AppleSupport on Twitter. "
            "Use a short, empathetic, and helpful tone. "
            "Use the historical examples as guidance. "
            f"The classified intent is: {intent}.\n\n"
            f"Historical examples:\n\n{examples_block}\n\n"
            "Write ONLY the reply text. "
            "Keep the reply under 280 characters."
        )

        try:
            response = _call_with_retries(lambda: self.client.models.generate_content(
                model=self.model,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=150,
                ),
            ))

            return _extract_text(response).strip(), examples

        except Exception:
            best_reply = (
                examples[0]["past_reply"]
                if examples
                else None
            )

            return best_reply, examples


class EscalationDecider:
    """
    Deterministic escalation rules.
    """

    def __init__(
        self,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        sensitive_intents=SENSITIVE_INTENTS,
    ):
        self.confidence_threshold = confidence_threshold
        self.sensitive_intents = sensitive_intents

    def decide(self, intent, confidence):
        if confidence < self.confidence_threshold:
            return (
                "escalate",
                (
                    f"classifier confidence {confidence:.2f} "
                    f"is below threshold "
                    f"{self.confidence_threshold}"
                ),
            )

        if intent in self.sensitive_intents:
            return (
                "escalate",
                (
                    f"intent '{intent}' involves money or "
                    "account access — routed to a human"
                ),
            )

        return (
            "auto_handle",
            (
                f"high-confidence ({confidence:.2f}) "
                "non-sensitive intent"
            ),
        )


class AppleSupportAgent:
    """
    Main Apple Support Agent.
    """

    def __init__(self):
        self.has_api = create_gemini_client() is not None

        if self.has_api:
            few_shot, self.few_shot_ids = (
                load_few_shot_examples()
            )

            self.classifier = IntentClassifier(
                few_shot_examples=few_shot
            )
        else:
            self.few_shot_ids = set()
            self.classifier = LocalIntentClassifier()

        self.retriever = ReplyRetriever()

        self.drafter = ReplyDrafter(
            retriever=self.retriever
        )

        self.decider = EscalationDecider()

    def handle(self, text):
        classification = self.classifier.classify(
            text
        )

        intent = classification["intent"]

        confidence = float(
            classification.get("confidence", 0.0)
        )

        decision, reason = self.decider.decide(
            intent,
            confidence,
        )

        reply = None
        grounding = []

        if decision == "auto_handle":
            reply, grounding = self.drafter.draft(
                text,
                intent,
            )

        return {
            "message": text,
            "intent": intent,
            "confidence": confidence,
            "classifier_reasoning": (
                classification.get("reasoning")
            ),
            "decision": decision,
            "decision_reason": reason,
            "draft_reply": reply,
            "grounding_examples": grounding,
        }


if __name__ == "__main__":
    if create_gemini_client() is None:
        print(
            "No GEMINI_API_KEY set — running with the "
            "local classifier and retrieval-only replies "
            "(no API calls made).\n"
        )

    message = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "My iPhone won't update to the latest iOS, "
            "it just keeps failing"
        )
    )

    agent = AppleSupportAgent()

    result = agent.handle(message)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )
