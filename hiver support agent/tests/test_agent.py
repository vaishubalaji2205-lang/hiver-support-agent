"""
Unit tests for agent.py. The Gemini client is mocked throughout so
these run without an API key or network access.
"""

import sys
import os
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import EscalationDecider, IntentClassifier, ReplyRetriever, _parse_json_response  # noqa: E402


def make_mock_response(text):
    """Mimics a Gemini response object, which exposes a simple .text property."""
    response = MagicMock()
    response.text = text
    return response


class TestEscalationDecider:
    def setup_method(self):
        self.decider = EscalationDecider(
            confidence_threshold=0.6,
            sensitive_intents={"payment_billing", "refund_request"},
        )

    def test_low_confidence_escalates(self):
        decision, reason = self.decider.decide("device_software_issue", 0.3)
        assert decision == "escalate"
        assert "confidence" in reason

    def test_sensitive_intent_escalates_even_with_high_confidence(self):
        decision, reason = self.decider.decide("refund_request", 0.95)
        assert decision == "escalate"
        assert "refund_request" in reason

    def test_high_confidence_non_sensitive_auto_handles(self):
        decision, reason = self.decider.decide("device_software_issue", 0.9)
        assert decision == "auto_handle"


class TestIntentClassifier:
    def test_classify_parses_valid_json(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_mock_response(
            '{"intent": "device_software_issue", "confidence": 0.87, "reasoning": "mentions iOS update"}'
        )
        classifier = IntentClassifier(client=mock_client)
        result = classifier.classify("my phone won't update")

        assert result["intent"] == "device_software_issue"
        assert result["confidence"] == 0.87

    def test_classify_falls_back_on_bad_json(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_mock_response("not valid json at all")
        classifier = IntentClassifier(client=mock_client)
        result = classifier.classify("hello")

        assert result["intent"] == "general_information"
        assert result["confidence"] == 0.0

    def test_unknown_intent_falls_back_to_general(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_mock_response(
            '{"intent": "made_up_intent", "confidence": 0.5, "reasoning": "test"}'
        )
        classifier = IntentClassifier(client=mock_client)
        result = classifier.classify("something")
        assert result["intent"] == "general_information"

    def test_classify_with_no_client_returns_general_information(self):
        classifier = IntentClassifier(client=None)
        classifier.client = None  # simulate no API key configured
        result = classifier.classify("something")
        assert result["intent"] == "general_information"
        assert result["confidence"] == 0.0


class TestReplyRetriever:
    def test_retrieve_returns_k_results(self, tmp_path):
        csv_path = tmp_path / "conversations.csv"
        pd.DataFrame({
            "tweet_id": [1, 2, 3],
            "text": [
                "my iphone battery drains fast",
                "app keeps crashing on ios 17",
                "how do I reset my password",
            ],
            "reply_tweet_id": [2, 4, 6],
            "reply_text": [
                "DM us your device details",
                "Try reinstalling the app",
                "Reset it from Settings",
            ],
        }).to_csv(csv_path, index=False)

        retriever = ReplyRetriever(conversations_path=str(csv_path))
        results = retriever.retrieve("my battery is draining quickly", k=2)

        assert len(results) == 2
        assert "past_message" in results[0]
        assert "past_reply" in results[0]


def test_parse_json_response_strips_code_fences():
    raw = '```json\n{"intent": "general_information", "confidence": 0.5, "reasoning": "x"}\n```'
    parsed = _parse_json_response(raw)
    assert parsed["intent"] == "general_information"