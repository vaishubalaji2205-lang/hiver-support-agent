"""
Unit tests for judge.py. The Gemini client is mocked so these run
without an API key or network access.
"""

import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judge import ReplyJudge, CRITERIA  # noqa: E402


def make_mock_response(text):
    response = MagicMock()
    response.text = text
    return response


class TestReplyJudge:
    def test_judge_parses_valid_scores(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = make_mock_response(
            '{"groundedness": 4, "helpfulness": 5, "tone": 4, "conciseness": 5, '
            '"overall": 4, "reasoning": "solid, on-brand reply"}'
        )
        judge = ReplyJudge(client=mock_client)
        result = judge.judge("my app keeps crashing", "Try force closing the app.")

        assert result["overall"] == 4
        assert result["helpfulness"] == 5
        for c in CRITERIA:
            assert c in result

    def test_judge_raises_without_a_client(self):
        judge = ReplyJudge(client=None)
        judge.client = None  # simulate no API key configured
        with pytest.raises(RuntimeError):
            judge.judge("message", "reply")

    def test_judge_handles_api_failure_gracefully(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("rate limited")
        judge = ReplyJudge(client=mock_client)
        result = judge.judge("message", "reply")

        assert result["overall"] is None
        assert "rate limited" in result["reasoning"]