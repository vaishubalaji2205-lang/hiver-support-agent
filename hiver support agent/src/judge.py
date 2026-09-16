"""
judge.py — LLM-as-judge for reply quality.

Scores a (customer message, draft reply) pair on a 4-criterion rubric,
each 1-5, plus a holistic overall 1-5 score. This is the one piece of
the pipeline that genuinely needs an LLM — there's no honest local
fallback for "how good is this reply," so GEMINI_API_KEY is required
to actually run it.

Usage:
    python src/judge.py "customer message" "draft reply text"
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))
from agent import create_gemini_client, _extract_text, _parse_json_response, _call_with_retries, MODEL_NAME  # noqa: E402
from google.genai import types  # noqa: E402

RUBRIC = {
    "groundedness": (
        "Does the reply stick to what Apple would plausibly say, without "
        "inventing facts, policies, or promises that aren't supported by "
        "the situation?"
    ),
    "helpfulness": (
        "Does the reply actually address the customer's specific problem, "
        "rather than a generic non-answer?"
    ),
    "tone": (
        "Is the reply empathetic, professional, and consistent with "
        "Apple's real support voice — calm and concise, not robotic or "
        "dismissive?"
    ),
    "conciseness": (
        "Is the reply appropriately brief and clear for a Twitter reply, "
        "without unnecessary padding?"
    ),
}

CRITERIA = list(RUBRIC.keys())


class ReplyJudge:
    def __init__(self, client=None, model=MODEL_NAME):
        self.client = client if client is not None else create_gemini_client()
        self.model = model

    def _system_prompt(self):
        lines = [
            "You are grading a customer support reply written by an AI "
            "agent responding as @AppleSupport on Twitter.",
            "Score the reply on these criteria, each from 1 (poor) to 5 (excellent):",
            "",
        ]
        for name, desc in RUBRIC.items():
            lines.append(f"- {name}: {desc}")
        lines.append(
            "\nAlso give an 'overall' score from 1-5 that is your holistic "
            "judgment, not necessarily the average of the above.\n"
            "\nRespond with ONLY a JSON object in this exact form:\n"
            '{"groundedness": <1-5>, "helpfulness": <1-5>, "tone": <1-5>, '
            '"conciseness": <1-5>, "overall": <1-5>, "reasoning": "<one short sentence>"}'
        )
        return "\n".join(lines)

    def judge(self, customer_message, draft_reply, grounding_context=""):
        if self.client is None:
            raise RuntimeError(
                "ReplyJudge needs GEMINI_API_KEY set — the judge itself "
                "must be an LLM call, so there's no local fallback for "
                "this one (unlike the classifier and reply drafter)."
            )

        user_content = (
            f'Customer message: "{customer_message}"\n\n'
            f'AI agent\'s draft reply: "{draft_reply}"\n'
        )
        if grounding_context:
            user_content += f"\nHistorical context the reply was grounded in:\n{grounding_context}\n"

        try:
            response = _call_with_retries(lambda: self.client.models.generate_content(
                model=self.model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt(),
                    temperature=0,
                    response_mime_type="application/json",
                    max_output_tokens=250,
                ),
            ))
            raw = _extract_text(response)
            scores = _parse_json_response(raw)
            for c in CRITERIA + ["overall"]:
                scores.setdefault(c, None)
            return scores
        except Exception as error:
            scores = {c: None for c in CRITERIA}
            scores["overall"] = None
            scores["reasoning"] = f"Judge call failed: {str(error)[:150]}"
            return scores


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python src/judge.py "customer message" "draft reply"')
        sys.exit(1)

    judge = ReplyJudge()
    result = judge.judge(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))