# Hiver SDE Intern Assignment — Apple Support Agent

An AI agent for **@AppleSupport** customer-service tweets that:
1. **Classifies** each message into one of 6 intents
2. **Drafts a reply** grounded in how Apple has historically resolved similar issues
3. **Decides** whether to auto-handle or escalate to a human, with a stated reason

## Setup (~5 min)

```bash
git clone <your-repo-url>
cd hiver-support-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

## Data (~5 min)

Download the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
from Kaggle and place `twcs.csv` at:

```
data/raw/twcs.csv
```

(~500MB — not committed to this repo, see `.gitignore`.)

`data/processed/golden_set.csv` and `data/processed/apple_conversations.csv` ARE
committed since they're small and are what the pipeline is evaluated against.

## Reproduce the headline result (~5 min)

```bash
python src/prepare_data.py   # builds data/processed/support_data.csv (Apple-only sample + trivial baseline labels)
python src/evaluate.py       # scores trivial baseline (+ LLM classifier, if ANTHROPIC_API_KEY is set) against the golden set
```

Expected output: the trivial keyword baseline scores **~65-68% accuracy but only
~0.18-0.19 macro F1** on the golden set — accuracy looks fine at a glance but is
propped up almost entirely by one dominant class (`device_software_issue`), while
recall on `general_information`, `payment_billing`, and `refund_request` is ~0.
See the report's "what's misleading about my headline number" section.

## Try the agent on one message

```bash
python src/agent.py "My iPhone won't update to the latest iOS, it just keeps failing"
```

## Run tests

```bash
pytest tests/
```

Tests mock the Anthropic client, so they run without an API key or network access.

## Repo structure

```
hiver-support-agent/
├── data/
│   ├── raw/            # twcs.csv goes here (gitignored)
│   └── processed/      # golden_set.csv, apple_conversations.csv, support_data.csv
├── src/
│   ├── prepare_data.py # builds the Apple-only sample + trivial keyword baseline
│   ├── agent.py         # IntentClassifier + ReplyDrafter + EscalationDecider
│   └── evaluate.py      # classification metrics vs. trivial baseline
├── tests/
│   └── test_agent.py
├── requirements.txt
├── .env.example
└── README.md
```

## Golden set methodology

250 `@AppleSupport` customer messages were sampled uniformly at random
(`random_state=42`) from the full ~2.8M-row Twitter support dataset, after
de-duplicating identical message text. Each was hand-labeled into one of 6
intents defined by reading through the data first (not decided up front):
`device_software_issue`, `account_access`, `payment_billing`,
`refund_request`, `purchase_order`, `general_information`. The resulting
distribution is heavily skewed toward `device_software_issue` (166/250) —
expected, since most Apple support traffic is device/software complaints,
but it means accuracy alone is a misleading metric (see above).

Of the 250, 12 (2 per intent) are held out as fixed few-shot exemplars for
`agent.py`'s classifier prompt; `evaluate.py` excludes exactly those 12 from
scoring, leaving 238 for actual evaluation.

## Status

- [x] Golden set (250 hand-labeled examples, 6 intents)
- [x] Conversation pairs for grounding (~82k Apple customer↔reply pairs)
- [x] Trivial baseline (keyword rules) — scored against golden set
- [x] LLM few-shot intent classifier
- [x] Grounded reply drafting (TF-IDF retrieval + LLM)
- [x] Escalation decision layer (rule-based, with reasons)
- [ ] Simple baseline #2 (e.g. TF-IDF + logistic regression) for the report
- [ ] LLM-as-judge for reply quality + human-agreement check
- [ ] Report (`REPORT.md`)
- [ ] Decision log (`DECISIONS.md`)
