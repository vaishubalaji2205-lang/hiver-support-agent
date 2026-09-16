# hiver-support-agent
# Apple Support Agent — Report

## Problem

Build an agent for `@AppleSupport` customer-service tweets that (1) classifies
intent, (2) drafts a reply grounded in how Apple has historically resolved
similar issues, and (3) decides whether to auto-handle or escalate to a
human, with a stated reason.

## Approach

**Golden set.** 250 `@AppleSupport` customer messages sampled uniformly at
random (`random_state=42`) from ~2.8M rows in the Twitter customer-support
dataset, after de-duplicating identical text. Hand-labeled into 6 intents
identified by reading the data first, not decided upfront:
`device_software_issue` (166), `general_information` (57), `account_access`
(11), `purchase_order` (9), `refund_request` (4), `payment_billing` (3). The
skew toward `device_software_issue` reflects real traffic patterns, but the
three tiny classes limit what any evaluation on this set can prove.

**Pipeline.** `IntentClassifier` → `EscalationDecider` → `ReplyDrafter`.
Escalation is rule-based and auditable: escalate if confidence < 0.6, or if
the intent is `payment_billing`, `refund_request`, or `account_access`
(money/account access always routes to a human regardless of confidence);
otherwise auto-handle. Every decision carries a plain-English reason.
Reply drafting retrieves the 3 most similar historical (customer message,
Apple reply) pairs via TF-IDF cosine similarity over ~82k real past
conversations, then an LLM drafts a new reply grounded in them. When no LLM
is available, the pipeline falls back to the single closest historical
reply used as-is — the whole agent runs fully offline, at zero cost.

## Results vs. baselines

| Classifier | Accuracy | Macro F1 | n |
|---|---|---|---|
| Trivial (keyword rules) | 66.8% | 0.186 | 238 |
| Simple (TF-IDF + Multinomial Naive Bayes, 3-fold CV) | 66.4% | 0.133 | 250 |
| LLM few-shot (Gemini Flash-Lite) | 73.3% | 0.401 | 15 |

(238 = golden set minus 12 examples held out as the LLM's few-shot exemplars,
to avoid testing it on rows it had already seen. The LLM row is a random
15-example subsample, not the full 238 — see limitations below.)

## What's misleading about the headline number

66.8% accuracy on the trivial baseline sounds like a working classifier. It
isn't: `device_software_issue` alone is 69% of the eval set, and the
baseline predicts it for nearly everything (96% recall), while getting 0%
recall on 3 of the 6 classes entirely. Macro F1 (0.186) exposes this;
accuracy hides it.

The simple ML baseline is the sharper version of the same story: it actually
scores *worse* on macro F1 (0.133) than the hand-written keyword rules,
despite near-identical accuracy. With only 250 labeled examples and 3
classes under 10 examples each, a trained model has nothing to learn from
the minority classes and just collapses to the majority-class habit even
harder than the rules did (100% recall, 66% precision on
`device_software_issue` — it predicts that class for literally every input).
More sophisticated modeling is not automatically better; here, the labeled
data is the bottleneck, not the algorithm.

The LLM row is real progress (macro F1 more than doubles), but it's measured
on 15 examples because of free-tier API quota limits — only 3 of the 6
intents even appear in that sample. Treat it as a promising signal, not a
settled number.

## Reply quality: LLM-as-judge vs. human agreement

A judge LLM scores drafted replies 1-5 on groundedness, helpfulness, tone,
conciseness, and an overall holistic score. Validated against my own manual
ratings on 17 replies drawn from the golden set (auto-handled messages only
— escalated ones have nothing to judge):

- Mean absolute error: 0.65
- Exact match rate: 52.9%
- Within ±1 point: 82.4%
- Spearman correlation: 0.35

The MAE and within-±1 numbers look reasonable, but the weak rank correlation
reveals a **leniency bias**: the judge rated 14 of 17 replies a 5, compressing
its own usable range. When one rater barely touches the bottom of the scale,
there's little room for rankings to agree or disagree, so correlation
collapses even while raw scores stay numerically close. This limits the
judge's usefulness for distinguishing "good" from "great" replies, though it
still tracks genuinely bad ones (its one 3-score matched a human 3).

## Failure analysis

- **Minority intents are effectively unlearnable by non-LLM baselines.**
  `refund_request` and `payment_billing` have 3-4 examples total in the
  golden set; neither baseline predicts them correctly even once (0%
  recall, both baselines).
- **The keyword baseline's `general_information` bucket is dead on
  arrival.** It's the explicit fallback when nothing else matches, yet it
  scores 0% recall — because `device_software_issue`'s keyword list (`app`,
  `update`, `ios`, `error`, etc.) is broad enough to fire on most
  Apple-related tweets regardless of actual intent, intercepting messages
  before they ever reach the fallback branch. The rule is too generic to be
  a real filter.
- **Escalation correctly separates confident from uncertain cases** — e.g.
  a clear device complaint got 0.98 confidence and auto-handled; a vaguer
  message got 0.30 and correctly escalated instead of guessing.

## Limitations & next steps (one more week)

1. Re-run the LLM classifier and judge on the full held-out set (not a
   15/17-example subsample) once free-tier quota resets or a paid tier is
   available, for statistically robust numbers.
2. Expand the golden set with targeted (not purely random) sampling for
   `refund_request` and `payment_billing`, since minority-class metrics are
   currently meaningless with only 3-4 examples each.
3. Restore the full 4-criteria judge rubric (currently simplified to a
   single overall score to make manual labeling tractable) to see whether
   per-criterion agreement diverges from the overall-score agreement.
4. Investigate mitigating the judge's leniency bias — e.g. more explicit
   per-score-level anchors in the rubric, or pairwise comparison instead of
   absolute 1-5 scoring, which tends to reduce this bias in LLM judges.
5. A/B test LLM-generated replies against the retrieval-only fallback at
   scale, once quota allows judging enough of both to compare meaningfully.
# Decision Log

Non-obvious calls made while building this, and why.

1. **Picked Apple as the one brand**, and fixed a bug where `prepare_data.py`
   originally sampled across all brands in the dataset — inconsistent with
   the assignment's "pick one brand" requirement and with the other scripts,
   which were already Apple-only.
2. **Defined the 6-intent taxonomy by reading the data first**, not before —
   avoided guessing categories that might not match real traffic patterns.
3. **Held out 12 golden-set examples as fixed few-shot exemplars** for the
   LLM classifier's prompt, and excluded exactly those from evaluation, so
   the model isn't scored on rows it already saw.
4. **Chose TF-IDF + Multinomial Naive Bayes over Logistic Regression** for
   the "simple" baseline — not for accuracy reasons, but because Logistic
   Regression's compiled solver got blocked by a Windows security policy
   (Application Control) on my machine; Naive Bayes avoids that module
   entirely and is an equally standard choice for text classification.
5. **Used 3-fold (not 5-fold) stratified cross-validation** for the simple
   baseline, because the smallest class (`payment_billing`, n=3) can't
   support more folds than it has examples.
6. **Set escalation confidence threshold at 0.6**, with `payment_billing`,
   `refund_request`, and `account_access` always escalating regardless of
   confidence, since money and account access carry higher risk if the
   model is wrong.
7. **Built a fully offline fallback path** (local classifier + retrieval-only
   replies) so the whole pipeline is testable and demoable with zero API
   cost — this doubled as the actual solution when API access broke.
8. **Switched providers from Anthropic to Google Gemini mid-project**,
   prioritizing a genuinely free path after hitting a billing setup
   blocker.
9. **Avoided hardcoding a dated Gemini model string** (`gemini-2.5-flash`
   was already in its deprecation window); used an alias, then pinned to a
   specific Flash-Lite model after hitting a 20-request/day quota wall on
   the newest flagship model.
10. **Capped the LLM classifier evaluation to a random 15-example
    subsample**, not the full 238, because of free-tier daily quota
    limits — documented explicitly here and in the report rather than
    silently presenting a partial-coverage number as comprehensive.
11. **Added retry-with-backoff only for transient errors** (503/429 server
    overload), not for hard quota exhaustion — retrying a daily cap wastes
    time without fixing anything.
12. **Simplified the human-rating rubric from 4 criteria + overall to a
    single overall score**, trading rubric granularity for actually
    completing a real human-agreement check within a reasonable time
    budget.
13. **Reported Spearman correlation alongside MAE and exact-match**, since
    a systematically lenient judge can look close on absolute-difference
    metrics while showing weak rank agreement — and that's exactly what
    happened here.
14. **Used retrieval-only replies (the closest historical Apple reply,
    verbatim) as the offline fallback**, rather than skipping reply
    generation entirely when no LLM is available.
15. **Didn't silently swallow API errors** — every fallback path logs why
    it fell back, after an earlier version's silent `except Exception`
    made a real quota error look like a mysterious bug during testing.
