# Decision log

Plain list of non-obvious decisions and why. This is a living document during
build — entries marked **[TBD]** get filled in once we've run the pipeline on
real data (brand choice, taxonomy specifics, threshold values).

1. **Retrieval is TF-IDF, not embeddings.** The build sandbox's network is
   locked to GitHub/PyPI (no Hugging Face, no GPU), so a downloaded
   sentence-transformers checkpoint or a hosted embeddings API isn't
   reliably available — and a reviewer reproducing this in <15 min with no
   GPU shouldn't need either. TF-IDF + cosine over cleaned tweet text is a
   reasonable match for "find a similar past issue" at this scale, at the
   cost of missing paraphrases that share no vocabulary.
2. **Escalation is a rule stack, not a learned classifier.** For a support
   agent, "why did it decide that" must be answerable in one sentence to a
   reviewer and to the brand's own team. A small ordered set of rules is
   auditable and directly editable; a learned escalation model would need
   its own explainability layer to hit the same bar. Cost: the rule
   thresholds are hand-set, not learned from data — see report.md's
   "misleading headline number" section.
3. **Always-escalate intents/keywords fire before the confidence check.**
   Ordering rules "most safety-critical first" means a high-confidence
   misclassification can't quietly bypass a hard safety rule.
4. **Classification unit = the opening customer message of a thread**, not
   every customer turn in a multi-turn conversation. Intent is tied to "why
   did this conversation start," not to conversational moves mid-thread
   (a later "yes that worked, thanks" isn't a new intent to classify). Turn
   count is still used as an *escalation* signal (repeated back-and-forth
   without resolution), just not as a re-classification trigger.
5. **Golden-set labels are LLM-drafted, human-finalized, and the
   finalize script blocks until every row is filled.** Pure LLM labels would
   make classifier evaluation circular (grading the model against its own
   guesses); `03b_finalize_golden_set.py` also reports the human-vs-LLM
   disagreement rate as evidence the labels weren't rubber-stamped.
6. **Both baselines are evaluated via 5-fold CV on the golden set itself**,
   not a separate held-out split. With only 150-250 labeled rows, a fixed
   train/test split would waste most of the golden set on training a
   throwaway logistic regression. CV gives every row an out-of-fold
   baseline prediction, so the LLM (zero-shot, no training split needed) and
   both baselines are compared on the *same* full set of items.
7. **The reply-quality judge scores 4 narrow dimensions (grounded, relevant,
   tone, actionable) plus an overall**, instead of one holistic 1-5. A
   single score can't tell you *why* a reply is bad; four narrow ones make
   failure modes diagnosable (e.g., "on-tone but ungrounded" vs "grounded
   but irrelevant") without needing a second pass.
8. **Retrieval excludes the exact conversation being evaluated
   (`exclude_root_id`).** Without this, a golden-set item's own historical
   resolution would sometimes be its own nearest neighbor, silently
   inflating both reply quality and the judge's "grounded" score.
9. **Golden-set sampling is stratified (random + keyword-oversampled +
   multi-turn-oversampled), not pure random.** A uniform random sample of
   150-250 items from a long-tailed intent distribution would make rare but
   consequential intents (billing disputes, safety/anger signals)
   statistically invisible. The oversampling is a documented bias, not a
   hidden one — see report.md.
10. **The reply-quality headline number is cut by auto-handled vs. escalated
    items.** A drafted reply for an item the policy would escalate anyway
    never reaches a customer in production, so blending its judge score into
    the topline average is exactly the kind of "misleading headline number"
    the report calls out explicitly.
11. **All LLM calls are cached to disk (`.cache/llm_cache.jsonl`) keyed by
    (model, system, prompt, temperature).** Needed for the "reproduce in
    under 15 minutes" requirement to actually hold on a second run, and
    keeps eval reruns deterministic instead of re-rolling every generation.
12. **Multi-reply branches in the raw data (`response_tweet_id` can be a
    comma-separated list) are resolved by following the first still-present
    id**, producing one linear thread per conversation rather than exploring
    every branch. Matches how a human agent reads a thread top-to-bottom;
    documented simplification rather than a silent one.
13. **Raw-CSV processing and everything downstream run on different
    machines, bridged by one dependency-free script.** The build sandbox
    couldn't reach Kaggle's servers, and separately couldn't move the raw
    ~500MB+ `twcs.csv` wholesale onto the machine running the rest of the
    pipeline (a hard transfer-size limit). Rather than force both onto one
    machine, `scripts/local_extract_candidates.py` is a standalone,
    zero-import script (only needs pandas) that does the one thing that
    needs the raw file -- rank brands and reconstruct conversation threads --
    and writes small per-brand JSON. Everything else (`src/`, `eval/`, the
    rest of `scripts/`) only ever touches those small JSON files via
    `data_loader.load_conversations`. A reviewer with normal, unrestricted
    access to both the dataset and their own machine can just run it locally
    with everything else, in the same place -- the split is a constraint of
    how this was built, not a requirement to reproduce it.
14. **Brand: Uber_Support.** Of the top 3 candidates by volume with clean
    English text (Uber_Support, AppleSupport, AmazonHelp -- AmazonHelp was
    noticeably multilingual, which would confound a taxonomy built assuming
    English), Uber_Support gives the escalation-decision half of the
    assignment real stakes: genuine safety incidents (crashes, driver
    misconduct) and payment disputes sit alongside routine questions, so
    "auto-handle vs. escalate, with a reason" has to do real work, not just
    triage mild annoyance. 56,270 total tweets, 7,488 reconstructed
    conversations sampled, 46% multi-turn.
15. **9-intent taxonomy (+ `other`), read off ~120 real sampled messages,
    not guessed; expanded once from an initial 8 after reading a second
    batch.** Final list: `safety_incident`, `driver_conduct_complaint`,
    `delivery_order_issue`, `billing_dispute`, `payment_or_app_technical_issue`,
    `account_access_or_security`, `lost_item`, `driver_account_onboarding`,
    `general_inquiry`, `other`. Notable calls:
    - `delivery_order_issue` (Uber Eats: wrong/missing/cold food, no-show
      courier) was split out as its own intent rather than stretched onto
      `driver_conduct_complaint`, because it's a different product line with
      a different resolution pattern (redeliver/refund an order, not
      reconnect with a ride driver) and it was common enough in the sample
      to be statistically real, not a one-off.
    - `account_access_or_security` (hacked, impersonated, banned/disabled
      without explanation, can't log in) was split out from
      `driver_account_onboarding` because it's the *rider's* account, not a
      driver's, and it showed up often enough (several independent examples)
      to deserve its own bucket rather than falling into `other`.
    - "Lost item due to driver negligence" was folded into `lost_item` rather
      than `driver_conduct_complaint`, because the resolution path (reconnect
      rider and driver) is the same regardless of fault, and that resolution
      path is what reply-drafting actually needs to get right.
    - "I want a refund" was folded into `billing_dispute` rather than kept
      separate, since in the sampled data it was always attached to a
      specific bad trip/charge, never a standalone category.
    - `driver_account_onboarding` stayed separate from `general_inquiry`
      because drivers, not riders, send it, with a background-check/payout
      resolution pattern that has nothing to do with a rider's trip.
16. **Only `safety_incident` is on `ALWAYS_ESCALATE_INTENTS`** -- not
    `account_access_or_security` too, even though both are high-stakes.
    `billing_dispute`, `lost_item`, and `account_access_or_security` are all
    common and usually resolvable without a human (see the sampled data), so
    they're deliberately routed through the normal
    confidence/anger-keyword/repeated-turns rules instead of a blanket
    escalation -- otherwise "escalate" would fire on a large fraction of all
    traffic and stop meaning anything. Instead, the anger-keyword list was
    extended with account-security terms ("hacked", "impersonat", "banned",
    "disabled my account") plus a couple of severe-harassment terms
    ("kidnap", "sexually harass") that appeared verbatim-ish in the real
    sample, so the worst `account_access_or_security` and
    `driver_conduct_complaint` cases still escalate via that rule even though
    the intent itself isn't blanket-escalated.
