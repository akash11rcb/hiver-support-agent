# Report

*Brand: **Uber_Support** (Twitter/X handle @Uber_Support). Chosen over AppleSupport and AmazonHelp for its genuine escalation stakes (safety incidents, payment disputes) — see decision_log.md #14.*

## Problem framing

The task: given an incoming customer message to @Uber_Support, (1) classify it into one of a small set of intents derived from this brand's actual traffic, (2) draft a reply grounded in this brand's own historically-resolved conversations, and (3) decide whether to auto-handle or escalate to a human, with a stated reason.

**What "good" means here, concretely.** This brand's traffic mixes routine questions (how do I remove a vehicle from my account) with genuinely high-stakes reports (suspected drunk driving, account takeover with real money moving, discrimination allegations). "Good" means: the classifier sorts these into a taxonomy read directly off real messages rather than a generic template; the drafted reply doesn't invent policy details Uber never actually offered in a past resolution; and — the part with real consequences if wrong — the escalation decision routes anything touching physical safety, financial fraud, or discrimination to a human, while not burning human time on the large fraction of traffic that's routine. Of the two error directions on escalation, missing a real escalation is far more costly than an unnecessary one, so the failure analysis below weights false negatives more heavily than false positives.

**What we chose not to build (scope).** No multi-turn dialogue state tracking — each message is classified on its opening turn only, with the thread's turn count used purely as an *escalation* signal (a customer stuck in back-and-forth without resolution), not a reason to re-classify mid-conversation. No actual sending — this system drafts and decides; dispatch is a separate concern. No multilingual handling — the taxonomy and prompts assume English (the golden set surfaced at least one French-language row, flagged as a known gap rather than silently mishandled). No PII handling beyond what the dataset already anonymizes.

**Scale.** 56,270 total tweets for this brand, 7,488 reconstructed conversation threads (46% multi-turn), a 9-intent-plus-`other` taxonomy built from reading ~120 real sampled messages (see decision_log.md #15), and a 200-row golden set built with human review of every row (4.5% intent disagreement with the LLM's draft suggestions — evidence the labels weren't rubber-stamped, and a number this report leans on again below).

## Results vs. baselines

All three methods evaluated on the same 200-row golden set; the two baselines use 5-fold cross-validation over the golden set itself so every row gets a fair out-of-fold prediction (decision_log.md #6).

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (majority class) | 0.290 | 0.045 |
| Simple (TF-IDF + Logistic Regression, 5-fold CV) | 0.540 | 0.410 |
| **LLM classifier (zero-shot)** | **0.955** | **0.954** |

The trivial baseline just always guesses `billing_dispute` (29% of the golden set) and its macro-F1 collapses to near-zero, since it never predicts the other 9 classes at all — this is the floor, not a real competitor. The TF-IDF + logistic regression baseline roughly doubles accuracy on lexical cues alone, but still whiffs entirely on rarer classes (`safety_incident` and `driver_account_onboarding` both score 0 F1 in the per-class breakdown) — it has no way to generalize beyond vocabulary it saw enough of. The LLM classifier's jump to 95.5%/0.954 is real and substantial, but **this number needs a caveat that gets its own section below** — it is not the fully independent number it looks like.

**Escalation decision:**

| Metric | Value |
|---|---|
| Accuracy | 0.845 |
| Precision (escalate) | 0.667 |
| Recall (escalate) | 0.652 |
| F1 (escalate) | 0.659 |
| Confusion matrix [[TN, FP], [FN, TP]] | [[139, 15], [16, 30]] |

84.5% accuracy sounds solid until you notice 77% of the golden set legitimately shouldn't escalate — so a policy that *never* escalated anything would already score 77% "accuracy" while being useless. The honest number is the F1 on the escalate class: **0.659**, driven by 16 missed escalations (recall 65%) and 15 unnecessary ones (precision 67%). Both error types are examined with real examples in the failure analysis below.

## Reply quality (LLM-as-judge) and judge-human agreement

The LLM judge scored every drafted reply on 4 dimensions plus an overall (decision_log.md #7):

| Dimension | Mean (1-5) |
|---|---|
| Grounded | 4.95 |
| Relevant | 4.94 |
| Tone | 4.925 |
| Actionable | 4.945 |
| **Overall** | **4.925** (4.923 restricted to the 155 auto-handled items only — decision_log.md #10) |

To test whether this judge can be trusted, 40 of the judged replies were also rated blind by a human (1-5 overall, no visibility into the judge's scores) and compared:

| Metric | Value |
|---|---|
| n | 40 |
| Spearman ρ | 0.141 (p = 0.386 — not statistically significant) |
| Quadratic-weighted κ | 0.012 (≈ chance-level agreement) |
| Exact match rate | 15% |
| Within-1-point rate | 65% |

**The judge cannot currently be trusted as a quality signal.** On this same 40-item sample, the human's scores spread meaningfully (six 2s, nine 3s, nineteen 4s, six 5s), while the judge gave **39 of the 40 items a flat 5** and the other a 4. A score that barely varies cannot correlate with anything, which is exactly what the near-zero kappa and non-significant correlation show. See "What's misleading about my headline number" for what this means for the 4.92/5 topline.

## Failure analysis: top 5 failure modes

**1. Escalation policy misses financially serious cases that don't match its keyword list (16 false negatives).** Examples: `id=2979748` ("charged me $126 for 4 rides from the airport I never took and never got receipts emailed"), `id=2710013` ("Sign in I didn't authorize... 4 orders for rides & food have gone through"), `id=1428026` ("charged hundreds of pounds to my account for rides I have not taken in Russia"), `id=1690418` ("This is the 4th time this bs happened and now my account is in the negatives"). None contain the rule engine's literal anger-marker vocabulary, and most are single-turn (so the repeated-turns rule can't fire either) — `default_auto_handle` wins by default. **Hypothesis:** the policy needs a rule keyed to fraud-pattern language ("never took," "didn't authorize," "unauthorized") and/or a detected dollar amount, not a fixed keyword list.

**2. Escalation policy over-triggers on a substring-matching bug (12 of 15 false positives).** The anger-marker "sue" is checked as a substring, so it fires inside completely ordinary words: `id=2412917`'s "had a few **issue**s" and `id=168914`'s "since my was an **issue**" both trip "sue" ⊂ "issue" with zero mention of legal action. Separately, "cancel my" fires on plain, everyday cancellation requests (`id=2650143`, `2253961`, `2574041`, `1136398`) and "worst" fires on generic venting (`id=2028859` "worst tech support team", `2705073`, `2970774`) with no actual severity signal behind it. **Fix:** word-boundary matching instead of containment, and drop or narrow "worst"/"cancel my" from the list.

**3. The LLM classifier over-triggers `safety_incident` on hostile-toned driver complaints that aren't in-ride danger.** `id=701844` (driver retaliation/false-reporting), `id=2255891` (missing window decals), `id=1239082` ("driver stealing from customers") were all classified `safety_incident` when they're really `driver_conduct_complaint`. This is the taxonomy's lowest-F1 class (0.857) largely because of this one pattern — hostile tone plus any driver mention seems to override the actual content. Since `safety_incident` is always-escalate, this specific confusion over-escalates rather than under-escalates, which is the safer direction to be wrong in, but it still costs unnecessary human time.

**4. The classifier reads surface phrasing over the underlying monetary ask.** `id=1273699` ("it's your map software that gave the awful route... want refund") and `id=1600760` ("I only got $50 for my referral instead of $500") were both classified `general_inquiry`; both are `billing_dispute` — there's a specific dispute over money in each, just phrased as a question ("why"/an explanation of a discrepancy) rather than a demand.

**5. Taxonomy gap: no clean bucket for identity/personal-info misuse.** `id=2740887` ("David has signed up using my mobile number") doesn't fit any of the 9 intents well — the LLM defaulted to `other`, and even manual review wavered before settling on `account_access_or_security` as the nearest fit. Worth a dedicated intent if this pattern recurs at scale.

## What is misleading about my headline number?

Three numbers in this report look better than they should on their own:

**"95.5% accuracy, crushing both baselines."** The golden-set intent labels started life as this same model family's own zero-shot draft suggestions (`suggested_intent`), and the human review pass only overrode 4.5% of them. That 4.5% disagreement rate is genuine evidence the labels weren't rubber-stamped wholesale — but it also means the classifier is being graded substantially against its own prior judgment on the other 95.5% of rows, not a fully independent standard established by a human labeling blind. The 9 disagreements found (failure modes #3 and #4 above) are real, defensible errors, so the classifier isn't fake-good — but "95.5%" should be read as "agreement with a lightly-corrected version of its own prior guess," not "accuracy against ground truth a human produced with no exposure to the model's opinion."

**"Reply quality averages 4.92/5."** The judge-human agreement check above is the direct evidence this number can't be trusted: quadratic-weighted kappa of 0.012 is agreement indistinguishable from chance, and the judge scored 39 of 40 sampled replies a flat 5 regardless of whether a human found them excellent or mediocre. The likely mechanism: the drafting prompt already instructs the model to stay polite, on-topic, and non-committal, and the judge — from the same model family — appears to reward exactly that surface behavior rather than whether the underlying issue was actually addressed. The biggest judge-vs-human gaps are telling: the judge gave 5/5 to generic "send us a DM" replies for cases that were actually serious enough that they were *also* missed escalations in failure mode #1 above (`id=2979748`, `2710013`, `1435137`, `1630039`) — a human rated every one of these a 2.

**"84.5% escalation accuracy."** 77% of the golden set legitimately shouldn't escalate, so this number is inflated by class imbalance in a way that makes the policy look better than it is. The F1 on the escalate class (0.659) is the honest number, and it says the policy misses about a third of cases that should go to a human.

## What I'd do next with one more week

- **Fix the two concrete escalation bugs found above:** word-boundary matching for anger markers (kills the "sue"/"cancel my"/"worst" false positives), and a new rule for fraud-pattern language / detected dollar amounts (targets the false negatives). Re-run evaluation to confirm the F1 gain is real, not just moved elsewhere.
- **Replace or fix the judge.** Either use a judge from a genuinely different model family than the drafter (reduces shared-bias risk), or switch from independent absolute scoring to pairwise/ranked comparison (forces discrimination between replies instead of letting every reply cluster at the ceiling). Re-run the judge-human agreement check afterward — if kappa doesn't meaningfully improve, the judge isn't usable as an automated proxy at all and every report of "reply quality" needs to say so explicitly.
- **Get a genuinely blind accuracy number for the classifier:** re-label a fresh batch of golden-set-sized data with zero exposure to `suggested_intent`, then compare to the LLM's zero-shot output afterward, to replace the anchored 95.5% with an honest one.
- **Grow the golden set**, oversampling the intents currently under 10 examples (`driver_account_onboarding`=3, `lost_item`=4, `other`=7, `safety_incident`=9) — per-class F1 on samples this small swings a lot on just one or two examples.
- **Add a basic language-detection gate** rather than silently running English-tuned prompts on non-English input (surfaced by one French-language golden-set row).
- **Revisit retrieval** (TF-IDF was a sandbox-network constraint, not a principled choice — decision_log.md #1) with embeddings, and re-measure grounded-reply quality once the judge itself is trustworthy enough to measure it with.
