# Pre-registered designs for three extensions (2026-09-26)

Written before any of the new results exist, following an outside review of the submission draft. The analyses
below are fixed now; whatever they show is reported. Anything added later is marked as such, with its date.

## A. More GradSafe reference draws (Llama-2, official code)

- **Runs.** Reference sets `refset5`–`refset28` (24 new draws), made by the unchanged `gradsafe_driver.py
  --stage refsets` code path used for `refset0`–`refset4`: two unsafe prompts from the release's AdvBench and
  MaliciousInstruct pools and two safe prompts from its Dolly pool, drawn with `stable_seed(k, "gradsafe:refset")`.
  Together with the official set and `refset0`–`refset4`, this makes 30 reference sets scored on identical prompts.
- **Per set**, at the text-disjoint real-user 5% threshold (the rule of every other GradSafe number):
  - mean AUROC over the twelve families;
  - SAA AUROC and SAA detection (TPR);
  - the worst-family TPR (families of 50 or more prompts);
  - the realised false-positive rate.
- **Reported:**
  - min, median and max over the 30 sets of mean AUROC and SAA TPR;
  - how many sets detect SAA below 0.2 and above 0.5;
  - the Spearman correlation across sets between mean AUROC and SAA TPR.
- **Reference variation against sampling noise.**
  - Bootstrap: B = 2,000, seed 20260926. Each replicate resamples the calibration prompts and the 500 SAA prompts
    once, shared by all sets (paired, as in v3), and recomputes every set's threshold.
  - Statistic: the between-set standard deviation of the point SAA TPR, divided by the root mean within-set
    bootstrap variance.
  - Decision rule: if the ratio exceeds 2, variation across reference sets is reported as exceeding sampling noise.
  - Output: `results/uncertainty/gradsafe_refsets_v4.json`. v3, the six-set pairwise analysis with Holm, is kept
    unchanged. The paper replaces the six-set pairwise sentence with the 30-set result (a pairwise test over 435
    pairs would say little).

## B. AUROC, partial AUROC and detection at the operating point (no new runs)

- **Points:** every detector (5), model (4) and release family of 50 or more prompts, on the matched prompts, at the
  text-disjoint real-user calibration (the setting of Table VI).
- **Per point:**
  - AUROC against the 500 evaluated benign prompts;
  - standardised partial AUROC over FPR in [0, 0.05] (McClish; `sklearn.metrics.roc_auc_score(max_fpr=0.05)`);
  - TPR at the real-user 5% threshold.
- **Reported:**
  - Spearman rho of AUROC with TPR@5% and of partial AUROC with TPR@5%, over all points and over the model-internal
    detectors only;
  - the number of points with AUROC >= 0.85 and TPR < 0.2, and how many of those have partial AUROC above 0.75;
  - Llama-3 MTK SAA's partial AUROC.
- **Framing fixed now.** Partial AUROC is computed on the evaluation's own benign prompts, so it cannot reflect which
  benign data set the threshold. The Dolly and real-user thresholds of Table I give the same partial AUROC but
  different realised FPRs. This is stated whatever the correlations show.
- **Output:** `results/rule_test/pauc_vs_tpr.json`.

## C. A second real-user benign distribution (WildChat)

Written before any WildChat prompt was selected or scored.

- **Source.** `allenai/WildChat-1M` (ODC-BY) at revision `7d6490e462285cf85d91eabea0f9a954fbddcd1f`, file
  `data/train-00000-of-00014.parquet` only. These are real users' conversations with a ChatGPT front end, a
  deployment different from ToxicChat's (the Vicuna demo).
- **Benign pool.**
  - Conversations with `language == "English"`, `toxic == False` and `redacted == False`.
  - The prompt is the first user turn, stripped, and must be 10 to 2,000 characters long.
  - Deduplicate on normalised text (lower case, whitespace collapsed), keeping the first occurrence in file order.
  - Remove any prompt whose normalised text occurs in the release's ToxicChat benign files (all four models) or in
    its Dolly pool.
  - WildChat's toxicity flags are automatic (OpenAI moderation and Detoxify), not human labels like ToxicChat's.
    This is stated as a limitation.
- **Sample.** `random.Random(20260926).sample(pool, 1500)`: the first 1,000 are the WildChat calibration set, the
  last 500 the WildChat evaluation set (the sizes of the ToxicChat sets).
- **Scores.** Every detector as in the matched evaluation, each with the consistency check of the Dolly runs (the
  release test set, or 40 matched benign prompts, is rescored and must match the saved scores):
  - MTK: the release's configured bank and forest;
  - the probe, HiddenDetect and windowed perplexity: as in `dolly_other_detectors.py`;
  - GradSafe: official code on Llama-2, the port elsewhere, as in `gradsafe_dolly.py`.
- **Pre-registered comparisons.** For each detector and model at the 5% budget, the false-positive rate realised on
  each evaluation set by the threshold from each calibration set:
  - Dolly threshold (text-disjoint split v2) on WildChat evaluation. The main test: does curated calibration
    overshoot on a second real-user distribution?
  - ToxicChat threshold (text-disjoint) on WildChat evaluation, and WildChat threshold on ToxicChat evaluation:
    transfer between the two real-user distributions.
  - WildChat threshold on WildChat evaluation: the in-distribution check.
  - Also the mean detection over the twelve families under the WildChat and ToxicChat thresholds.
- **Headline statistics:**
  - the Dolly-threshold FPR on WildChat, as a range over detectors and models, and the ratio to 5% for MTK;
  - the number of detector-model pairs whose Dolly threshold exceeds 5% on WildChat;
  - the ranges of the two cross-real-user FPRs.
- **One secondary sensitivity check, fixed now.** WildChat is known to contain jailbreak attempts, which would count
  as false positives here. The WildChat evaluation FPRs are recomputed with any prompt containing (case-insensitive)
  "jailbreak", "do anything now", "developer mode", "ignore all previous", "ignore previous instructions", "stay in
  character" or the word "DAN" removed. Thresholds are not changed.
- **Reporting.** Whatever these show is reported, including if ToxicChat thresholds also overshoot on WildChat
  (then the claim becomes: calibration does not transfer between benign distributions, curated or real).
- **Distribution.** No WildChat text goes into the public artifact: prompts are kept by conversation hash and
  SHA-256, and are rebuilt from the pinned file.

---

## Outcomes (added 2026-09-26, after the runs; the designs above were not changed)

- **A (GradSafe, 30 sets).**
  - Mean AUROC 0.848–0.941; SAA TPR 0.012–0.962 (median 0.160; 19 sets below 0.2, 5 above 0.5).
  - Worst-family TPR at most 0.048 in every set. Spearman(mean AUROC, SAA TPR) = +0.32.
  - Between-set SD 0.278 against an RMS within-set bootstrap SD of 0.182: ratio 1.53, below the pre-set 2. So the
    variation across reference sets is **not** reported as exceeding sampling noise; the paper calls it comparable
    to the sampling uncertainty.
  - A variance decomposition or any other test that might pass was deliberately not substituted.
- **B (partial AUROC).**
  - 175 points (140 model-internal). Spearman with TPR@5%: AUROC 0.91, partial AUROC 0.97 (model-internal: 0.91,
    0.96).
  - 18 points have AUROC >= 0.85 and TPR < 0.2; none has partial AUROC above 0.75. Llama-3 MTK SAA: 0.49.
- **C (WildChat).**
  - Pool 20,833 of 59,857 conversations. Dolly thresholds exceed 5% on WildChat in 19 of 20 detector-model pairs
    (0.016–0.368; MTK 2.7–7.4x); the same four detectors are above it on every model.
  - Jailbreak-marker check: 10 evaluation prompts removed, 0.016–0.357.
  - ToxicChat thresholds on WildChat 0.024–0.080; WildChat thresholds on ToxicChat 0.012–0.090; in-distribution
    0.032–0.060.

---

## D. The same noise test for MTK's bank-only redraws (added 2026-09-26, before computing it)

An outside reviewer asked why the factor-of-2 test was applied to GradSafe (where it failed) and not to MTK. This
applies the identical test to MTK. It was written before any of it was computed; whatever it shows is reported.

- **Draws.** For each model, exactly the draws of Table IV's lower block (`make_final_tables.by_seed(..., "bank_only")`):
  the bank is redrawn, and the evaluation sample and forest stay at the configured seed. 50 or 51 draws per model,
  with per-prompt scores saved; the configured seed's as-shipped scores stand in for its own bank.
- **Statistic.**
  - SAA detection at the text-disjoint real-user 5% threshold. The recomputed point value must equal each draw's
    recorded `sweep_v2` metric, or the run stops.
  - Between-draw SD of the point TPR, divided by the root mean within-draw bootstrap variance: B = 2,000, seed
    20260926; each replicate resamples the calibration prompts and the 500 SAA prompts once, shared by all draws of
    a model, and recomputes every threshold.
- **Decision.**
  - Ratio above 2: the variation across banks is reported as exceeding sampling noise.
  - Otherwise it is reported as comparable, and the paper's per-model claim is softened as GradSafe's was.
  - If the RMS within-draw SD is below 0.005 (SAA detected at about 0 in every draw), the model has no variation to
    test; this is reported as such, not as a pass.
- **Output:** `results/uncertainty/mtk_bank_noise.json` (`mtk_bank_noise.py`).

- **D (MTK bank-only draws, outcome added after the run).**
  - Every recomputed point SAA TPR equals its sweep_v2 record.
  - Mistral: SAA TPR 0.000–1.000 (median 0.099), between-draw SD 0.279, RMS within-draw SD 0.164, ratio 1.69:
    **below 2**.
  - Llama-2: 0.000–0.114, ratio 0.25.
  - Llama-3 (0.000–0.002) and Vicuna (0.000–0.026): RMS within-draw SD below 0.005, so nothing to test.
  - Per the rule, MTK's random-draw claim is softened as GradSafe's was: the swings are real (the prompts are
    identical across draws) but comparable to prompt-sampling noise. The deliberate-anchor and formatting-row
    interventions are unaffected.
