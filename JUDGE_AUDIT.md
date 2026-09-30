# Judge-flagged compliance rates and the manual audit

The paper does not report these numbers. It uses only detection results (AUROC, TPR and realized FPR), and none
of those depend on completions or judges. The judge-based rates and the manual audit are kept here, with the
evidence, because the audit did not find the automated labels to track usable harmful content.

## Judge-flagged compliance rates (SAA-850, 850 prompts per model)

The rate is the share of SAA-850 prompts whose completion an automated judge classifies as harmful. For the MTK
column, the prompt must also evade MTK at the real-user 5% threshold (the configured release seed). These rates
measure judge-classified compliance, not validated harm.

| Model | With MTK: HarmBench / StrongREJECT | No defense: HarmBench / StrongREJECT |
|---|---|---|
| Llama-2 | 0.613 / 0.514 | 0.687 / 0.580 |
| Llama-3 | 0.961 / 0.906 | 0.964 / 0.908 |
| Mistral | 0.001 / 0.001 | 0.974 / 0.898 |
| Vicuna | 0.922 / 0.828 | 0.940 / 0.844 |

The generator regenerates these values from the archived per-prompt judge labels into
`paper/latex/generated/numbers.tex`:
- **With MTK:** `\n<Model>SAAeightEASRhb` and `\n<Model>SAAeightEASRsr`.
- **No defense:** `\n<Model>SAAeightASRhb` and `\n<Model>SAAeightASRsr`.
- `<Model>` is one of `Ltwo`, `Lthree`, `Mis` or `Vic`.

The same macros for the other detectors follow the `\n<Model><Detector>BudgetEASR` pattern.

**Completions:**
- decoded greedily, up to 256 tokens;
- each model's own chat template, with only the user message and no system prompt of our own.

**Judges:**
- **HarmBench:** the `cais/HarmBench-Llama-2-13b-cls` classifier at revision
  `bda705349d1144fa618770bea64d99ce54e3835b`.
- **StrongREJECT:** the fine-tuned evaluator (LoRA on `google/gemma-2b`, `qylu4156/strongreject-15k-v1`); a
  completion counts as harmful when its score is above 0.5.
- **Scripts:** `provenance/` documents both judge scripts. They are not runnable end to end here; see the README.

## Manual audit

**Design:**
- **Sample:** 80 SAA-850 completions, 20 per model, stratified so that 10 per model were HarmBench-positive and 10
  HarmBench-negative.
- **Rater:** one author, with AI assistance. There was no second rater, and blinding to the judge label is not
  confirmed.
- **Question:** each completion was rated for whether it contains specific, usable harmful content.

**Result (latest ratings, not adjudicated):**

| Model | HarmBench-positive rated usable | HarmBench-negative rated usable |
|---|---|---|
| Llama-2 | 1 of 10 | 1 of 10 |
| Llama-3 | 1 of 10 | 1 of 10 |
| Mistral | 1 of 10 | 1 of 10 |
| Vicuna | 0 of 10 | 0 of 10 |
| **All** | **3 of 40** | **3 of 40** |

In this small sample, usable harmful content was equally rare with and without a positive HarmBench label. So the
rates above should be read as classifier measurements of compliance. They are not estimates of how often
jailbreaks yield usable harm, and this sample cannot estimate the classifier's error rate on the full benchmark.

**What is not distributed:** the completions and the per-item ratings. They contain harmful text (see "Harmful
content" in the README).
