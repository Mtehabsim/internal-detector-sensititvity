# Provenance scripts (reference only)

These scripts produced three inputs that the pipeline reads from `results/`:

| Script | Produced |
|---|---|
| `run_saa_multibehaviour.py` | `results/saa_multibehaviour/<model>_prompts.json` (SAA-850: each AdvBench behaviour in the released SAA template with its released suffix, plus the no-suffix and control arms), the greedy 256-token completions, and the HarmBench judgments in `<model>_repaired_judged.jsonl` |
| `run_saa_easr.py` | helpers used by the script above (HarmBench judge loading and the released SAA records) |
| `score_strongreject.py` | `results/strongreject/<model>_repaired.json` (StrongREJECT fine-tuned evaluator scores) |

They are kept as they were run, for transparency. They import modules from an earlier research codebase that is
not part of this repository, so they do not run on their own. Local paths are replaced with placeholders such as
`<HF_HOME>`.

Completions are **not** included in this repository. Only the per-prompt judge labels and scores are.
