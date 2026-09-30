# Third-party material

This repository is licensed under CC BY-NC 4.0 (`LICENSE`), except for the third-party material below, which
remains under its own terms. Licenses are as each source declares them (GitHub and Hugging Face metadata and the
sources' own license files, checked 2026-09-26).

## Prompt texts: not included

No third-party prompt text is distributed. Wherever a result refers to one, it holds a SHA-256 of the exact text
(or of the excerpt the analysis used) instead. `scripts/release/rebuild_texts.py` rebuilds the texts on your
machine from the sources that `make setup` fetches, and checks every hash:

```bash
make setup
python scripts/release/rebuild_texts.py              # SAA-850 prompt files, needed by the GPU stages
python scripts/release/rebuild_texts.py --excerpts   # also every hashed excerpt, into restored_texts/
python scripts/release/rebuild_texts.py --wildchat   # also the WildChat benign set (downloads 231 MB)
```

The rebuilt files are ignored by git (`.gitignore`); please do not publish them.

| Result files | What is hashed | Rebuilt from | Original source and declared license |
|---|---|---|---|
| `results/saa_multibehaviour/<model>_prompts_index.json` (stands in for `<model>_prompts.json`) | 13,600 SAA-850 prompts: 850 per model in each of four arms (`repaired`; `nosuffix`; `shufsuffix`, the record's own suffix tokens permuted; `randsuffix`, length-matched random tokens), plus the whole file's hash | the MTK release's `datasets/<model>_test/saa_1.json` (each record's goal, target and adversarial suffix) and the SAA template recovered from it; the control arms also need each model's tokenizer | goals from AdvBench (520 records; [llm-attacks](https://github.com/llm-attacks/llm-attacks), MIT) and HEx-PHI (330 records; Qi et al., ICLR 2024; [LLM-Tuning-Safety/HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI), under its own terms of use), with the targets as shipped in the MTK release; the Simple Adaptive Attack template ([llm-adaptive-attacks](https://github.com/tml-epfl/llm-adaptive-attacks), MIT); suffixes as shipped in the MTK release ([Rookie143/mtk](https://github.com/Rookie143/mtk), no license declared) |
| `results/release_pipeline/mechanism_v2/<model>.json` (`top_interleaving_anchors[].text80_sha256`) | the first 80 characters of the ten malicious anchors that most often precede benign neighbours, per model and population | the release's anchor pools | AdvBench (MIT), MaliciousInstruct ([Princeton-SysML/Jailbreak_LLM](https://github.com/Princeton-SysML/Jailbreak_LLM), no license declared), PKU-SafeRLHF prompts ([PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), CC BY-NC 4.0) |
| `results/rule_test/anchor_length*.json` (`top5[].text40_sha256`) | the first 40 characters of the five highest-share anchors | the release's anchor pools | as above |
| `results/gradsafe/summary_refset<k>.json` (`reference`) | the two unsafe and two safe prompts of each redrawn GradSafe reference set | the release's AdvBench, MaliciousInstruct and Dolly pools | as above; databricks-dolly-15k ([databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), CC BY-SA 3.0) |
| `results/METHODS_MANIFEST.json` (`gradsafe.reference_prompts`) | GradSafe's four official reference prompts | GradSafe's code (`WITH_GRADSAFE=1 make setup`) | [xyq7/GradSafe](https://github.com/xyq7/GradSafe), Apache-2.0 |
| `results/dolly_calibration/split_v2/<model>.json` (`dropped[].text80_sha256`) | 7 Dolly prompts removed from the calibration split as repeats or as texts that also occur in ToxicChat | the release's Dolly pool | databricks-dolly-15k, CC BY-SA 3.0 |
| `results/wildchat/selection.json` (`prompt_sha256`, with WildChat's own `conversation_hash`) | 1,500 benign first user turns (the second real-user distribution) | one pinned file of WildChat-1M, by the selection rules in `mtkaudit/wildchat.py` | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), ODC-BY |
| `results/release_pipeline/official_runs/vicuna/*_results_detail.csv` (`prompt_sha256`) | every prompt the release's Vicuna runner scored | the release's `datasets/vicuna_test/` | ToxicChat ([lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat), CC BY-NC 4.0) and the release's attack sets (no license declared) |
| `results/uncertainty/toxicchat_labels.json` (`labels`: prompt SHA-256 → label) | ToxicChat's label (toxic, human-annotated non-toxic, automatic non-toxic, or not in its test split) for each evaluated and calibration benign prompt; no text | the release's ToxicChat benign file, matched to ToxicChat's 0124 test split at revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8` (`scripts/calibration/toxicchat_label_check.py`) | ToxicChat ([lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat), CC BY-NC 4.0) |

Every other result file holds scores, labels, indices, hashes or aggregate statistics. The benign prompts the
release uses (ToxicChat; its Dolly, Alpaca-format and OR-Bench files) are referenced by index only. For reference,
their sources declare: Stanford Alpaca data CC BY-NC 4.0; OR-Bench ([bench-llm/or-bench](https://huggingface.co/datasets/bench-llm/or-bench)) CC BY 4.0.

## Code

| Where | What | Source | License |
|---|---|---|---|
| `src/mtkaudit/gradsafe/official.py` (`official_score`) | GradSafe's `cos_sim_toxic` loop, changed to score one prompt | [xyq7/GradSafe](https://github.com/xyq7/GradSafe) | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| `src/mtkaudit/baselines.py` (`REFUSAL_LST`) | HiddenDetect's list of 20 refusal tokens | [leigest519/HiddenDetect](https://github.com/leigest519/HiddenDetect) | none declared |
| `src/mtkaudit/chat_template_vicuna_v1_1.jinja` | the Vicuna v1.1 conversation format, including its default system message | [lm-sys/FastChat](https://github.com/lm-sys/FastChat) | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |

The MTK release has no license. Its code and data are not copied here; the harness imports them from the clone that
`scripts/setup_release.sh` makes. GradSafe's repository is likewise fetched, not copied.
