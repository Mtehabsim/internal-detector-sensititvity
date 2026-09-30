# Model-Internal Jailbreak Detection under Deployment-Like Data — code and evidence

Code and archived per-prompt evidence for *Model-Internal Jailbreak Detection under Deployment-Like Data:
Calibration, Reference Sensitivity, and Low-FPR Coverage*, submitted to IEEE BigData 2026 (Special Session on
Privacy and Security of Big Data), by Mohammad Tehabsim, Basil Al-Housani (equal contribution), Nour Alhussien and
Phung Lai, University at Albany, SUNY.

The paper studies model-internal jailbreak detectors: MTK (Manifold Trajectory Kinetics, USENIX Security 2026),
GradSafe and an LLM adaptation of HiddenDetect, with a supervised linear probe and a windowed-perplexity filter,
all scored on identical prompts on four chat models. MTK is run from the authors' latest public release (as of
September 2026; [github.com/Rookie143/mtk](https://github.com/Rookie143/mtk) at commit
`c5e2f18d913bd18b09b9357159543487beb2ce75`) and serves as the worked case for the reference-bank experiments.
Every number in the paper is a macro that `scripts/paper/make_final_tables.py` writes from the files under
`results/`, and `scripts/paper/verify_final_claims.py` recomputes them independently.

The central distinction is that calibration data move the threshold, while reference data change the scores.
MTK is the primary audited system; GradSafe is official on Llama-2 and ported on the other models. HiddenDetect
is an adaptation and the probe is a simple supervised baseline, not independent official reproductions.
The paper separates observed reference effects from the stronger, unsupported claim that random-reference
variation generally dominates prompt-sampling uncertainty.

## Quick start: check every number against the archive (CPU, no model weights)

```bash
make install     # pip install -c requirements.txt -e ".[test]"  (package, tests, at the versions we used)
make setup       # fetch the MTK release at c5e2f18 into third_party/ and prepare scratch/ (safe to rerun)
make test        # regression tests, seconds
make gate        # the harness reproduces the release's own runners, prompt by prompt
make tables      # paper/latex/generated/: numbers.tex, every table and figure (fails on any missing input)
make verify      # independent recomputation of the claims: must report 0 failures
make selftest    # corrupts inputs in memory; every corruption must be caught
make analysis    # every CPU analysis, rerun from the archived scores; each reproduces its file exactly
```

These targets read the **archived** runs under `results/`. They never look at a run you make yourself; see
[Checking a fresh run](#checking-a-fresh-run). No third-party prompt text is distributed: results hold a SHA-256
in place of each text, and `make texts` (`scripts/release/rebuild_texts.py`) rebuilds the texts locally from the
fetched release and checks every hash (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The targets above
do not need them; the GPU stages do. Each target is one or more `python scripts/<area>/<script>.py`
calls (see the `Makefile`). The paper's LaTeX source is not distributed: `make tables` writes every number,
table and figure it uses to `paper/latex/generated/` (not tracked), and `scripts/paper/paper_figures.py` (part of
`make tables`) draws Figs. 2-4 at their printed size.

Three analyses write a new file and refuse to overwrite an existing one, so they are not part of `make analysis`:
`scripts/analysis/goal_overlap.py` (evaluation goals equal to an AdvBench anchor, and detection with and without
them), `scripts/calibration/toxicchat_label_check.py` (ToxicChat's own labels for the evaluated benign prompts; it
downloads the pinned ToxicChat file and saves only hashes and labels) and `scripts/paper/per_family_heatmap.py`
(per-family detection for every detector and model, `results/matched_detectors/per_family_heatmap.pdf`). Move the
existing output aside to rerun one.

Local verification on 2026-09-29: 22 tests pass, all four archived equivalence gates pass, 1,480 numerical checks
report zero failures, and all three corruption self-tests are detected. The generator emits 1,976 macros with
no missing inputs, identical to those in the submitted paper. This includes the non-SAA coverage counts: at nominal 5%, 14/16 internal-detector–model pairs
have a non-SAA family with at least 50 prompts and TPR < 0.2; 9/16 also have AUROC ≥ 0.85. These are dependent
descriptive counts over the tested implementations. No GPU experiments were rerun for this editorial revision.

Tested with Python 3.12.3 and the versions pinned in `requirements.txt` (torch was NVIDIA's 2.10 build, nv25.11;
a CPU build of torch 2.4 or later suffices for the targets above). `pyproject.toml` alone allows newer versions,
which may not reproduce the archived outputs exactly; `make install` applies the pins as constraints.

## Layout

```
src/mtkaudit/            the shared library (import it; nothing here is a script)
  paths.py               repository root and environment configuration
  release.py             harness around the MTK release: pinned copy, seeded sampling, text-disjoint calibration,
                         scoring through the release's own extraction, rank and forest code
  mechanism.py           interleaving counts behind MTK's rank feature, anchor shares
  matched.py             the matched prompt set every detector is scored on
  gradsafe/official.py   GradSafe's official code on Llama-2 (single-GPU adapter, fast scorer)
  gradsafe/port.py       GradSafe ported to the other chat templates
  baselines.py           HiddenDetect (LLM adaptation), windowed perplexity, model loading
  stats.py               bootstrap helpers
  results.py             loaders for the archived results
  caches.py              exact comparison of feature caches
  runs.py                comparison of a fresh harness run with the archived one
  texts.py               hashes in place of third-party texts, and the SAA-850 rebuild
  wildchat.py            the WildChat benign set: pinned source and selection rules
  manifest.py            as-run and distributed hashes in the methods manifest
  cli.py                 command-line helpers (--help for option-less scripts, absolute path options)
scripts/                 command-line programs, grouped by what they study; each imports only mtkaudit
  setup_release.sh       fetch the release (and GradSafe) and prepare the working copy
  release/               run the release, the equivalence gate, compare fresh and archived runs, recompute
                         metrics, rebuild texts, rebuild and compare caches
  calibration/           Dolly vs real-user thresholds, text-disjoint split, equal-size control, GCG interval,
                         WildChat (second real-user distribution), ToxicChat's own labels
  mechanism/             interleaving, formatting-row swap, placed anchors, Mistral offset, anchor-length check
  detectors/             GradSafe (official and port), HiddenDetect and perplexity, the linear probe
  analysis/              bootstraps, GradSafe reference sets (6 and 30), the AUROC rule, partial AUROC, the
                         Gaussian reference, goal overlap with the AdvBench anchors
  paper/                 generator of every number, table and figure; Figs. 2-4 as printed; per-family heatmap;
                         independent verifier; methods manifest
tests/                   regression tests (path handling, anchor shares, manifest, fresh-run comparison)
results/                 per-prompt scores and records for every run in the paper (about 120 MB)
paper/latex/generated/   written by `make tables`: numbers.tex, tables and figures (not tracked)
provenance/              scripts that produced the SAA-850 prompts and judge labels (reference only; see its README)
  analysis_design/       the dated design document of the WildChat, partial-AUROC, GradSafe and noise analyses,
                         with the SHA-256 recorded before each was run
JUDGE_AUDIT.md           judge-flagged compliance rates and the manual audit of completions (not in the paper)
LICENSE                  CC BY-NC 4.0; LICENSES/ holds the licenses of third-party code (THIRD_PARTY_NOTICES.md)
```

Without installing, set `PYTHONPATH=src`. Configuration comes from environment variables (`MTK_RELEASE_CHECKOUT`,
`MTK_RELEASE_SCRATCH`, `MTK_MODEL_DIR`, `GRADSAFE_DIR`, and `MTKAUDIT_ROOT` for a non-editable install), with
defaults inside the repository; `src/mtkaudit/paths.py` documents each. Path options on the command line are
relative to the directory you run the command from.

## What this artifact supports

| Level | What | Needs |
|---|---|---|
| Verification of the archive | regenerate every number, table and figure from the archived per-prompt scores and judge labels; the independent verifier; the equivalence gate against the archived runner CSVs; every CPU analysis | CPU, the MTK release (fetched by `make setup`) |
| Detector re-scoring | rerun the MTK release, GradSafe, HiddenDetect, perplexity, the probe, the Dolly calibration and the mechanism experiments, and compare the fresh outputs with the archive | GPUs, the four checkpoints, GradSafe's code |
| Not included end to end | generating the SAA-850 completions and judging them with HarmBench and StrongREJECT | `provenance/` documents these scripts, but they depend on modules from an earlier codebase that are not distributed; the archived judge labels are used instead |

## Rerunning the detectors (GPU)

The archived scores make the GPU runs unnecessary for checking the paper. To regenerate them you need the four
checkpoints (`meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Meta-Llama-3-8B-Instruct`,
`mistralai/Mistral-7B-Instruct-v0.2`, `lmsys/vicuna-7b-v1.5`) in one directory, under the names the release uses
(`llama2`, `llama3`, `mistral_7b`, `vicuna-7b-v1_5`). We used four 80 GB A100s, one model per GPU.

```bash
pip install -c requirements.txt -e ".[gpu]"
export MTK_MODEL_DIR=/path/to/checkpoints
WITH_GRADSAFE=1 bash scripts/setup_release.sh
python scripts/release/rebuild_texts.py          # SAA-850 prompts, all four arms (uses the tokenizers)
pip install -c requirements.txt -e ".[data]"      # for the WildChat set: pyarrow, huggingface_hub
python scripts/release/rebuild_texts.py --wildchat
# 1. the release's own runners, as shipped, into scratch/official_runs/
cd scratch/release/llm
for m in llama2 llama3 mistral; do python mtk_$m.py --output-dir ../../official_runs/$m; done
bash detection_vicuna.bash --output-dir ../../official_runs/vicuna
cd -
# 2. the harness, per model (llama2, llama3, mistral, vicuna), into rerun/<model>/ beside the archive
python scripts/release/release_driver.py --model llama2 --seeds 0-49 --out rerun/llama2
# 3. the other experiments, per model; these write to their archived locations under results/ (see below)
python scripts/mechanism/release_mechanism.py --model llama2
python scripts/mechanism/release_markup.py --model llama2                 # llama2, mistral, llama3
python scripts/mechanism/release_sentinel.py --model llama2
python scripts/detectors/matched_detectors.py --model llama2 --stage score # HiddenDetect + perplexity
python scripts/detectors/probe_baseline.py --model llama2
python scripts/calibration/dolly_calibration.py --model llama2
python scripts/calibration/dolly_other_detectors.py --model llama2
python scripts/calibration/gradsafe_dolly.py --model llama2
python scripts/calibration/wildchat_mtk.py --model llama2                 # WildChat: after rebuild_texts.py --wildchat
python scripts/calibration/dolly_other_detectors.py --model llama2 --source wildchat
python scripts/calibration/gradsafe_dolly.py --model llama2 --source wildchat
# once:
python scripts/detectors/gradsafe_driver.py --stage validate
python scripts/detectors/gradsafe_driver.py --stage score
python scripts/detectors/gradsafe_driver.py --stage refsets --refsets 0-28
python scripts/detectors/gradsafe_port.py --stage check
python scripts/detectors/gradsafe_port.py --stage score --model llama3    # and mistral, vicuna
python scripts/mechanism/mistral_offset.py
python scripts/mechanism/anchor_length_check.py
python scripts/paper/write_methods_manifest.py                           # needs the models' tokenizers
```

`make setup` and `scripts/setup_release.sh` keep an existing working copy whose release files match the pinned
commit, together with the feature caches the runs store inside it; `--reset` replaces it. Each script's docstring
states its inputs, outputs and design. `results/archive/` holds the SHA-256 sums of the caches and of the release
copy we used.

## Checking a fresh run

Verification of the archive and verification of a fresh run are separate steps:

```bash
# the harness: every fresh record (scores, bank, metrics) against the archived one
python scripts/release/compare_runs.py --fresh rerun
# the release's runners: the fresh harness scores against the fresh runner CSVs
python scripts/release/release_gate.py --driver rerun --official scratch/official_runs
```

`compare_runs.py` compares per-prompt scores (tolerance `--tol`, default 1e-6) and every metric (default 1e-9)
with the archived text-disjoint records, and writes `rerun/compare_report.json`. Without `--out`,
`release_driver.py` instead resumes from the archived records in `results/release_pipeline/<model>/` and adds
only missing ones.

The scripts in step 3 above write to the archived locations under `results/`. Run them in a separate clone (or
after committing), then `git status results/` and `git diff --stat results/` show which archived outputs
changed, and `make tables verify` rebuilds and checks the paper from the fresh files.

## Notes

- **Relation to the scripts as run.** The paper's numbers were produced by the same code as flat research
  scripts. For release it was reorganised: shared code moved into the `mtkaudit` package, and scripts were grouped
  by purpose and import only from the package. Path constants became environment variables, and file paths inside
  archived records were replaced with placeholders (`$MTK_RELEASE_SCRATCH`, `third_party/mtk`, `$MTK_MODEL_DIR`).
  Apart from imports, command-line handling, hashes in place of third-party texts and the fixes listed next,
  function bodies are unchanged, and the
  archive checks above reproduce every archived output exactly. `results/METHODS_MANIFEST.json` records the
  as-run script hashes (`scripts_sha256`, kept when the manifest is regenerated) and the hashes of every file
  distributed here (`distributed_scripts_sha256`: code, templates, tests, build and dependency files).
- **Fixes after the runs.** `release_driver.py --out` and the cache scripts' directory options are now resolved
  where the command runs (they used to be resolved after the harness changes into the release's working copy).
  `anchor_length_check.py` divided each anchor's count by the number of anchors instead of query-layers; its
  stored shares were too small by a constant factor per model, which left every correlation, ordering and median
  it reports unchanged. The corrected output is `results/rule_test/anchor_length_v2.json`, which the paper reads and
  the verifier recomputes; `anchor_length.json` is kept.
- **Harmful content.** No model completions and no prompt texts are included, only per-prompt judge labels,
  scores and hashes. The judge-based compliance rates and the manual audit are summarised in
  [`JUDGE_AUDIT.md`](JUDGE_AUDIT.md). SAA-850 puts each of its 850 requests (520 from AdvBench, 330 from
  HEx-PHI) into the released SAA template with that record's released suffix; `rebuild_texts.py` rebuilds it
  from the release.
- **Third-party material.** The MTK release, GradSafe, the models and the datasets belong to their authors. This
  repository fetches the release and GradSafe instead of copying them, replaces every third-party prompt text with
  its hash, and contains small pieces of third-party code. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
  lists each item, where it comes from and the license its source declares.

## License

[CC BY-NC 4.0](LICENSE) (Creative Commons Attribution-NonCommercial 4.0 International): you may use, share and
adapt this code and evidence for non-commercial purposes, provided you give credit, for example by citing the
paper below. Commercial use is not permitted. Third-party material remains under its own terms (see
`THIRD_PARTY_NOTICES.md`); in particular, GradSafe's scoring loop in `src/mtkaudit/gradsafe/official.py` and the
Vicuna template are under Apache-2.0 (`LICENSES/Apache-2.0.txt`).

## Citation

If you use this code or evidence, please cite:

```bibtex
@misc{highauroc2026,
  title  = {High AUROC, Low Detection: Calibration and Reference Sensitivity in Model-Internal
            Jailbreak Detectors},
  author = {Mohammad Tehabsim and Basil Al-Housani and Nour Alhussain},
  year   = {2026},
  note   = {Submitted to the IEEE International Conference on Big Data (BigData 2026). Code and evidence:
            \url{https://github.com/Mtehabsim/internal-detector-sensititvity}}
}
```
