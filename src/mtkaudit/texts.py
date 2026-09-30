"""Third-party prompt texts are not distributed. Result files hold a SHA-256 in place of each text, and the texts
are rebuilt from the sources scripts/setup_release.sh fetches (scripts/release/rebuild_texts.py; see
THIRD_PARTY_NOTICES.md).

Formats, as the distributed result files and the scripts that write them use them:
  mechanism_v2/<model>.json   top_interleaving_anchors[]: "text80_sha256" of the anchor's first 80 characters
  anchor_length*.json         top5[]: {"text40_sha256", "length", "share"} (first 40 characters after stripping)
  summary_refset<k>.json      reference: {"unsafe_sha256": [...], "safe_sha256": [...]}
  split_v2/<model>.json       dropped[]: "text80_sha256" of the normalised text's first 80 characters
  METHODS_MANIFEST.json       gradsafe.reference_prompts: {"unsafe_sha256", "safe_sha256", "source"}
  SAA-850                     <model>_prompts_index.json in place of <model>_prompts.json (see saa850_index)
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

ARMS = ("repaired", "nosuffix", "shufsuffix", "randsuffix")
CONTROL_SEED = 20260921            # per record: CONTROL_SEED + index (as run; provenance/run_saa_multibehaviour.py)
MODEL_DIRS = {"llama2": "llama2", "llama3": "llama3", "mistral": "mistral_7b", "vicuna": "vicuna-7b-v1_5"}
GRADSAFE_SOURCE = "GradSafe's code/find_critical_parameters.py (xyq7/GradSafe, Apache-2.0)"
FORMATTING_ROWS = ("[\\list]", "[list]")      # the PKU-SafeRLHF residue rows the paper discusses


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    """The normalisation the Dolly split compares texts under (lower case, whitespace collapsed)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _hashed_anchor(a: dict) -> dict:
    return {k if k != "text" else "text80_sha256": v if k != "text" else text_sha256(v) for k, v in a.items()}


def _walk_anchors(obj):
    if isinstance(obj, dict):
        if "top_interleaving_anchors" in obj:
            obj["top_interleaving_anchors"] = [_hashed_anchor(a) for a in obj["top_interleaving_anchors"]]
        for v in obj.values():
            _walk_anchors(v)
    elif isinstance(obj, list):
        for v in obj:
            _walk_anchors(v)


def strip_texts(rel: str, obj):
    """The distributed form of result file ``rel`` (a path relative to the repository), texts replaced by hashes.
    Files without texts are returned unchanged; already stripped files are left as they are."""
    if rel.startswith("results/release_pipeline/mechanism_v2/"):
        _walk_anchors(obj)
    elif rel.startswith("results/rule_test/anchor_length"):
        for per_model in obj.values():
            for pop in per_model.values():
                if isinstance(pop, dict) and pop.get("top5") and not isinstance(pop["top5"][0], dict):
                    pop["top5"] = [{"text40_sha256": text_sha256(t), "length": n, "share": s} for t, n, s in pop["top5"]]
    elif re.fullmatch(r"results/gradsafe/summary_refset\d+\.json", rel) and isinstance(obj.get("reference"), dict) \
            and "unsafe" in obj["reference"]:
        obj["reference"] = {"unsafe_sha256": [text_sha256(t) for t in obj["reference"]["unsafe"]],
                            "safe_sha256": [text_sha256(t) for t in obj["reference"]["safe"]]}
    elif rel.startswith("results/dolly_calibration/split_v2/"):
        obj["dropped"] = [_hashed_anchor(d) for d in obj["dropped"]]
    elif rel == "results/METHODS_MANIFEST.json":
        ref = obj.get("gradsafe", {}).get("reference_prompts", {})
        if "unsafe" in ref:
            obj["gradsafe"]["reference_prompts"] = hashed_reference(ref["unsafe"], ref["safe"], GRADSAFE_SOURCE)
    return obj


def hashed_reference(unsafe: list[str], safe: list[str], source: str | None = None) -> dict:
    out = {"unsafe_sha256": [text_sha256(t) for t in unsafe], "safe_sha256": [text_sha256(t) for t in safe]}
    if source:
        out["source"] = source
    return out


def is_formatting_row(text80_sha256: str) -> bool:
    """Whether a hashed anchor excerpt is one of the formatting rows (as read, with or without its line end)."""
    return text80_sha256 in {text_sha256(r + end) for r in FORMATTING_ROWS for end in ("", "\n", "\r\n")}


def dump(obj) -> str:
    """The JSON layout every result file uses."""
    return json.dumps(obj, indent=1)


# ----------------------------------------------------------------------------------------------- SAA-850

def lc_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def saa850_index(prompts_bytes: bytes, template: str) -> dict:
    """The distributed stand-in for <model>_prompts.json: one hash per prompt and one for the whole file."""
    arms = json.loads(prompts_bytes)
    return {"about": "SAA-850 prompts are not distributed. scripts/release/rebuild_texts.py rebuilds "
                     "<model>_prompts.json from the MTK release's datasets/<model>_test/saa_1.json and checks it "
                     "against these hashes (THIRD_PARTY_NOTICES.md).",
            "template_sha256": text_sha256(template), "control_seed": CONTROL_SEED,
            "control_tokenizer": "AutoTokenizer.from_pretrained(<model dir>, use_fast=False)",
            "file_sha256": hashlib.sha256(prompts_bytes).hexdigest(),
            "arms": {a: [{"index": r["index"], "prompt_sha256": text_sha256(r["prompt"])} for r in rows]
                     for a, rows in arms.items()}}


def saa_template(record: dict, advbench_1: str, template_sha256: str) -> str:
    """Recover the SAA template from one released record, which is TEMPLATE(target, AdvBench #1) + suffix: the
    prefix of the record that, with the target and request put back as placeholders, has the recorded hash."""
    jb = record["jailbreak"]
    for cut in range(len(jb), 0, -1):
        t = jb[:cut].replace(record["target"], "{TARGET}").replace(lc_first(advbench_1), "{GOAL_LC}")
        if text_sha256(t) == template_sha256:
            return t
    raise ValueError("no prefix of the released record matches the SAA template hash")


def rebuild_saa850(records: list[dict], advbench_1: str, template: str, tokenizer=None) -> dict[str, list[dict]]:
    """The four SAA-850 arms as provenance/run_saa_multibehaviour.py built them (the two control arms only when a
    tokenizer is given)."""
    arms = {a: [] for a in ARMS}
    for i, r in enumerate(records):
        released_body = template.replace("{TARGET}", r["target"]).replace("{GOAL_LC}", lc_first(advbench_1))
        assert r["jailbreak"].startswith(released_body), f"record {i} is not template + suffix"
        suffix = r["jailbreak"][len(released_body):]
        body = template.replace("{TARGET}", r["target"]).replace("{GOAL_LC}", lc_first(r["goal"]))
        base = {"index": i, "goal": r["goal"], "target": r["target"]}
        arms["repaired"].append({**base, "prompt": body + suffix})
        arms["nosuffix"].append({**base, "prompt": body})
        if tokenizer is not None:
            ids = tokenizer(suffix, add_special_tokens=False).input_ids
            rng = np.random.default_rng(CONTROL_SEED + i)
            shuf = [ids[j] for j in rng.permutation(len(ids))]
            rand = [int(v) for v in rng.integers(0, tokenizer.vocab_size, len(ids))]
            arms["shufsuffix"].append({**base, "prompt": body + tokenizer.decode(shuf)})
            arms["randsuffix"].append({**base, "prompt": body + tokenizer.decode(rand)})
    return {a: rows for a, rows in arms.items() if rows}


def check_against_index(arms: dict[str, list[dict]], index: dict) -> list[str]:
    """Every rebuilt prompt must have its recorded hash; returns the mismatches (empty when all agree)."""
    bad = []
    for arm, rows in arms.items():
        want = index["arms"][arm]
        if len(rows) != len(want):
            bad.append(f"{arm}: {len(rows)} prompts, index has {len(want)}")
            continue
        bad += [f"{arm}[{i}]" for i, (r, w) in enumerate(zip(rows, want))
                if r["index"] != w["index"] or text_sha256(r["prompt"]) != w["prompt_sha256"]]
    return bad


TEXT_KEYS = {"text", "prompt", "goal", "target", "jailbreak", "unsafe", "safe"}
NOT_TEXTS = {("results/METHODS_MANIFEST.json", "/windowed_perplexity/text")}      # a setting, not a prompt


def find_texts(rel: str, obj) -> list[str]:
    """Paths of fields in result file ``rel`` that hold third-party text (empty for a distributable file)."""
    if rel.endswith("_prompts.json"):
        return ["<a rebuilt SAA-850 prompt file>"]
    found = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}/{k}"
                is_text = isinstance(v, str) or (isinstance(v, list) and any(isinstance(x, str) for x in v))
                if k in TEXT_KEYS and is_text and (rel, p) not in NOT_TEXTS:
                    found.append(p)
                elif k == "top5" and isinstance(v, list) and v and isinstance(v[0], list):
                    found.append(p)
                else:
                    walk(v, p)
        elif isinstance(o, list):
            for v in o:
                walk(v, path + "[]")
    walk(obj, "")
    return found
