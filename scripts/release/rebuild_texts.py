"""Rebuild the third-party texts this repository does not distribute, and check each against its recorded hash.

    python scripts/release/rebuild_texts.py              # SAA-850 -> results/saa_multibehaviour/<model>_prompts.json
    python scripts/release/rebuild_texts.py --excerpts   # also every hashed excerpt -> restored_texts/
    python scripts/release/rebuild_texts.py --wildchat   # also the WildChat set -> results/wildchat/wildchat_prompts.json

Needs the MTK release (make setup). SAA-850 (THIRD_PARTY_NOTICES.md): each record of the release's
datasets/<model>_test/saa_1.json is the SAA template, filled with the record's target and AdvBench's first
request, plus an adversarial suffix; the template is recovered from the first record by its recorded hash. The
repaired and no-suffix arms need nothing else. The two control arms (the suffix's own tokens permuted; random
tokens of the same length) need each model's tokenizer, loaded as the original build did (use_fast=False) from
$MTK_MODEL_DIR/<model dir>; without it they are skipped and the file holds the first two arms, which is all the
GPU stages other than release_sentinel.py read. With all four arms the rebuilt file is byte-identical to the one
the runs used (its SHA-256 is in the index).

--excerpts restores the hashed excerpts (anchor prompts, GradSafe reference prompts, dropped Dolly prompts, the
Vicuna runner CSVs' prompts) by matching every hash against the prompts in the release's pools and test sets (and
GradSafe's code, if fetched), and writes copies of those files with the texts put back under restored_texts/,
which is not distributed.

--wildchat downloads the pinned WildChat-1M file (231 MB, ODC-BY), reruns the selection of mtkaudit.wildchat and
requires every prompt to match results/wildchat/selection.json; the WildChat scoring runs read the rebuilt file.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import io
import json

from mtkaudit import release as rd
from mtkaudit import wildchat as wcm
from mtkaudit.paths import GRADSAFE_DIR, MODEL_DIR, ROOT
from mtkaudit.texts import (ARMS, MODEL_DIRS, check_against_index, dump, norm, rebuild_saa850, saa_template,
                            text_sha256)

SAA = ROOT / "results/saa_multibehaviour"
RESTORED = ROOT / "restored_texts"


def tokenizer_for(model: str):
    path = MODEL_DIR / MODEL_DIRS[model]
    if not (path / "tokenizer_config.json").exists():
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, use_fast=False)


def saa850(fp) -> None:
    lines = [l.strip() for l in fp.read_lines(rd.COPY / "datasets/train_data/AdvBench.txt") if l.strip()]
    advbench_1 = lines[0]
    for model in rd.SHIPPED_SEED:
        index = json.loads((SAA / f"{model}_prompts_index.json").read_text())
        target = SAA / f"{model}_prompts.json"
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == index["file_sha256"]:
            print(f"{model}: already rebuilt, identical to the file the runs used", flush=True)
            continue
        records = json.loads((rd.COPY / f"datasets/{model}_test/saa_1.json").read_text())
        template = saa_template(records[0], advbench_1, index["template_sha256"])
        tok = tokenizer_for(model)
        arms = rebuild_saa850(records, advbench_1, template, tok)
        bad = check_against_index(arms, index)
        if bad:
            raise SystemExit(f"{model}: {len(bad)} rebuilt prompts differ from the recorded hashes, e.g. {bad[:3]}")
        text = json.dumps(arms, indent=0)
        if len(arms) == len(ARMS) and hashlib.sha256(text.encode()).hexdigest() != index["file_sha256"]:
            raise SystemExit(f"{model}: every prompt matches, but the file differs from the one the runs used")
        target.write_text(text)
        print(f"{model}: {sum(map(len, arms.values()))} prompts in arms {list(arms)}, every hash matches"
              + ("; file identical to the one the runs used" if len(arms) == len(ARMS) else
                 f"; control arms skipped (no tokenizer under {MODEL_DIR / MODEL_DIRS[model]})"), flush=True)


# ---------------------------------------------------------------------------------------------- excerpts

def candidates(fp) -> dict[str, str]:
    """Hash -> text for every excerpt form the result files use, over the release's prompts and GradSafe's code."""
    texts = set()
    readers = [fp.read_lines, importlib.import_module("extract_trainset_hiddenstates_mistral").read_mistral_training_lines]
    for path in sorted((rd.COPY / "datasets/train_data").glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        texts.update(raw)
        for reader in readers:
            try:
                texts.update(reader(path))
            except Exception:                                  # a reader that does not apply to this file
                pass
    for path in sorted((rd.COPY / "datasets").glob("*_test/*.json")):
        texts.update(r["prompt"] for r in fp.load_records(path))            # as the release's runners read them
        stack = [json.loads(path.read_text(encoding="utf-8", errors="ignore"))]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                stack += o.values()
            elif isinstance(o, list):
                stack += o
            elif isinstance(o, str):
                texts.add(o)
    gs = GRADSAFE_DIR / "code/find_critical_parameters.py"
    if gs.exists():
        texts.update(n.value for n in ast.walk(ast.parse(gs.read_text()))
                     if isinstance(n, ast.Constant) and isinstance(n.value, str))
    out = {}
    for t in texts:
        runner_csv = t[:500] + "..." if len(t) > 500 else t       # how mtk_vicuna.py writes a prompt to its CSV
        for form in (t, t.strip(), t[:80], t.strip()[:40], norm(t)[:80], runner_csv):
            out[text_sha256(form)] = form
    return out


def restore_excerpts(fp) -> None:
    table = candidates(fp)
    found = missing = 0

    def back(h):
        nonlocal found, missing
        if h in table:
            found += 1
            return table[h]
        missing += 1
        return None

    def put(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("text80_sha256", "text40_sha256"):
                    out[k.replace("_sha256", "")] = back(v)
                elif k in ("unsafe_sha256", "safe_sha256"):
                    out[k.replace("_sha256", "")] = [back(h) for h in v]
                else:
                    out[k] = put(v)
            return out
        return [put(v) for v in obj] if isinstance(obj, list) else obj

    files = (sorted((ROOT / "results/release_pipeline/mechanism_v2").glob("*.json"))
             + sorted((ROOT / "results/rule_test").glob("anchor_length*.json"))
             + sorted((ROOT / "results/gradsafe").glob("summary_refset*.json"))
             + sorted((ROOT / "results/dolly_calibration/split_v2").glob("*.json"))
             + [ROOT / "results/METHODS_MANIFEST.json"])
    for path in files:
        rel = path.relative_to(ROOT)
        (RESTORED / rel).parent.mkdir(parents=True, exist_ok=True)
        (RESTORED / rel).write_text(dump(put(json.loads(path.read_text()))))
    for path in sorted((ROOT / "results/release_pipeline/official_runs").rglob("*_results_detail.csv")):
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"), newline="")))
        if "prompt_sha256" not in rows[0]:
            continue
        j = rows[0].index("prompt_sha256")
        rows[0][j] = "prompt"
        for r in rows[1:]:
            r[j] = back(r[j]) or ""
        rel = path.relative_to(ROOT)
        (RESTORED / rel).parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        (RESTORED / rel).write_text(buf.getvalue(), encoding="utf-8")
    print(f"excerpts: {found} restored, {missing} not found in the release's files -> {RESTORED}", flush=True)
    if missing:
        raise SystemExit("some hashes did not match any text of the release")


def wildchat(fp) -> None:
    sel = json.loads((wcm.OUT / "selection.json").read_text())
    _, cal, ev = wcm.select(fp, rd.COPY)
    for part, got in (("calibration", cal), ("evaluation", ev)):
        if wcm.hashes(got) != sel[part]:
            raise SystemExit(f"WildChat {part}: the rebuilt selection differs from selection.json")
    wcm.PROMPTS.write_text(json.dumps({"calibration": [t for _, t in cal], "evaluation": [t for _, t in ev]}, indent=1))
    print(f"WildChat: {len(cal)} + {len(ev)} prompts rebuilt, every hash matches -> {wcm.PROMPTS.relative_to(ROOT)}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excerpts", action="store_true", help="also restore the hashed excerpts into restored_texts/")
    ap.add_argument("--wildchat", action="store_true", help="also rebuild the WildChat benign set")
    a = ap.parse_args()
    rd.verify_copy()
    fp = rd.import_release()
    saa850(fp)
    if a.excerpts:
        restore_excerpts(fp)
    if a.wildchat:
        wildchat(fp)


if __name__ == "__main__":
    main()
