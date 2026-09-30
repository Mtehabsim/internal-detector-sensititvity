"""The WildChat benign set: a second real-user distribution (design: paper/PREREG_20260926_extensions.md, section C,
in the research record; summarised in scripts/calibration/wildchat_select.py).

select() rebuilds the selection deterministically from one pinned file of WildChat-1M (ODC-BY). The texts are never
distributed: results/wildchat/selection.json holds conversation hashes and prompt SHA-256s, and
scripts/release/rebuild_texts.py --wildchat rebuilds wildchat_prompts.json locally and checks every hash.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from mtkaudit.paths import ROOT
from mtkaudit.texts import norm, text_sha256

REPO, REVISION = "allenai/WildChat-1M", "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
FILE, FILE_SHA256 = "data/train-00000-of-00014.parquet", "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c"
SEED, N_CAL, N_EVAL = 20260926, 1000, 500
OUT = ROOT / "results/wildchat"
PROMPTS = OUT / "wildchat_prompts.json"          # local only (.gitignore)


def download() -> Path:
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION, cache_dir=str(ROOT / "data_cache")))
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == FILE_SHA256, f"downloaded file differs from the pinned one ({got})"
    return path


def select(fp, copy: Path):
    """(counts, calibration, evaluation): lists of (conversation_hash, prompt) chosen by the pre-registered rules."""
    import pyarrow.parquet as pq
    path = download()
    exclude = set()
    for m in ("llama2", "llama3", "mistral", "vicuna"):
        exclude |= {norm(r["prompt"]) for r in fp.load_records(copy / f"datasets/{m}_test/toxic-chat_benign_0.json")}
    exclude |= {norm(l) for l in (copy / "datasets/train_data/databricks-dolly-15k.txt")
                .read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()}
    table = pq.read_table(path, columns=["conversation_hash", "conversation", "language", "toxic", "redacted"])
    counts = {"conversations": table.num_rows}
    pool, seen = [], set()
    kept = {"english_nontoxic_unredacted": 0, "first_user_turn_length_ok": 0, "not_in_toxicchat_or_dolly": 0}
    for row in table.to_pylist():
        if row["language"] != "English" or row["toxic"] or row["redacted"]:
            continue
        kept["english_nontoxic_unredacted"] += 1
        first = next((t for t in row["conversation"] if t.get("role") == "user"), None)
        text = (first or {}).get("content") or ""
        text = text.strip()
        if not 10 <= len(text) <= 2000:
            continue
        kept["first_user_turn_length_ok"] += 1
        n = norm(text)
        if n in exclude:
            continue
        kept["not_in_toxicchat_or_dolly"] += 1
        if n in seen:
            continue
        seen.add(n)
        pool.append((row["conversation_hash"], text))
    counts.update(kept)
    counts["pool_after_dedup"] = len(pool)
    pick = random.Random(SEED).sample(pool, N_CAL + N_EVAL)
    return counts, pick[:N_CAL], pick[N_CAL:]


def hashes(pairs):
    return [{"conversation_hash": h, "prompt_sha256": text_sha256(t)} for h, t in pairs]
