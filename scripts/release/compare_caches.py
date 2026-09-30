"""Compare a freshly regenerated scratch tree with the original, file by file (CPU).

    python scripts/release/compare_caches.py --original ORIG_SCRATCH --fresh FRESH_SCRATCH

Every file listed in results/archive/cache_SHA256SUMS is compared. Byte-identical files pass directly;
otherwise .pt files are loaded and compared as content (tensors exactly, strings with each tree's scratch root
normalised, because the release's manifests record absolute paths). Writes
results/release_pipeline/cache_regeneration/fresh_scratch_comparison.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mtkaudit import caches as rg
from mtkaudit.cli import abs_path
from mtkaudit.paths import ROOT


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=abs_path, required=True)
    ap.add_argument("--fresh", type=abs_path, required=True)
    a = ap.parse_args()
    rg.ROOTS.extend(sorted({str(a.original), str(a.fresh)}, key=len, reverse=True))
    report = {}
    for line in (ROOT / "results/archive/cache_SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        f = a.fresh / rel
        if not f.exists():
            report[rel] = "MISSING"
        elif sha(f) == digest:
            report[rel] = "byte-identical"
        elif f.suffix == ".pt":
            same = rg.equal(torch.load(a.original / rel, map_location="cpu", weights_only=False),
                            torch.load(f, map_location="cpu", weights_only=False))
            report[rel] = "content-identical (paths normalised)" if same else "DIFFERS"
        else:
            report[rel] = "DIFFERS"
        print(f"{report[rel]:40s} {rel}", flush=True)
    out = ROOT / "results/release_pipeline/cache_regeneration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fresh_scratch_comparison.json").write_text(json.dumps(report, indent=1))
    bad = [k for k, v in report.items() if v in ("MISSING", "DIFFERS")]
    print(f"{len(report) - len(bad)}/{len(report)} files reproduced", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
