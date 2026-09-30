"""Select a second real-user benign set from WildChat (CPU; downloads one pinned file of WildChat-1M).

    python scripts/calibration/wildchat_select.py   # writes results/wildchat/selection.json and wildchat_prompts.json

Design fixed before selection: from the pinned file, English, non-toxic, non-redacted conversations; the first user
turn, stripped, 10 to 2,000 characters; deduplicated on normalised text in file order; minus any prompt whose
normalised text occurs in the release's ToxicChat benign files (all four models) or its Dolly pool.
random.Random(20260926).sample(pool, 1500): the first 1,000 calibrate, the last 500 evaluate (mtkaudit.wildchat).
selection.json records conversation hashes and prompt SHA-256s only; wildchat_prompts.json holds the texts for the
scoring runs and is not distributed.
"""
from __future__ import annotations

import json

from mtkaudit import release as rd
from mtkaudit import wildchat as wcm
from mtkaudit.cli import run


def main():
    fp = rd.import_release()
    counts, cal, ev = wcm.select(fp, rd.COPY)
    wcm.OUT.mkdir(parents=True, exist_ok=True)
    sel = {"source": {"repo": wcm.REPO, "revision": wcm.REVISION, "file": wcm.FILE, "file_sha256": wcm.FILE_SHA256,
                      "license": "ODC-BY"},
           "design": "paper/PREREG_20260926_extensions.md, section C", "seed": wcm.SEED, "counts": counts,
           "calibration": wcm.hashes(cal), "evaluation": wcm.hashes(ev)}
    (wcm.OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    wcm.PROMPTS.write_text(json.dumps({"calibration": [t for _, t in cal], "evaluation": [t for _, t in ev]}, indent=1))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
