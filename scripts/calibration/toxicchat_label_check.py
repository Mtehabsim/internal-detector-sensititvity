"""How "benign" are the ToxicChat benign prompts? A check of the calibration-transfer result against ToxicChat's labels.

    python scripts/calibration/toxicchat_label_check.py      # writes results/uncertainty/toxicchat_labels.json

The release's ToxicChat benign file carries prompt texts only. ToxicChat (0124 test split, pinned revision below)
labels each prompt toxic or not, and marks which labels a human annotator gave; the rest were assigned
automatically. Each evaluated and calibration prompt is matched to that file by its whitespace-normalised text and
labelled: "toxic" (any matching row labelled toxic), "human" (non-toxic, human-annotated), "auto" (non-toxic, not
human-annotated) or "unmatched" (not in the test split). For every detector, the Dolly-set 5% threshold
(results/dolly_calibration/summary_v2.json) is then applied to three subsets of the 500 evaluated prompts: all,
human-labelled non-toxic only, and all but toxic and unmatched. Only prompt hashes and labels are saved, no text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np
from huggingface_hub import hf_hub_download

from mtkaudit import release as rd
from mtkaudit import results as mt
from mtkaudit.cli import run

REPO, REVISION = "lmsys/toxic-chat", "29df8e4dba60e1f4af4b4075c0705c5b313548a8"
FILE = "data/0124/toxic-chat_annotation_test.csv"
DETECTORS = ("MTK", "Probe", "GradSafe", "HiddenDetect", "Windowed PPL")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def toxicchat_labels() -> tuple[dict, dict]:
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    csv.field_size_limit(sys.maxsize)
    label = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        t = norm(r["user_input"])
        new = "toxic" if r["toxicity"] == "1" else ("human" if r["human_annotation"] == "True" else "auto")
        old = label.get(t)
        rank = {"toxic": 2, "human": 1, "auto": 0}      # duplicates: the most conservative label wins
        if old is None or rank[new] > rank[old]:
            label[t] = new
    return label, {"repo": REPO, "revision": REVISION, "file": FILE,
                   "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def main():
    label, source = toxicchat_labels()
    fp = rd.import_release()
    summary = json.loads((mt.RES / "dolly_calibration/summary_v2.json").read_text())
    out = {"source": source, "rule": "whitespace-normalised exact text match; toxic > human > auto for duplicates",
           "models": {}}
    for m in mt.MODELS:
        by_hash = {sha(t): label.get(norm(t), "unmatched") for t in rd.benign_texts(fp, m)}
        items, dets = mt.matched_scores(m)
        dets = dict(dets)
        dets["Probe"] = np.load(mt.RES / "matched_detectors" / m / "probe.npy")
        rec = {"labels": {}, "counts": {}, "dolly_threshold_fpr": {}}
        for pop in ("benign_eval", "benign_calibration"):
            sel = [i for i in items if i["population"] == pop]
            labs = [by_hash[i["prompt_sha256"]] for i in sel]
            rec["counts"][pop] = dict(Counter(labs))
            rec["labels"].update({i["prompt_sha256"]: lab for i, lab in zip(sel, labs)})
        ev = [i for i in items if i["population"] == "benign_eval"]
        labs = np.array([by_hash[i["prompt_sha256"]] for i in ev])
        rows = np.array([i["row"] for i in ev])
        for d in DETECTORS:
            s = np.asarray(dets[d], float)[rows]
            t = summary[m][d]["threshold"]["dolly"]
            fpr = lambda mask: float((s[mask] > t).mean())
            rec["dolly_threshold_fpr"][d] = {"all": fpr(np.ones(len(s), bool)), "human": fpr(labs == "human"),
                                             "clean": fpr((labs == "human") | (labs == "auto")),
                                             "paper": summary[m][d]["dolly"]["fpr_toxicchat_eval"]}
            if abs(rec["dolly_threshold_fpr"][d]["all"] - rec["dolly_threshold_fpr"][d]["paper"]) > 1e-12:
                raise SystemExit(f"{m} {d}: recomputed FPR differs from the archived Dolly result")
        out["models"][m] = rec
        print(m, rec["counts"]["benign_eval"],
              {d: {k: round(v, 3) for k, v in x.items()} for d, x in rec["dolly_threshold_fpr"].items() if d == "MTK"})
    p = mt.RES / "uncertainty" / "toxicchat_labels.json"
    if p.exists():
        raise SystemExit(f"{p} exists; write a new version instead of overwriting")
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
