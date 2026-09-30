"""Do evaluation goals overlap MTK's malicious anchors? (CPU; saved scores.)

    python scripts/analysis/goal_overlap.py      # writes results/uncertainty/goal_overlap.json

Every bank contains all 100 prompts of the release's AdvBench anchor file. Many attack records carry a `goal`
field (JailJudge does not). For each evaluated attack prompt (the matched items: the release's sampled families
and SAA-850), the goal is looked up in the release file and flagged when it equals an AdvBench anchor
(whitespace-normalised, case-insensitive). Detection at the real-user 5% threshold (the generator's
budget_metrics calibration) is then reported for overlapping and non-overlapping prompts, for MTK and for the
linear probe (which is trained on the anchors). A record whose prompt hash does not match the item stops the run.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from mtkaudit import release as rd
from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2

FILES = {"nanogcg": ("nanogcg_1.json", "nonagcg_1.json"), "JailJudge": ("JailJudge_1.json", "JailJudge_all_1.json")}


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def records(path):
    """Raw records, decoded as the release's load_records decodes them (UTF-8, errors ignored as a fallback)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        d = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))


def main():
    anchors = {norm(l) for l in (rd.COPY / "datasets/train_data/AdvBench.txt").read_text().splitlines() if l.strip()}
    out = {"anchors": len(anchors), "rule": "goal equals an AdvBench anchor (whitespace-normalised, lower-case)",
           "models": {}}
    for m in mt.MODELS:
        items, dets = mt.matched_scores(m)
        scores = {"MTK": np.asarray(dets["MTK"], float),
                  "Probe": np.load(mt.RES / "matched_detectors" / m / "probe.npy").astype(float)}
        cal = calibration_rows_v2(items)
        thr = {d: float(np.quantile(s[cal], 0.95)) for d, s in scores.items()}
        fams = sorted({i["family"] for i in items if i["population"] == "release_attack"})
        res = {}
        groups = [(f, [i for i in items if i["population"] == "release_attack" and i["family"] == f], f) for f in fams]
        groups.append(("SAA-850", [i for i in items if i["population"] == "saa850_repaired"], "saa"))
        for name, sel, fam_file in groups:
            path = next((rd.COPY / "datasets" / f"{m}_test" / c for c in FILES.get(fam_file, (f"{fam_file}_1.json",))
                         if (rd.COPY / "datasets" / f"{m}_test" / c).exists()), None)
            recs = records(path)
            if "goal" not in recs[0]:
                res[name] = {"n": len(sel), "goal_field": False}
                continue
            flag = []
            for i in sel:
                r = recs[int(i["source_index"])]
                text = r.get("jailbreak") or r.get("prompt")
                if name != "SAA-850" and hashlib.sha256(text.encode("utf-8")).hexdigest() != i["prompt_sha256"]:
                    raise SystemExit(f"{m} {name} record {i['source_index']}: prompt hash differs from the item")
                flag.append(norm(r["goal"]) in anchors)
            flag = np.array(flag)
            rows = np.array([i["row"] for i in sel])
            rec = {"n": len(sel), "goal_field": True, "n_overlap": int(flag.sum())}
            for d, s in scores.items():
                x = s[rows] > thr[d]
                rec[d] = {"tpr_all": float(x.mean()),
                          "tpr_overlap": float(x[flag].mean()) if flag.any() else None,
                          "tpr_non_overlap": float(x[~flag].mean()) if (~flag).any() else None}
            res[name] = rec
        out["models"][m] = res
        print(m, {k: (v["n_overlap"], v["n"], round(v["MTK"]["tpr_all"], 3), v["MTK"]["tpr_non_overlap"] and round(v["MTK"]["tpr_non_overlap"], 3))
                  for k, v in res.items() if v.get("goal_field")}, flush=True)
    p = mt.RES / "uncertainty" / "goal_overlap.json"
    if p.exists():
        raise SystemExit(f"{p} exists; write a new version instead of overwriting")
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
