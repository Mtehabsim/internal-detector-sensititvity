"""Equivalence gate: does release_driver.py reproduce the release's own runner, prompt by prompt?

    python scripts/release/release_gate.py                         # the archive: results/release_pipeline
    python scripts/release/release_gate.py --driver rerun --official scratch/official_runs      # a fresh run

For each model, the release runner (`mtk_<model>.py`, or `detection_vicuna.bash`) was executed once, as
shipped, at its configured seed, and wrote a per-prompt CSV per dataset. The driver's shipped-seed
scores must reproduce those CSVs: same prompts in the same order (Source_Index), same scores, same
per-family AUROC. Llama-2/3/Mistral CSVs hold the forest's decision value (driver anomaly = -value);
Vicuna's hold round(-fused, 4). Any mismatch beyond the stated tolerance fails the gate.

--driver is the directory holding <model>/rows.json and <model>/scores/ (default: the archived runs;
for a fresh run, the parent of the release_driver.py --out directories). --official holds the runners'
<model>/<run>/report/ CSVs (default: the archived copies, else $MTK_RELEASE_SCRATCH/official_runs). The report
is written to <driver>/gate.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mtkaudit import release as rd
from mtkaudit.cli import abs_path
from mtkaudit.paths import display

TOL = {"llama2": 1e-6, "llama3": 1e-6, "mistral": 1e-6, "vicuna": 5e-5}


def default_official() -> Path:
    base = rd.OUT / "official_runs"                   # the runners' CSVs, archived with the results
    return base if base.exists() else rd.SCRATCH / "official_runs"


def official_dir(model: str, root: Path) -> Path:
    base = root / model
    dirs = [d for d in base.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0] / "report"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", type=abs_path, default=rd.OUT)
    ap.add_argument("--official", type=abs_path, default=None)
    a = ap.parse_args()                  # before import_release(), which changes the working directory
    official = a.official or default_official()
    print(f"driver scores: {a.driver}\nrunner CSVs:   {official}", flush=True)
    fp = rd.import_release()
    report = {"tolerance": TOL, "driver": display(a.driver), "official": display(official), "models": {}}
    for model, seed in rd.SHIPPED_SEED.items():
        out = a.driver / model
        key = "exact_shipped" if model == "vicuna" else "release_cache"
        score_path = out / "scores" / f"as_shipped_{key}_seed{seed}.npy"
        if not score_path.exists():
            report["models"][model] = {"status": "MISSING driver scores"}
            continue
        anomaly = np.load(score_path)
        layout = json.loads((out / "rows.json").read_text())[key]
        rep_dir = official_dir(model, official)
        prefix = "vicuna_test" if model == "vicuna" else f"{model}_test"
        official_auroc = {r["Attack Method"]: float(r["AUROC"])
                          for r in read_csv(rep_dir / "all_attack_auroc_results.csv")}
        worst, per = 0.0, {}
        benign_off = None
        for name, info in layout["datasets"].items():
            suffix = "0" if name == rd.BENIGN_DATASET else "1"
            rows = read_csv(rep_dir / f"{prefix}_{name}_{suffix}_results_detail.csv")
            pos = rd.sampled_positions(fp, info["n"], seed, name)
            ours = anomaly[rd.rows_of(info, pos)]
            if model == "vicuna":
                theirs = np.array([-float(r["Anomaly_Score"]) for r in rows])
                ours_cmp = np.round(ours, 4)
            else:
                src = [info["source_indices"][p] for p in pos]
                if src != [int(r["Source_Index"]) for r in rows]:
                    raise SystemExit(f"{model}/{name}: prompt order differs from the runner")
                theirs = np.array([-float(r["Anomaly_Score"]) for r in rows])
                ours_cmp = ours
            diff = float(np.abs(ours_cmp - theirs).max())
            worst = max(worst, diff)
            per[name] = {"n": len(rows), "max_abs_score_diff": diff}
            if name == rd.BENIGN_DATASET:
                benign_off = theirs
        from sklearn.metrics import roc_auc_score
        auroc_diff = {}
        for name, info in layout["datasets"].items():
            if name == rd.BENIGN_DATASET:
                continue
            pos = rd.sampled_positions(fp, info["n"], seed, name)
            s = anomaly[rd.rows_of(info, pos)]
            b = anomaly[rd.rows_of(layout["datasets"][rd.BENIGN_DATASET],
                                   rd.sampled_positions(fp, layout["datasets"][rd.BENIGN_DATASET]["n"],
                                                        seed, rd.BENIGN_DATASET))]
            if model == "vicuna":
                s, b = np.round(s, 4), np.round(b, 4)
            ours_a = roc_auc_score(np.r_[np.zeros(len(b)), np.ones(len(s))], np.r_[b, s])
            auroc_diff[name] = abs(ours_a - official_auroc[name])
        passed = worst <= TOL[model] and max(auroc_diff.values()) <= 1e-9 + (1e-3 if model == "vicuna" else 0)
        report["models"][model] = {"seed": seed, "status": "PASS" if passed else "FAIL",
                                   "max_abs_score_diff": worst, "max_abs_auroc_diff": max(auroc_diff.values()),
                                   "official_mean_auroc": float(np.mean(list(official_auroc.values()))),
                                   "per_dataset": per, "auroc_diff": auroc_diff}
        print(model, report["models"][model]["status"], "score diff", worst,
              "auroc diff", max(auroc_diff.values()), flush=True)
    (a.driver / "gate.json").write_text(json.dumps(report, indent=1))
    if any(v["status"] != "PASS" for v in report["models"].values()):
        raise SystemExit("equivalence gate FAILED")


if __name__ == "__main__":
    main()
