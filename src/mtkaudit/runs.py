"""Comparison of a fresh run of the release harness with the archived one (used by scripts/release/compare_runs.py).

A run directory holds, per model, what mtkaudit.release writes: rows.json (which row is which prompt),
sweep.jsonl (one record per seed, variant and test-feature set) and scores/<variant>_<features>_seed<seed>.npy.
The driver computes metrics with the text-disjoint calibration, which for the archived runs is recorded in
sweep_v2.jsonl (their sweep.jsonl holds the first, position-disjoint calibration).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

IGNORED = ("seconds",)          # wall-clock time differs between runs by nature
# every record the driver writes has these; a fresh record missing one fails (archived records may add fields)
REQUIRED = ("seed", "variant", "test_features", "forest_seed", "sample_seed", "shipped", "bank_sources", "metrics")


def load_records(path: Path) -> dict[tuple, dict]:
    recs = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        recs[(r["seed"], r["variant"], r["test_features"])] = r
    return recs


def metric_diffs(a, b, path: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf that differs, with its absolute difference (inf for a structural difference)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [(f"{path}/{{keys}}", math.inf)]
        return [d for k in a for d in metric_diffs(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(f"{path}/{{len}}", math.inf)]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in metric_diffs(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if math.isnan(a) or math.isnan(b):                   # NaN never compares equal to a number
            return [] if (math.isnan(a) and math.isnan(b)) else [(path, math.inf)]
        d = abs(a - b)
        return [(path, d)] if d > 0 else []
    return [] if a == b else [(path, math.inf)]


def compare_model(fresh: Path, archived: Path, tol: float, metric_tol: float) -> dict:
    """Compare <fresh>/ with <archived>/ for one model; see the module docstring for the layout."""
    report = {"fresh": str(fresh), "archived": str(archived), "records": {}, "missing_in_archive": [],
              "rows_identical": json.loads((fresh / "rows.json").read_text())
              == json.loads((archived / "rows.json").read_text())}
    old = load_records(archived / "sweep_v2.jsonl")
    worst_score, worst_metric = 0.0, 0.0
    for key, rec in sorted(load_records(fresh / "sweep.jsonl").items()):
        seed, variant, feats = key
        tag = f"{variant}_{feats}_seed{seed}"
        if key not in old:
            report["missing_in_archive"].append(tag)
            continue
        a = np.load(fresh / "scores" / f"{tag}.npy").astype(np.float64)
        b = np.load(archived / "scores" / f"{tag}.npy").astype(np.float64)
        ok_scores = a.shape == b.shape and bool(np.isfinite(a).all()) and bool(np.isfinite(b).all())
        score_diff = float(np.abs(a - b).max()) if ok_scores else math.inf
        missing = [k for k in REQUIRED if k not in rec]
        diffs = [(f"/{k}", math.inf) for k in missing]
        diffs += metric_diffs({k: rec[k] for k in REQUIRED if k in rec},
                              {k: old[key][k] for k in REQUIRED if k in rec})
        m = max((d for _, d in diffs), default=0.0)
        worst_score, worst_metric = max(worst_score, score_diff), max(worst_metric, m)   # both finite or inf
        report["records"][tag] = {"max_abs_score_diff": score_diff, "max_abs_metric_diff": m, "missing_fields": missing,
                                  "differing_metrics": [p for p, d in diffs if d > metric_tol][:10]}
    report["max_abs_score_diff"], report["max_abs_metric_diff"] = worst_score, worst_metric
    report["pass"] = (report["rows_identical"] and bool(report["records"])
                      and worst_score <= tol and worst_metric <= metric_tol)
    return report
