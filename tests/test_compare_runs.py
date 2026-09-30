"""compare_runs: an unchanged copy of archived records passes; a changed score or metric fails."""
import json
import shutil

import numpy as np

from mtkaudit.release import OUT
from mtkaudit.runs import compare_model

ARCHIVED = OUT / "llama2"


def _fresh_copy(dest, n=2):
    dest.mkdir()
    (dest / "scores").mkdir()
    shutil.copy(ARCHIVED / "rows.json", dest / "rows.json")
    recs = [json.loads(l) for l in (ARCHIVED / "sweep_v2.jsonl").read_text().splitlines()[:n]]
    lines = []
    for r in recs:
        for k in ("recomputed_from", "v1_recompute_matches_record"):   # added by the recomputation, not the driver
            r.pop(k, None)
        tag = f"{r['variant']}_{r['test_features']}_seed{r['seed']}"
        shutil.copy(ARCHIVED / "scores" / f"{tag}.npy", dest / "scores" / f"{tag}.npy")
        lines.append(json.dumps(r))
    (dest / "sweep.jsonl").write_text("\n".join(lines) + "\n")
    return recs


def test_identical_records_pass(tmp_path):
    _fresh_copy(tmp_path / "llama2")
    r = compare_model(tmp_path / "llama2", ARCHIVED, 1e-6, 1e-9)
    assert r["pass"] and len(r["records"]) == 2 and r["max_abs_score_diff"] == 0


def test_a_changed_score_fails(tmp_path):
    recs = _fresh_copy(tmp_path / "llama2")
    tag = f"{recs[0]['variant']}_{recs[0]['test_features']}_seed{recs[0]['seed']}"
    f = tmp_path / "llama2" / "scores" / f"{tag}.npy"
    s = np.load(f)
    s[0] += 1e-3
    np.save(f, s)
    assert not compare_model(tmp_path / "llama2", ARCHIVED, 1e-6, 1e-9)["pass"]


def test_a_changed_metric_fails(tmp_path):
    recs = _fresh_copy(tmp_path / "llama2", n=1)
    recs[0]["metrics"]["mean_auroc"] += 0.01
    (tmp_path / "llama2" / "sweep.jsonl").write_text(json.dumps(recs[0]) + "\n")
    r = compare_model(tmp_path / "llama2", ARCHIVED, 1e-6, 1e-9)
    assert not r["pass"] and any("mean_auroc" in p for rec in r["records"].values() for p in rec["differing_metrics"])


def _one_record(tmp_path):
    recs = _fresh_copy(tmp_path / "llama2", n=1)
    tag = f"{recs[0]['variant']}_{recs[0]['test_features']}_seed{recs[0]['seed']}"
    return recs[0], tmp_path / "llama2" / "scores" / f"{tag}.npy"


def test_a_nan_score_fails(tmp_path):
    rec, f = _one_record(tmp_path)
    s = np.load(f)
    s[3] = np.nan
    np.save(f, s)
    assert not compare_model(tmp_path / "llama2", ARCHIVED, 1e-6, 1e-9)["pass"]


def test_a_nan_metric_fails(tmp_path):
    rec, _ = _one_record(tmp_path)
    rec["metrics"]["mean_auroc"] = float("nan")
    (tmp_path / "llama2" / "sweep.jsonl").write_text(json.dumps(rec) + "\n")
    assert not compare_model(tmp_path / "llama2", ARCHIVED, 1e-6, 1e-9)["pass"]


def test_a_missing_field_fails(tmp_path):
    rec, _ = _one_record(tmp_path)
    rec.pop("metrics")
    (tmp_path / "llama2" / "sweep.jsonl").write_text(json.dumps(rec) + "\n")
    r = compare_model(tmp_path / "llama2", ARCHIVED, 1e-6, 1e-9)
    assert not r["pass"] and "metrics" in next(iter(r["records"].values()))["missing_fields"]
