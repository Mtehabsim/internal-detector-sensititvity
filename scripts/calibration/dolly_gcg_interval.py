"""The GCG interval under Dolly calibration, on the text-disjoint split (CPU; saved scores only).

    python scripts/calibration/dolly_gcg_interval.py     # writes results/dolly_calibration/gcg_interval.json

dolly_calibration.py reported, per family, a 95% interval for detection at the Dolly-set 5% threshold by
resampling the 200 Dolly calibration prompts (B=2000, seed 20260925, threshold recomputed); the family's own
prompts are not resampled, so the interval reflects threshold uncertainty only. It was computed on the first-200
split. This recomputes it with the same method on the text-disjoint split (split_v2) for the release's Llama-2 GCG
family (nanoGCG, 42 prompts), and first checks that the method reproduces the archived first-200 interval.
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.paths import ROOT

D = ROOT / "results/dolly_calibration"
B, SEED = 2000, 20260925


def interval(d_cal, x):
    rng = np.random.default_rng(SEED)
    boot = np.quantile(d_cal[rng.integers(0, len(d_cal), (B, len(d_cal)))], 0.95, axis=1)
    bt = (x[None, :] > boot[:, None]).mean(axis=1)
    return [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]


def main():
    items, dets = mt.matched_scores("llama2")
    s = np.asarray(dets["MTK"], float)
    x = s[[i["row"] for i in items if i["population"] == "release_attack" and i["family"] == "nanogcg"]]
    dolly = np.load(D / "llama2_dolly_scores.npy").astype(float)
    archived = json.loads((D / "llama2.json").read_text())["families"]["nanogcg"]["tpr_dolly_threshold_ci"]
    check = interval(dolly[:200], x)
    assert np.allclose(check, archived), (check, archived)
    split = json.loads((D / "split_v2" / "llama2.json").read_text())
    d_cal = dolly[split["calibration_idx"]]
    thr = np.quantile(d_cal, 0.95)
    out = {"family": "nanogcg", "model": "llama2", "n": len(x), "B": B, "seed": SEED,
           "resamples": "Dolly calibration prompts only (threshold uncertainty); the family's prompts are fixed",
           "archived_first200_interval_reproduced": check,
           "n_calibration": len(d_cal), "detected": int((x > thr).sum()), "tpr": float((x > thr).mean()),
           "interval": interval(d_cal, x)}
    (D / "gcg_interval.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    run(main, __doc__)
