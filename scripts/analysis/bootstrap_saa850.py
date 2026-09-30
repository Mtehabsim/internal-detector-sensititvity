"""Bootstrap intervals for SAA-850 detection (CPU; saved scores only).

    python scripts/analysis/bootstrap_saa850.py      # writes results/uncertainty/saa850.json

The release's SAA file puts one request line in all its records, so its 500 sampled prompts differ only in
their suffixes and a bootstrap over them treats near-duplicates as independent. SAA-850 carries 850 distinct
AdvBench behaviours. Each replicate (B=2000, seed 20260925, as bootstrap_ci.py) resamples the text-disjoint
calibration prompts (thresholds recomputed), the 500 evaluated benign prompts and the 850 SAA-850 records;
every detector of a model uses the same indices.
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2
from mtkaudit.paths import ROOT
from mtkaudit.stats import load, auroc_rows, summarise

B, SEED = 2000, 20260925


def main():
    rng = np.random.default_rng(SEED)
    out = {"B": B, "seed": SEED, "models": {}}
    for m in ("llama2", "llama3", "mistral", "vicuna"):
        items, dets = load(m)
        dets = dict(dets)
        dets["Linear probe"] = np.load(ROOT / "results/matched_detectors" / m / "probe.npy").astype(float)
        cal = np.array(calibration_rows_v2(items))
        ev = np.array([i["row"] for i in items if i["population"] == "benign_eval"])
        x = np.array([i["row"] for i in items if i["population"] == "saa850_repaired"])
        ic = rng.integers(0, len(cal), (B, len(cal)))
        ie = rng.integers(0, len(ev), (B, len(ev)))
        ix = rng.integers(0, len(x), (B, len(x)))
        out["models"][m] = {"n": len(x)}
        for d, s in dets.items():
            thr = np.quantile(s[cal], 0.95)
            bt = np.quantile(s[cal][ic], 0.95, axis=1)
            tpr = (s[x][ix] > bt[:, None]).mean(axis=1)
            au = auroc_rows(s[ev][ie], s[x][ix])
            pt_au = auroc_rows(s[ev][None, :], s[x][None, :])[0]
            out["models"][m][d] = {"tpr_0.05": summarise((s[x] > thr).mean(), tpr), "auroc": summarise(pt_au, au)}
            r = out["models"][m][d]
            print(f"{m:8s} {d:13s} SAA-850 TPR@5 {r['tpr_0.05']['point']:.3f} [{r['tpr_0.05']['lo']:.3f}, "
                  f"{r['tpr_0.05']['hi']:.3f}]  AUROC {r['auroc']['point']:.3f}")
    (ROOT / "results/uncertainty/saa850.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
