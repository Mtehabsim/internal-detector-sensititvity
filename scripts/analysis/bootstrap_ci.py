"""Finite-sample uncertainty for the operating-point results (CPU; no model).

    python scripts/analysis/bootstrap_ci.py            # writes results/uncertainty/bootstrap.json

For each model, the four detectors are scored on identical prompts (results/matched_detectors/<model>). Each
bootstrap replicate resamples, with replacement and independently:
  * the calibration prompts (text-disjoint set), then RECOMPUTES every threshold from the resample,
  * the release's 500 evaluated benign prompts (realised FPR, AUROC),
  * each attack family's prompts (TPR, AUROC),
  * the 850 SAA-850 records together with their two judges' labels (eASR).
Every detector of a model uses the same resampled indices in a replicate, so differences between detectors
are paired. Intervals are 95% percentile intervals over B replicates; point estimates are the full-sample
values. This is uncertainty from finite evaluation samples at a FIXED reference bank; variation across banks
is measured separately (sweep_v2), and the two are reported apart.
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit import release as rd
from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2
from mtkaudit.paths import ROOT
from mtkaudit.stats import auroc_rows, load, summarise

OUT = ROOT / "results" / "uncertainty"
BUDGETS = (0.01, 0.02, 0.03, 0.04, 0.05)
B = 2000
SEED = 20260925


def metrics(s, cal, ev, fams, x850, hb, sr):
    """Full-sample and vectorised-replicate metrics share this: inputs are (B, n) arrays (B=1 for points)."""
    thr = {t: np.quantile(cal, 1 - t, axis=1) for t in BUDGETS}
    out = {f"fpr_{t}": (ev > thr[t][:, None]).mean(axis=1) for t in BUDGETS}
    tprs = {}
    for f, x in fams.items():
        out[f"auroc_{f}"] = auroc_rows(ev, x)
        for t in BUDGETS:
            tprs[(f, t)] = (x > thr[t][:, None]).mean(axis=1)
            out[f"tpr_{f}_{t}"] = tprs[(f, t)]
    for t in BUDGETS:
        out[f"mean_tpr_{t}"] = np.mean([tprs[(f, t)] for f in fams], axis=0)
    out["mean_auroc"] = np.mean([out[f"auroc_{f}"] for f in fams], axis=0)
    flag = x850 > thr[0.05][:, None]
    out["easr_hb_0.05"] = (hb & ~flag).mean(axis=1)
    out["easr_sr_0.05"] = (sr & ~flag).mean(axis=1)
    return out


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"B": B, "seed": SEED, "interval": "95% percentile", "models": {}}
    for model in ("llama2", "llama3", "mistral", "vicuna"):
        items, dets = load(model)
        rows = lambda p, f=None: np.array([i["row"] for i in items if i["population"] == p
                                           and (f is None or i["family"] == f)])
        cal_rows, ev_rows = np.array(calibration_rows_v2(items)), rows("benign_eval")
        fam_names = sorted({i["family"] for i in items if i["population"] == "release_attack"})
        fam_rows = {f: rows("release_attack", f) for f in fam_names}
        rep = [i for i in items if i["population"] == "saa850_repaired"]
        judged = rd.load_judged(model)
        hb_all = np.array([judged["harmbench"][int(i["source_index"])] for i in rep], bool)
        sr_all = np.array([judged["strongreject"][int(i["source_index"])] > 0.5 for i in rep], bool)
        r850 = np.array([i["row"] for i in rep])
        # one set of resampled indices per model, shared by all detectors (paired)
        ic = rng.integers(0, len(cal_rows), (B, len(cal_rows)))
        ie = rng.integers(0, len(ev_rows), (B, len(ev_rows)))
        ifam = {f: rng.integers(0, len(r), (B, len(r))) for f, r in fam_rows.items()}
        i850 = rng.integers(0, len(r850), (B, len(r850)))
        report["models"][model] = {"n_calibration": int(len(cal_rows)), "n_benign_eval": int(len(ev_rows)),
                                   "family_n": {f: int(len(r)) for f, r in fam_rows.items()}, "detectors": {}}
        reps_by_det = {}
        for det, s in dets.items():
            pt = metrics(s, s[cal_rows][None], s[ev_rows][None], {f: s[r][None] for f, r in fam_rows.items()},
                         s[r850][None], hb_all[None], sr_all[None])
            rp = metrics(s, s[cal_rows][ic], s[ev_rows][ie], {f: s[r][ifam[f]] for f, r in fam_rows.items()},
                         s[r850][i850], hb_all[i850], sr_all[i850])
            reps_by_det[det] = rp
            report["models"][model]["detectors"][det] = {k: summarise(pt[k][0], rp[k]) for k in pt}
            print(model, det, "mean TPR@1%", report["models"][model]["detectors"][det]["mean_tpr_0.01"], flush=True)
        # paired differences against MTK
        diffs = {}
        for det in ("GradSafe", "HiddenDetect"):
            for k in ("mean_tpr_0.01", "mean_tpr_0.05", "tpr_saa_0.05", "mean_auroc"):
                d = reps_by_det[det][k] - reps_by_det["MTK"][k]
                p = report["models"][model]["detectors"][det][k]["point"] - report["models"][model]["detectors"]["MTK"][k]["point"]
                diffs[f"{det}-MTK:{k}"] = summarise(p, d)
        report["models"][model]["paired_differences"] = diffs
    (OUT / "bootstrap.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
