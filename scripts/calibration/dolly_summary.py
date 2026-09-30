"""Dolly-calibrated vs real-traffic-calibrated operating points, every detector on every model (CPU).

    python scripts/calibration/dolly_summary.py        # writes results/dolly_calibration/summary_v2.json

Uses the text-disjoint Dolly split of dolly_split_v2.py (split_v2/<model>.json: 199-200 calibration and 497-500
evaluation prompts). summary.json, written by the first version with the raw first-200 / last-500 split, is kept
as a record and superseded.

For each detector and model: the 5% threshold from 200 held-out Dolly prompts (MTK's calibration protocol)
and from our text-disjoint ToxicChat calibration set; the false-positive rate each realises on 500 further
Dolly prompts and on the release's 500 evaluated ToxicChat prompts; mean detection over the twelve families,
SAA, and the worst family of at least 50 prompts. A bootstrap over the 200 Dolly calibration prompts (B=2000,
threshold recomputed) gives the interval for the Dolly threshold's false-positive rate on ToxicChat.
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.paths import ROOT

D = ROOT / "results/dolly_calibration"
B = 2000


def main():
    rng = np.random.default_rng(20260925)
    out = {}
    for m in mt.MODELS:
        items, dets = mt.matched_scores(m)
        dets = dict(dets)
        dets["Probe"] = np.load(ROOT / "results/matched_detectors" / m / "probe.npy")
        dolly = {"MTK": np.load(D / f"{m}_dolly_scores.npy").astype(float)}
        for k in ("GradSafe", "HiddenDetect", "Windowed_PPL", "Probe"):
            f = D / "other_detectors" / f"{m}_{k}.npy"
            if f.exists():
                dolly[k.replace("_", " ")] = np.load(f).astype(float)
        from mtkaudit.matched import calibration_rows_v2
        rows = lambda p, f=None: [i["row"] for i in items if i["population"] == p and (f is None or i["family"] == f)]
        tc_cal, tc_ev = rows("benign_calibration"), rows("benign_eval")
        tc_cal = calibration_rows_v2(items)
        fams = sorted({i["family"] for i in items if i["population"] == "release_attack"})
        split = json.loads((D / "split_v2" / f"{m}.json").read_text())
        ci_, ei_ = np.array(split["calibration_idx"]), np.array(split["evaluation_idx"])
        out[m] = {"_split": {"n_calibration": len(ci_), "n_evaluation": len(ei_)}}
        for det, dsc in dolly.items():
            dsc = np.asarray(dsc, float)
            s = np.asarray(dets[det], float)
            d_cal, d_ev = dsc[ci_], dsc[ei_]
            thr = {"dolly": float(np.quantile(d_cal, 0.95)), "toxicchat": float(np.quantile(s[tc_cal], 0.95))}
            r = {"threshold": thr}
            for k, t in thr.items():
                famt = {f: float((s[rows("release_attack", f)] > t).mean()) for f in fams}
                big = {f: v for f, v in famt.items() if len(rows("release_attack", f)) >= 50}
                worst = min(big, key=big.get)
                r[k] = {"fpr_dolly_eval": float((d_ev > t).mean()), "fpr_toxicchat_eval": float((s[tc_ev] > t).mean()),
                        "mean_tpr": float(np.mean(list(famt.values()))), "saa_tpr": famt["saa"],
                        "worst_family": worst, "worst_tpr": big[worst], "families": famt,
                        "family_detected": {f: int((s[rows("release_attack", f)] > t).sum()) for f in fams},
                        "family_n": {f: len(rows("release_attack", f)) for f in fams}}
            bt = np.quantile(d_cal[rng.integers(0, len(d_cal), (B, len(d_cal)))], 0.95, axis=1)
            tc = s[tc_ev]
            fprs = (tc[None, :] > bt[:, None]).mean(axis=1)
            r["dolly"]["fpr_toxicchat_eval_ci"] = [float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5))]
            out[m][det] = r
            dd, tt = r["dolly"], r["toxicchat"]
            print(f"{m:8s} {det:13s} Dolly thr: FPR Dolly {dd['fpr_dolly_eval']:.3f} ToxicChat {dd['fpr_toxicchat_eval']:.3f} "
                  f"[{dd['fpr_toxicchat_eval_ci'][0]:.2f},{dd['fpr_toxicchat_eval_ci'][1]:.2f}] mean {dd['mean_tpr']:.3f} "
                  f"SAA {dd['saa_tpr']:.3f} worst {mt.FAM[dd['worst_family']]} {dd['worst_tpr']:.3f} || ToxicChat thr: "
                  f"FPR {tt['fpr_toxicchat_eval']:.3f} mean {tt['mean_tpr']:.3f} SAA {tt['saa_tpr']:.3f} "
                  f"worst {mt.FAM[tt['worst_family']]} {tt['worst_tpr']:.3f}", flush=True)
    (D / "summary_v2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
