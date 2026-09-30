"""Equal-size control for the Dolly vs ToxicChat comparison (CPU; saved scores only).

    python scripts/calibration/calib_size_control.py     # writes results/dolly_calibration/size_control.json

The Dolly thresholds rest on 199-200 prompts, the real-traffic thresholds on about 990. To separate the benign
distribution from the calibration-set size, each replicate draws, without replacement, as many prompts from the
text-disjoint ToxicChat calibration set as the model's Dolly calibration set holds, sets each detector's 5%
threshold on them, and evaluates it on the unchanged 500 evaluated ToxicChat prompts and the twelve families.
Design fixed before running: B=2000, seed 20260925, the same subsample indices for every detector of a model.
Reported per detector and model: median and 95% range of the realised ToxicChat FPR and of mean detection, the
Dolly threshold's FPR on the same prompts (summary_v2.json), and the share of subsamples whose FPR reaches it.
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2
from mtkaudit.paths import ROOT

D = ROOT / "results/dolly_calibration"
B, SEED = 2000, 20260925


def main():
    sm = json.loads((D / "summary_v2.json").read_text())
    rng = np.random.default_rng(SEED)
    out = {"B": B, "seed": SEED, "models": {}}
    for m in mt.MODELS:
        items, dets = mt.matched_scores(m)
        dets = dict(dets)
        dets["Probe"] = np.load(ROOT / "results/matched_detectors" / m / "probe.npy")
        cal = np.array(calibration_rows_v2(items))
        ev = [i["row"] for i in items if i["population"] == "benign_eval"]
        fams = sorted({i["family"] for i in items if i["population"] == "release_attack"})
        fam_rows = {f: [i["row"] for i in items if i["population"] == "release_attack" and i["family"] == f] for f in fams}
        n = sm[m]["_split"]["n_calibration"]
        idx = np.stack([rng.choice(len(cal), n, replace=False) for _ in range(B)])
        out[m] = None
        out["models"][m] = {"n_subsample": int(n), "n_toxicchat_calibration": len(cal), "detectors": {}}
        for det in ("MTK", "Probe", "GradSafe", "HiddenDetect", "Windowed PPL"):
            s = np.asarray(dets[det], float)
            thr = np.quantile(s[cal][idx], 0.95, axis=1)                         # (B,)
            fpr = (s[ev][None, :] > thr[:, None]).mean(axis=1)
            det_mean = np.mean([(s[fam_rows[f]][None, :] > thr[:, None]).mean(axis=1) for f in fams], axis=0)
            dolly_fpr = sm[m][det]["dolly"]["fpr_toxicchat_eval"]
            q = lambda x: [float(np.percentile(x, 2.5)), float(np.median(x)), float(np.percentile(x, 97.5))]
            out["models"][m]["detectors"][det] = {
                "fpr_toxicchat_eval": q(fpr), "mean_detection": q(det_mean),
                "dolly_threshold_fpr_toxicchat_eval": dolly_fpr,
                "dolly_threshold_mean_detection": sm[m][det]["dolly"]["mean_tpr"],
                "share_subsamples_fpr_at_least_dolly": float((fpr >= dolly_fpr).mean())}
            r = out["models"][m]["detectors"][det]
            print(f"{m:8s} {det:13s} n={n}: ToxicChat-200 FPR {r['fpr_toxicchat_eval'][1]:.3f} "
                  f"[{r['fpr_toxicchat_eval'][0]:.3f}, {r['fpr_toxicchat_eval'][2]:.3f}] vs Dolly {dolly_fpr:.3f} "
                  f"(share >= Dolly {r['share_subsamples_fpr_at_least_dolly']:.4f}) | detection "
                  f"{r['mean_detection'][1]:.3f} vs Dolly {r['dolly_threshold_mean_detection']:.3f}")
        del out[m]
    (D / "size_control.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
