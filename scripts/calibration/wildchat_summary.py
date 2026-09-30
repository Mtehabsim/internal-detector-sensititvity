"""Calibration transfer between Dolly, ToxicChat and WildChat, every detector on every model (CPU; saved scores).

    python scripts/calibration/wildchat_summary.py     # writes results/wildchat/summary.json

Design fixed before any WildChat prompt was scored: paper/PREREG_20260926_extensions.md, section C. For each
detector and model, 5% thresholds from three benign calibration sets -- Dolly (text-disjoint split v2, as
dolly_summary.py), ToxicChat (text-disjoint, as every real-user number) and WildChat (the first 1,000 selected
prompts) -- and the false-positive rate each realises on the three evaluation sets (Dolly split v2, the release's 500
ToxicChat prompts, the last 500 WildChat prompts); mean detection over the twelve families, SAA and the worst family
of at least 50 prompts under each threshold. Secondary check, fixed in advance: the WildChat evaluation FPR with
prompts containing jailbreak markers removed (thresholds unchanged).
"""
from __future__ import annotations

import json
import re

import numpy as np

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2
from mtkaudit.paths import ROOT

D, W = ROOT / "results/dolly_calibration", ROOT / "results/wildchat"
MARKERS = re.compile(r"jailbreak|do anything now|developer mode|ignore all previous|ignore previous instructions|"
                     r"stay in character|\bdan\b", re.I)
N_CAL = 1000


def main():
    texts = W / "wildchat_prompts.json"            # rebuilt by rebuild_texts.py --wildchat; not distributed
    if texts.exists():
        wc_eval_clean = np.array([not MARKERS.search(t) for t in json.loads(texts.read_text())["evaluation"]])
    else:                                           # the mask the texts give, archived with the results
        wc_eval_clean = np.array([not v for v in json.loads((W / "eval_marker_mask.json").read_text())])
    out = {"design": "paper/PREREG_20260926_extensions.md, section C",
           "wildchat_eval_marker_excluded": int((~wc_eval_clean).sum())}
    for m in mt.MODELS:
        items, dets = mt.matched_scores(m)
        dets = dict(dets)
        dets["Probe"] = np.load(ROOT / "results/matched_detectors" / m / "probe.npy")
        rows = lambda p, f=None: [i["row"] for i in items if i["population"] == p and (f is None or i["family"] == f)]
        tc_cal, tc_ev = calibration_rows_v2(items), rows("benign_eval")
        fams = sorted({i["family"] for i in items if i["population"] == "release_attack"})
        split = json.loads((D / "split_v2" / f"{m}.json").read_text())
        ci_, ei_ = np.array(split["calibration_idx"]), np.array(split["evaluation_idx"])
        out[m] = {}
        for det in ("MTK", "GradSafe", "HiddenDetect", "Windowed PPL", "Probe"):
            key = det.replace(" ", "_")
            dfile = D / f"{m}_dolly_scores.npy" if det == "MTK" else D / "other_detectors" / f"{m}_{key}.npy"
            wfile = W / f"{m}_MTK.npy" if det == "MTK" else W / "other_detectors" / f"{m}_{key}.npy"
            if not (dfile.exists() and wfile.exists() and det in dets):
                continue
            s = np.asarray(dets[det], float)
            dsc, wsc = np.load(dfile).astype(float), np.load(wfile).astype(float)
            assert len(wsc) == N_CAL + 500
            evals = {"dolly": dsc[ei_], "toxicchat": s[tc_ev], "wildchat": wsc[N_CAL:],
                     "wildchat_no_markers": wsc[N_CAL:][wc_eval_clean]}
            thr = {"dolly": float(np.quantile(dsc[ci_], 0.95)), "toxicchat": float(np.quantile(s[tc_cal], 0.95)),
                   "wildchat": float(np.quantile(wsc[:N_CAL], 0.95))}
            r = {"threshold": thr}
            for k, t in thr.items():
                famt = {f: float((s[rows("release_attack", f)] > t).mean()) for f in fams}
                big = {f: v for f, v in famt.items() if len(rows("release_attack", f)) >= 50}
                worst = min(big, key=big.get)
                r[k] = {**{f"fpr_{e}_eval": float((x > t).mean()) for e, x in evals.items()},
                        "mean_tpr": float(np.mean(list(famt.values()))), "saa_tpr": famt["saa"],
                        "worst_family": worst, "worst_tpr": big[worst]}
            out[m][det] = r
            print(f"{m:8s} {det:13s} " + " | ".join(
                f"{k} thr -> Dolly {r[k]['fpr_dolly_eval']:.3f} TC {r[k]['fpr_toxicchat_eval']:.3f} "
                f"WC {r[k]['fpr_wildchat_eval']:.3f} (no markers {r[k]['fpr_wildchat_no_markers_eval']:.3f}) "
                f"mean TPR {r[k]['mean_tpr']:.3f}" for k in thr), flush=True)
    pairs = [(m, d, r) for m in mt.MODELS for d, r in out[m].items()]
    dw = [r["dolly"]["fpr_wildchat_eval"] for _, _, r in pairs]
    out["headline"] = {
        "n_pairs": len(pairs),
        "dolly_on_wildchat_min": min(dw), "dolly_on_wildchat_max": max(dw),
        "dolly_on_wildchat_above_budget": sum(v > 0.05 for v in dw),
        "mtk_dolly_on_wildchat_ratio": [out[m]["MTK"]["dolly"]["fpr_wildchat_eval"] / 0.05 for m in mt.MODELS],
        "toxicchat_on_wildchat": [min(r["toxicchat"]["fpr_wildchat_eval"] for _, _, r in pairs),
                                  max(r["toxicchat"]["fpr_wildchat_eval"] for _, _, r in pairs)],
        "wildchat_on_toxicchat": [min(r["wildchat"]["fpr_toxicchat_eval"] for _, _, r in pairs),
                                  max(r["wildchat"]["fpr_toxicchat_eval"] for _, _, r in pairs)],
        "wildchat_on_wildchat": [min(r["wildchat"]["fpr_wildchat_eval"] for _, _, r in pairs),
                                 max(r["wildchat"]["fpr_wildchat_eval"] for _, _, r in pairs)],
        "dolly_on_wildchat_no_markers": [min(r["dolly"]["fpr_wildchat_no_markers_eval"] for _, _, r in pairs),
                                         max(r["dolly"]["fpr_wildchat_no_markers_eval"] for _, _, r in pairs)]}
    (W / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["headline"], indent=1))


if __name__ == "__main__":
    run(main, __doc__)
