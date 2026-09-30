"""AUROC, partial AUROC and detection at the operating point, per detector, model and family (CPU; saved scores).

    python scripts/analysis/pauc_vs_tpr.py      # writes results/rule_test/pauc_vs_tpr.json

Design fixed before running: paper/PREREG_20260926_extensions.md, section B. Points: every detector, model and
release family of 50 or more prompts on the matched prompts, at the text-disjoint real-user calibration (Table VI's
setting; AUROC and TPR@5% from the generator's own budget_metrics). Partial AUROC: McClish-standardised area over
FPR in [0, 0.05] against the 500 evaluated benign prompts (sklearn roc_auc_score(max_fpr=0.05)). Reported: Spearman
rho of AUROC and of partial AUROC with TPR@5%, over all points and over the model-internal detectors; the points
with AUROC >= 0.85 and TPR < 0.2, and how many of those have partial AUROC above 0.75; Llama-3 MTK SAA's value.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from mtkaudit import results as mt
from mtkaudit.cli import run

OUT = mt.RES / "rule_test" / "pauc_vs_tpr.json"


def main():
    points = []
    for m in mt.MODELS:
        for d, _ in mt.DETECTORS:
            items, s = mt.matched_scores_one(m, d)
            if s is None:
                continue
            b = mt.budget_metrics(m, items, s)
            ev = s[[i["row"] for i in items if i["population"] == "benign_eval"]]
            for f, v in b["families"].items():
                if v["n"] < 50:
                    continue
                x = s[[i["row"] for i in items if i["population"] == "release_attack" and i["family"] == f]]
                y = np.r_[np.zeros(len(ev)), np.ones(len(x))]
                points.append({"model": m, "detector": d, "family": f, "n": v["n"], "auroc": v["auroc"],
                               "pauroc_0.05": float(roc_auc_score(y, np.r_[ev, x], max_fpr=0.05)),
                               "tpr_0.05": v["tpr"][0.05]})

    def rho(pts, key):
        return float(spearmanr([p[key] for p in pts], [p["tpr_0.05"] for p in pts]).correlation)

    internal = [p for p in points if p["detector"] != "Windowed PPL"]
    hidden = [p for p in points if p["auroc"] >= 0.85 and p["tpr_0.05"] < 0.2]
    l3 = next(p for p in points if p["model"] == "llama3" and p["detector"] == "MTK" and p["family"] == "saa")
    rep = {"design": "paper/PREREG_20260926_extensions.md, section B", "n_points": len(points),
           "n_points_internal": len(internal),
           "spearman_auroc_vs_tpr": rho(points, "auroc"), "spearman_pauroc_vs_tpr": rho(points, "pauroc_0.05"),
           "spearman_auroc_vs_tpr_internal": rho(internal, "auroc"),
           "spearman_pauroc_vs_tpr_internal": rho(internal, "pauroc_0.05"),
           "high_auroc_low_tpr": len(hidden),
           "high_auroc_low_tpr_with_pauroc_above_0.75": sum(p["pauroc_0.05"] > 0.75 for p in hidden),
           "llama3_mtk_saa": l3, "points": points}
    OUT.write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "points"})


if __name__ == "__main__":
    run(main, __doc__)
