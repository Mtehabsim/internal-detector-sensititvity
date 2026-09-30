"""Does AUROC predict majority detection for concentrated families, beyond MTK? (CPU; saved scores only)

    python scripts/analysis/rule_crossdetector.py     # writes results/rule_test/cross_detector.json

For a family whose scores sit at one point, AUROC is the share of benign prompts below that point, so most of
the family is caught at budget alpha exactly when AUROC > 1 - alpha. Real families are not points. This tests the
rule on every detector (MTK, GradSafe, HiddenDetect, the probe, windowed perplexity), model and released family of
at least 50 prompts, at budgets 1-5% and 10%, with the text-disjoint ToxicChat calibration. Prediction: "most of
the family is detected" iff AUROC > 1 - alpha; outcome: detection > 0.5 at that budget. Concentration: the family's
score interquartile range divided by the evaluated benign prompts' interquartile range. Bins fixed in advance:
< 0.5 (concentrated), 0.5-1, >= 1 (spread).
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2
from mtkaudit.paths import ROOT

ALPHAS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.10)


def main():
    cells = []
    for m in mt.MODELS:
        items, dets = mt.matched_scores(m)
        cal = calibration_rows_v2(items)
        ev = [i["row"] for i in items if i["population"] == "benign_eval"]
        fams = sorted({i["family"] for i in items if i["population"] == "release_attack"})
        for d, s in dets.items():
            s = np.asarray(s, float)
            b = s[ev]
            biqr = np.subtract(*np.percentile(b, [75, 25]))
            for f in fams:
                x = s[[i["row"] for i in items if i["population"] == "release_attack" and i["family"] == f]]
                if len(x) < 50:
                    continue
                a = float(roc_auc_score(np.r_[np.zeros(len(b)), np.ones(len(x))], np.r_[b, x]))
                conc = float(np.subtract(*np.percentile(x, [75, 25])) / biqr)
                for al in ALPHAS:
                    tpr = float((x > np.quantile(s[cal], 1 - al)).mean())
                    cells.append({"model": m, "detector": d, "family": f, "alpha": al, "auroc": a,
                                  "concentration": conc, "tpr": tpr, "agree": (a > 1 - al) == (tpr > 0.5)})
    agree = lambda sel: {"n": len(sel), "agreement": float(np.mean([c["agree"] for c in sel])) if sel else None}
    out = {"n_cells": len(cells), "overall": agree(cells),
           "by_concentration": {"lt_0.5": agree([c for c in cells if c["concentration"] < 0.5]),
                                "0.5_to_1": agree([c for c in cells if 0.5 <= c["concentration"] < 1]),
                                "ge_1": agree([c for c in cells if c["concentration"] >= 1])},
           "by_detector": {d: agree([c for c in cells if c["detector"] == d]) for d in sorted({c["detector"] for c in cells})},
           "cells": cells}
    p = ROOT / "results/rule_test"
    p.mkdir(parents=True, exist_ok=True)
    (p / "cross_detector.json").write_text(json.dumps(out, indent=1))
    print("cells", out["n_cells"], "overall", round(out["overall"]["agreement"], 3))
    for k, v in out["by_concentration"].items():
        print(f"  concentration {k}: n={v['n']} agreement {v['agreement']:.3f}")
    for k, v in out["by_detector"].items():
        print(f"  {k}: n={v['n']} agreement {v['agreement']:.3f}")


if __name__ == "__main__":
    run(main, __doc__)
