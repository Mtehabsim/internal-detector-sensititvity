"""Are GradSafe's reference sets distinguishable at the operating point? (CPU; saved scores only)

    python scripts/analysis/gradsafe_refset_bootstrap.py      # writes results/uncertainty/gradsafe_refsets_v3.json

v2 added, for every pair, a two-sided bootstrap p-value (2 x the smaller tail share of the paired difference,
floored at 1/B) and Holm's step-down adjustment over the 15 pairs at family-wise alpha 0.05. v3 raises B from
2,000 to 20,000: at 2,000 one borderline pair's p-value (refset3-refset4, near its Holm cutoff) moved with Monte
Carlo noise; at 20,000 the counts are the same across seeds. v1 and v2 files are kept.

A confidence interval for one reference set says nothing about differences between sets, so this compares
them directly. Six reference sets on Llama-2 (the official one and five redraws) score identical prompts. Each
replicate (B=20000, seed 20260925; resampling as bootstrap_ci.py) resamples with replacement the text-disjoint calibration
prompts and the released SAA family's 500 prompts, recomputes every set's 5% threshold from the SAME resampled
calibration prompts, and computes every set's SAA detection on the SAME resampled SAA prompts, so differences
are paired. Reported: each set's 95% interval, and the 95% interval of the difference for all 15 pairs.
Decision rule fixed before running: a pair differs if its interval excludes 0.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2
from mtkaudit.paths import ROOT

B, SEED, ALPHA = 20000, 20260925, 0.05


def main():
    items = json.loads((ROOT / "results/matched_detectors/llama2/items.json").read_text())
    # the six sets this analysis was designed for (official + refset0-refset4); later redraws are analysed by
    # gradsafe_refset_v4.py, so they must not enter here (pinned 2026-09-26, when refset5-refset28 were added)
    files = [ROOT / "results/gradsafe/scores_official.jsonl"] + [ROOT / f"results/gradsafe/scores_refset{k}.jsonl" for k in range(5)]
    names, S = [], []
    for f in files:
        g = {json.loads(l)["row"]: json.loads(l)["score"] for l in f.read_text().splitlines()}
        names.append(f.stem.replace("scores_", ""))
        S.append(np.array([g[i["row"]] for i in items], float))
    S = np.stack(S)                                                   # (sets, items)
    cal = np.array(calibration_rows_v2(items))
    saa = np.array([i["row"] for i in items if i["population"] == "release_attack" and i["family"] == "saa"])
    point = np.array([(S[k, saa] > np.quantile(S[k, cal], 1 - ALPHA)).mean() for k in range(len(S))])
    rng = np.random.default_rng(SEED)
    reps = np.empty((B, len(S)))
    for b in range(B):
        c = cal[rng.integers(0, len(cal), len(cal))]
        a = saa[rng.integers(0, len(saa), len(saa))]
        thr = np.quantile(S[:, c], 1 - ALPHA, axis=1)
        reps[b] = (S[:, a] > thr[:, None]).mean(axis=1)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = {"B": B, "seed": SEED, "budget": ALPHA, "n_calibration": len(cal), "n_saa": len(saa),
           "sets": {n: {"saa_tpr": float(point[k]), "ci": ci(reps[:, k])} for k, n in enumerate(names)},
           "pairs": {}}
    for i, j in itertools.combinations(range(len(S)), 2):
        d = reps[:, i] - reps[:, j]
        lo, hi = ci(d)
        out["pairs"][f"{names[i]}-{names[j]}"] = {"diff": float(point[i] - point[j]), "ci": [lo, hi],
                                                  "excludes_zero": bool(lo > 0 or hi < 0)}
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out["pairs"][f"{names[i]}-{names[j]}"]["p_bootstrap"] = float(min(1.0, max(p, 1 / B)))
    out["n_pairs"] = len(out["pairs"])
    out["n_pairs_differ"] = sum(v["excludes_zero"] for v in out["pairs"].values())
    # Holm step-down over all pairs
    order = sorted(out["pairs"], key=lambda k: out["pairs"][k]["p_bootstrap"])
    stop = False
    for rank, k in enumerate(order):
        thr = 0.05 / (len(order) - rank)
        rej = (not stop) and out["pairs"][k]["p_bootstrap"] <= thr
        stop = stop or not rej
        out["pairs"][k]["holm_reject"] = bool(rej)
        out["pairs"][k]["holm_threshold"] = thr
    out["n_pairs_differ_holm"] = sum(v["holm_reject"] for v in out["pairs"].values())
    (ROOT / "results/uncertainty/gradsafe_refsets_v3.json").write_text(json.dumps(out, indent=1))
    for n, v in out["sets"].items():
        print(f"{n:10s} SAA@5% {v['saa_tpr']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
    for k, v in out["pairs"].items():
        print(f"{k:22s} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]{'  *' if v['excludes_zero'] else ''}")
    print("pairs whose interval excludes 0:", out["n_pairs_differ"], "of", out["n_pairs"],
          "| Holm-adjusted (FWER 0.05):", out["n_pairs_differ_holm"])
    for k, v in out["pairs"].items():
        if k.startswith("official-"):
            print(f"  {k}: p={v['p_bootstrap']:.4f} holm_reject={v['holm_reject']}")


if __name__ == "__main__":
    run(main, __doc__)
