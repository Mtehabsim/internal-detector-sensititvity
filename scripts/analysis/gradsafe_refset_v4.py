"""Thirty GradSafe reference sets on Llama-2: AUROC against detection at the operating point (CPU; saved scores).

    python scripts/analysis/gradsafe_refset_v4.py      # writes results/uncertainty/gradsafe_refsets_v4.json

Design fixed before the new draws finished: paper/PREREG_20260926_extensions.md, section A. Sets: the official
reference and every redraw (refset0-refset28), all scored on identical prompts by the same code path. Per set, at the
text-disjoint real-user 5% threshold (the generator's budget_metrics): mean AUROC over the twelve families, SAA AUROC,
SAA TPR, the worst-family TPR (families of 50 or more) and the realised FPR. Reported: min / median / max; sets with
SAA TPR below 0.2 and above 0.5; Spearman rho between mean AUROC and SAA TPR across sets. Reference variation against
sampling noise: B = 2,000 paired replicates (seed 20260926) resample the calibration prompts and the 500 SAA prompts
once for all sets and recompute every threshold; ratio = between-set SD of the point SAA TPR / root mean within-set
bootstrap variance; a ratio above 2 is reported as variation exceeding sampling noise. v3 (the six-set pairwise
analysis) is kept unchanged.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.matched import calibration_rows_v2

B, SEED = 2000, 20260926


def main():
    items = json.loads((mt.RES / "matched_detectors/llama2/items.json").read_text())
    # exactly the 30 sets of the design (official + refset0-refset28); a later file must not change the cohort
    files = [mt.RES / "gradsafe/scores_official.jsonl"] + [mt.RES / f"gradsafe/scores_refset{k}.jsonl" for k in range(29)]
    names, S = [], []
    for f in files:
        g = {json.loads(l)["row"]: json.loads(l)["score"] for l in f.read_text().splitlines()}
        if len(g) != len(items):
            raise SystemExit(f"{f.name} is incomplete ({len(g)} of {len(items)} prompts)")
        names.append(f.stem.replace("scores_", ""))
        S.append(np.array([g[i["row"]] for i in items], float))
    S = np.stack(S)
    per = {}
    for n, s in zip(names, S):
        b = mt.budget_metrics("llama2", items, s)
        big = {f: v["tpr"][0.05] for f, v in b["families"].items() if v["n"] >= 50}
        per[n] = {"mean_auroc": b["mean_auroc"], "saa_auroc": b["saa_auroc"], "saa_tpr": b["saa_tpr"][0.05],
                  "worst_tpr": min(big.values()), "worst_family": min(big, key=big.get), "fpr": b["fpr"][0.05]}
    cal = np.array(calibration_rows_v2(items))
    saa = np.array([i["row"] for i in items if i["population"] == "release_attack" and i["family"] == "saa"])
    rng = np.random.default_rng(SEED)
    ic, isa = rng.integers(0, len(cal), (B, len(cal))), rng.integers(0, len(saa), (B, len(saa)))
    boot = np.empty((len(names), B))
    for k, s in enumerate(S):
        thr = np.quantile(s[cal][ic], 0.95, axis=1)
        boot[k] = (s[saa][isa] > thr[:, None]).mean(axis=1)
    point = np.array([per[n]["saa_tpr"] for n in names])
    between_sd = float(point.std(ddof=1))
    within_rms = float(np.sqrt(boot.var(axis=1, ddof=1).mean()))
    col = lambda k: [per[n][k] for n in names]
    summ = lambda v: {"min": float(min(v)), "median": float(np.median(v)), "max": float(max(v))}
    rep = {"design": "paper/PREREG_20260926_extensions.md, section A", "n_sets": len(names), "B": B, "seed": SEED,
           "mean_auroc": summ(col("mean_auroc")), "saa_auroc": summ(col("saa_auroc")), "saa_tpr": summ(col("saa_tpr")),
           "worst_tpr": summ(col("worst_tpr")), "fpr": summ(col("fpr")),
           "sets_saa_tpr_below_0.2": int(sum(v < 0.2 for v in col("saa_tpr"))),
           "sets_saa_tpr_above_0.5": int(sum(v > 0.5 for v in col("saa_tpr"))),
           "spearman_mean_auroc_vs_saa_tpr": float(spearmanr(col("mean_auroc"), col("saa_tpr")).correlation),
           "between_set_sd_saa_tpr": between_sd, "within_set_bootstrap_sd_rms": within_rms,
           "ratio": between_sd / within_rms, "exceeds_sampling_noise": between_sd / within_rms > 2, "sets": per}
    (mt.RES / "uncertainty/gradsafe_refsets_v4.json").write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "sets"})


if __name__ == "__main__":
    run(main, __doc__)
