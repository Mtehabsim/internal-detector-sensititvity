"""How often would a detector with the same per-family AUROCs, but spread-out scores, miss some family? (CPU)

    python scripts/analysis/draw_miss_baseline.py     # writes results/rule_test/draw_miss_baseline_v2.json

Every one of the 202 re-seeded MTK runs has some family of at least 50 prompts detected below 0.2 at the 5%
budget. A minimum over twelve families is naturally low, so this computes an illustrative reference: an
equal-variance Gaussian (binormal) detector with the run's per-family AUROCs -- benign scores N(0, 1), a family's
scores N(d', 1) with d' = sqrt(2) * Phi^-1(AUROC). It is a simple reference model, not a claim about real score
distributions or about every detector with spread-out scores. Design: for each run, 2,000 simulations (seed
20260925); each draws one calibration sample of the run's calibration size, whose 95% quantile is the threshold
SHARED by all the run's families of at least 50 prompts (as in the evaluated detector), and each family's attack
scores at its size; a family is missed if detection < 0.2, and the run misses some family if any is. v2 fixes v1
(draw_miss_baseline.json, kept), which drew a separate threshold per family and combined families as independent;
the difference is small. Reported: the expected number of runs with a miss under the reference, against the observed
number, and the runs whose observed miss occurs at AUROC >= 0.85.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm

from mtkaudit import results as mt
from mtkaudit.cli import run
from mtkaudit.paths import ROOT

SIMS, SEED, ALPHA, MISS = 2000, 20260925, 0.05, 0.2


def p_any_miss(fams, n_cal: int, rng) -> float:
    """fams: [(auroc, n)]. One shared threshold per simulation, as in the evaluated detector."""
    thr = np.quantile(rng.standard_normal((SIMS, n_cal)), 1 - ALPHA, axis=1)          # (SIMS,)
    miss = np.zeros(SIMS, bool)
    for auroc, n in fams:
        d = np.sqrt(2) * norm.ppf(min(max(auroc, 1e-6), 1 - 1e-6))
        x = rng.standard_normal((SIMS, n)) + d
        miss |= (x > thr[:, None]).mean(axis=1) < MISS
    return float(miss.mean())


def main():
    rng = np.random.default_rng(SEED)
    out = {"sims": SIMS, "seed": SEED, "budget": ALPHA, "miss_below": MISS, "variants": {}}
    for variant in ("as_shipped", "bank_only"):
        runs = []
        for m in mt.MODELS:
            recs = [json.loads(l) for l in (mt.RP / m / "sweep_v2.jsonl").read_text().splitlines()]
            for seed, met in sorted(mt.by_seed(recs, m, variant).items()):
                fams = [(f, v) for f, v in met["families"].items() if v["n"] >= 50]
                p_any = p_any_miss([(v["auroc"], v["n"]) for _, v in fams], met["n_calibration"], rng)
                observed = [(f, v["auroc"], v["tpr"]["0.05"]) for f, v in fams if v["tpr"]["0.05"] < MISS]
                runs.append({"model": m, "seed": seed, "p_any_miss_binormal": p_any,
                             "observed_miss": bool(observed),
                             "observed_miss_at_auroc_ge_0.85": any(a >= 0.85 for _, a, _ in observed),
                             "observed_missed": observed})
        exp = float(sum(r["p_any_miss_binormal"] for r in runs))
        out["variants"][variant] = {
            "n_runs": len(runs), "observed_runs_with_miss": sum(r["observed_miss"] for r in runs),
            "expected_runs_with_miss_binormal": exp,
            "runs_with_miss_at_auroc_ge_0.85": sum(r["observed_miss_at_auroc_ge_0.85"] for r in runs),
            "runs_binormal_p_ge_0.5": sum(r["p_any_miss_binormal"] >= 0.5 for r in runs),
            "by_model": {m: {"n": sum(r["model"] == m for r in runs),
                             "observed": sum(r["observed_miss"] for r in runs if r["model"] == m),
                             "expected_binormal": float(sum(r["p_any_miss_binormal"] for r in runs if r["model"] == m)),
                             "miss_at_auroc_ge_0.85": sum(r["observed_miss_at_auroc_ge_0.85"] for r in runs if r["model"] == m)}
                         for m in mt.MODELS},
            "runs": runs}
        v = out["variants"][variant]
        print(f"{variant}: observed {v['observed_runs_with_miss']}/{v['n_runs']} runs miss some family; binormal "
              f"expectation {exp:.1f}; runs whose miss is at AUROC >= 0.85: {v['runs_with_miss_at_auroc_ge_0.85']}")
        for m, b in v["by_model"].items():
            print(f"   {m:8s} observed {b['observed']}/{b['n']}  binormal {b['expected_binormal']:.1f}  "
                  f"miss at AUROC>=0.85: {b['miss_at_auroc_ge_0.85']}")
    (ROOT / "results/rule_test/draw_miss_baseline_v2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
