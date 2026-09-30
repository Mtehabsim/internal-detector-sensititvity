"""The GradSafe noise test applied to MTK's bank-only redraws (CPU; saved scores).

    python scripts/analysis/mtk_bank_noise.py      # writes results/uncertainty/mtk_bank_noise.json

Design fixed before computing: paper/PREREG_20260926_extensions.md, section D. Per model, the draws of Table IV's lower
block (bank redrawn; evaluation sample and forest at the configured seed): SAA detection at the text-disjoint real-user
5% threshold, recomputed from the saved scores and required to equal each draw's recorded sweep_v2 metric.
Ratio = between-draw SD of the point TPR / root mean within-draw bootstrap variance (B = 2,000, seed 20260926; the
calibration prompts and the 500 SAA prompts resampled once per replicate for all draws of a model, thresholds
recomputed). Above 2: variation across banks exceeds sampling noise; RMS within-draw SD below 0.005: nothing to test.
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit import release as rd
from mtkaudit import results as mt

B, SEED = 2000, 20260926


def main():
    fp = rd.import_release()
    out = {"design": "paper/PREREG_20260926_extensions.md, section D", "B": B, "seed": SEED, "models": {}}
    rng = np.random.default_rng(SEED)
    for model in mt.MODELS:
        ship, key = rd.SHIPPED_SEED[model], mt.PRIMARY_KEY[model]
        lay = json.loads((rd.OUT / model / "rows.json").read_text())[key]
        ds = lay["datasets"]
        b = ds[rd.BENIGN_DATASET]
        ev = rd.sampled_positions(fp, b["n"], ship, rd.BENIGN_DATASET)
        cal = np.array(rd.rows_of(b, rd.calibration_positions_text_disjoint(fp, rd.benign_texts(fp, model), ev, ship)))
        saa = np.array(rd.rows_of(ds["saa"], rd.sampled_positions(fp, ds["saa"]["n"], ship, "saa")))
        recs = [json.loads(l) for l in (rd.OUT / model / "sweep_v2.jsonl").read_text().splitlines()]
        draws = mt.by_seed(recs, model, "bank_only")
        ic, isa = rng.integers(0, len(cal), (B, len(cal))), rng.integers(0, len(saa), (B, len(saa)))
        point, within = {}, {}
        for s in sorted(draws):
            tag = f"as_shipped_{key}_seed{s}" if s == ship else f"bank_only_{key}_seed{s}"
            sc = np.load(rd.OUT / model / "scores" / f"{tag}.npy").astype(float)
            thr = float(np.quantile(sc[cal], 0.95))
            tpr = float((sc[saa] > thr).mean())
            rec = draws[s]["families"]["saa"]["tpr"]["0.05"]
            if abs(tpr - rec) > 1e-9:
                raise SystemExit(f"{model} seed {s}: recomputed SAA TPR {tpr} differs from the record {rec}")
            point[s] = tpr
            bt = np.quantile(sc[cal][ic], 0.95, axis=1)
            within[s] = float(((sc[saa][isa] > bt[:, None]).mean(axis=1)).var(ddof=1))
        p = np.array(list(point.values()))
        between_sd, within_rms = float(p.std(ddof=1)), float(np.sqrt(np.mean(list(within.values()))))
        testable = within_rms >= 0.005
        ratio = between_sd / within_rms if testable else None
        out["models"][model] = {"n_draws": len(point), "saa_tpr_min": float(p.min()), "saa_tpr_median": float(np.median(p)),
                                "saa_tpr_max": float(p.max()), "between_draw_sd": between_sd,
                                "within_draw_bootstrap_sd_rms": within_rms, "testable": testable, "ratio": ratio,
                                "exceeds_sampling_noise": bool(testable and ratio > 2), "per_draw": point}
        print(model, {k: v for k, v in out["models"][model].items() if k != "per_draw"}, flush=True)
    (rd.ROOT / "results/uncertainty/mtk_bank_noise.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
