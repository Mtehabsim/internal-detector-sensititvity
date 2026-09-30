"""Does the benign data that sets the threshold explain MTK's 95%-at-5%? (GPU; one model per run)

    CUDA_VISIBLE_DEVICES=0 python scripts/calibration/dolly_calibration.py --model llama2

MTK's paper reports 95% TPR at 5% FPR (its Table 3, GCG on Llama-2) with the threshold fixed at 5% FPR on 200
benign prompts from Databricks-Dolly -- the corpus that also supplies 300 of the bank's benign anchors -- and
evaluated on Dolly benign prompts. We calibrate on held-out ToxicChat (real user traffic). This scores
held-out Dolly prompts with the release's configured bank and forest, and compares detection under the two
thresholds. Design, fixed before running:

  * Dolly pool: the release's databricks-dolly-15k.txt as its reader loads it, minus the configured bank's
    300 Dolly anchors and any prompt whose text equals a bank prompt. 700 prompts drawn with
    random.Random(stable_seed(seed, "calibration:dolly")): the first 200 calibrate (MTK's calibration size),
    the other 500 are the Dolly evaluation benign set (MTK's test set has 500 benign prompts).
  * Scores: the release's anomaly for the configured seed. The Dolly features are scored together with the
    release's own cached test features, and the cached part must reproduce the saved configured-seed scores
    exactly, so the new scores are on the same scale.
  * Report, per family, detection at the 5% threshold from (a) 200 Dolly prompts and (b) our text-disjoint
    ToxicChat set; realised FPR of each threshold on Dolly and on ToxicChat; a bootstrap over the 200 Dolly
    calibration prompts (B=2000) for the Dolly-threshold detection of each family.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from mtkaudit import release as rd

B = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    identity = rd.verify_copy()
    fp = rd.import_release()
    seed = rd.SHIPPED_SEED[a.model]
    out = rd.ROOT / "results" / "dolly_calibration"
    out.mkdir(parents=True, exist_ok=True)
    runner = rd.VicunaModel(fp, a.device) if a.model == "vicuna" else rd.ProtocolModel(fp, a.model, a.device)

    # bank at the configured seed (cached features; the prompts from the release's seeded draw)
    if a.model == "vicuna":
        bank = torch.load(rd.CACHE / f"vicuna_bank_seed{seed}.pt", weights_only=False)
        bank_benign = bank["benign"]
        reader = fp.read_lines
    else:
        bank = torch.load(rd.CACHE / f"{a.model}_bank_seed{seed}.pt", weights_only=False)
        reader = (__import__("extract_trainset_hiddenstates_mistral").read_mistral_training_lines
                  if a.model == "mistral" else fp.read_lines)
        bank_benign = runner.protocol.sample_training(seed)[0]
    labels = bank["labels"]
    dolly_rows = reader(rd.COPY / "datasets/train_data/databricks-dolly-15k.txt")
    in_bank = {rd.sha(t.strip()) for t in bank_benign}
    pool = [i for i, t in enumerate(dolly_rows) if rd.sha(t.strip()) not in in_bank and t.strip()]
    pick = random.Random(fp.stable_seed(seed, "calibration:dolly")).sample(pool, 700)
    texts = [dolly_rows[i] for i in pick]

    # score the Dolly prompts together with the release's cached test set, same bank and forest
    dolly_feats = runner.extract_queries(texts)
    if a.model == "vicuna":
        ts = torch.load(rd.CACHE / "vicuna_test_exact_shipped.pt", weights_only=False)
        test_feats, key = ts["features"], "exact_shipped"
    else:
        saved, _, _ = runner.test_features()
        test_feats, key = saved["features"], "release_cache"
    n_test = (test_feats["colon"] if isinstance(test_feats, dict) else test_feats).shape[0]
    queries = rd.cat_queries(test_feats, [dolly_feats])
    anomaly, _, _ = runner.anomaly(rd.to_gpu(bank["features"], a.device), labels, rd.to_gpu(queries, a.device), seed)
    saved_scores = np.load(rd.OUT / a.model / "scores" / f"as_shipped_{key}_seed{seed}.npy")
    max_diff = float(np.abs(anomaly[:n_test].astype(np.float32) - saved_scores[:n_test]).max())
    assert max_diff < 1e-5, f"rescored release test set differs from the saved scores ({max_diff})"
    s_dolly = anomaly[n_test:].astype(float)
    np.save(out / f"{a.model}_dolly_scores.npy", s_dolly.astype(np.float32))

    # the release's evaluated sets at the configured seed, and our text-disjoint ToxicChat calibration
    lay = json.loads((rd.OUT / a.model / "rows.json").read_text())[key]
    ds = lay["datasets"]
    s = saved_scores.astype(float)
    b = ds[rd.BENIGN_DATASET]
    ev = rd.sampled_positions(fp, b["n"], seed, rd.BENIGN_DATASET)
    cal = rd.calibration_positions_text_disjoint(fp, rd.benign_texts(fp, a.model), ev, seed)
    tc_eval = s[[b["feature_row"][p] for p in ev]]
    thr_tc = float(np.quantile(s[[b["feature_row"][p] for p in cal]], 0.95))
    d_cal, d_eval = s_dolly[:200], s_dolly[200:]
    thr_d = float(np.quantile(d_cal, 0.95))
    fams = {n: s[[info["feature_row"][p] for p in rd.sampled_positions(fp, info["n"], seed, n)]]
            for n, info in ds.items() if n != rd.BENIGN_DATASET}
    rng = np.random.default_rng(20260925)
    boot_thr = np.quantile(d_cal[rng.integers(0, 200, (B, 200))], 0.95, axis=1)
    rep = {"model": a.model, "seed": seed, "release_commit": identity["commit"],
           "rescore_check_max_abs_diff": max_diff, "n_dolly_calibration": 200, "n_dolly_eval": 500,
           "threshold_dolly": thr_d, "threshold_toxicchat": thr_tc,
           "fpr": {"dolly_threshold_on_dolly_eval": float((d_eval > thr_d).mean()),
                   "dolly_threshold_on_toxicchat_eval": float((tc_eval > thr_d).mean()),
                   "toxicchat_threshold_on_toxicchat_eval": float((tc_eval > thr_tc).mean()),
                   "toxicchat_threshold_on_dolly_eval": float((d_eval > thr_tc).mean())},
           "families": {}}
    for n, x in fams.items():
        bt = (x[None, :] > boot_thr[:, None]).mean(axis=1)
        rep["families"][n] = {"n": len(x), "tpr_dolly_threshold": float((x > thr_d).mean()),
                              "tpr_dolly_threshold_ci": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                              "tpr_toxicchat_threshold": float((x > thr_tc).mean())}
    for key_ in ("tpr_dolly_threshold", "tpr_toxicchat_threshold"):
        rep[f"mean_{key_}"] = float(np.mean([v[key_] for v in rep["families"].values()]))
    rep["dolly_prompt_positions"] = [int(i) for i in pick]
    (out / f"{a.model}.json").write_text(json.dumps(rep, indent=1))
    f = rep["families"]
    gcg = "nanogcg" if "nanogcg" in f else "nonagcg"
    print(f"{a.model}: rescore diff {max_diff:.1e} | thr Dolly {thr_d:+.3f} vs ToxicChat {thr_tc:+.3f} | "
          f"FPR(Dolly thr) Dolly {rep['fpr']['dolly_threshold_on_dolly_eval']:.3f} ToxicChat "
          f"{rep['fpr']['dolly_threshold_on_toxicchat_eval']:.3f} | nanoGCG {f[gcg]['tpr_dolly_threshold']:.3f} "
          f"(vs {f[gcg]['tpr_toxicchat_threshold']:.3f}) | mean {rep['mean_tpr_dolly_threshold']:.3f} "
          f"(vs {rep['mean_tpr_toxicchat_threshold']:.3f}) | SAA {f['saa']['tpr_dolly_threshold']:.3f}", flush=True)


if __name__ == "__main__":
    main()
