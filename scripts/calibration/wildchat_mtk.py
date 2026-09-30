"""MTK's scores for the WildChat benign prompts (GPU; one model per run).

    CUDA_VISIBLE_DEVICES=0 python scripts/calibration/wildchat_mtk.py --model llama2     # writes results/wildchat/<model>_MTK.npy

Design: paper/PREREG_20260926_extensions.md, section C. As dolly_calibration.py does for Dolly: the 1,500 selected
WildChat prompts (results/wildchat/wildchat_prompts.json, calibration then evaluation) are extracted with the
release's own extractor and scored together with the release's cached test features by the configured-seed bank
and forest; the cached part must reproduce the saved configured-seed scores, so the new scores are on the same scale.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mtkaudit import release as rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rd.verify_copy()
    fp = rd.import_release()
    seed = rd.SHIPPED_SEED[a.model]
    out = rd.ROOT / "results/wildchat"
    wc = json.loads((out / "wildchat_prompts.json").read_text())
    texts = wc["calibration"] + wc["evaluation"]
    runner = rd.VicunaModel(fp, a.device) if a.model == "vicuna" else rd.ProtocolModel(fp, a.model, a.device)
    bank = torch.load(rd.CACHE / f"{a.model}_bank_seed{seed}.pt", weights_only=False)
    feats = runner.extract_queries(texts)
    if a.model == "vicuna":
        ts = torch.load(rd.CACHE / "vicuna_test_exact_shipped.pt", weights_only=False)
        test_feats, key = ts["features"], "exact_shipped"
    else:
        saved, _, _ = runner.test_features()
        test_feats, key = saved["features"], "release_cache"
    n_test = (test_feats["colon"] if isinstance(test_feats, dict) else test_feats).shape[0]
    queries = rd.cat_queries(test_feats, [feats])
    anomaly, _, _ = runner.anomaly(rd.to_gpu(bank["features"], a.device), bank["labels"], rd.to_gpu(queries, a.device), seed)
    saved_scores = np.load(rd.OUT / a.model / "scores" / f"as_shipped_{key}_seed{seed}.npy")
    max_diff = float(np.abs(anomaly[:n_test].astype(np.float32) - saved_scores[:n_test]).max())
    assert max_diff < 1e-5, f"rescored release test set differs from the saved scores ({max_diff})"
    np.save(out / f"{a.model}_MTK.npy", anomaly[n_test:].astype(np.float32))
    (out / f"{a.model}_MTK_check.json").write_text(json.dumps({"rescore_check_max_abs_diff": max_diff,
                                                                "n_prompts": len(texts)}, indent=1))
    print(a.model, "MTK WildChat scored", len(texts), "rescore check", max_diff, flush=True)


if __name__ == "__main__":
    main()
