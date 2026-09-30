"""Does the release's one-block offset on Mistral matter? (GPU; one run)

    CUDA_VISIBLE_DEVICES=0 python scripts/mechanism/mistral_offset.py     # writes results/release_pipeline/mistral_offset/

On Mistral the release reads bank anchors after transformer blocks 1-32 (hidden_states[1:], endpoint
"mistral_slash_token_transformer_layers_1_32") but queries from the embedding and blocks 1-31
(hidden_states[:-1], endpoint "..._embedding_plus_layers_1_31_project_compatible"), so layer l of a query is
compared with layer l+1 of the anchors. This re-extracts every test prompt (and SAA-850 with and without its
suffix) with the anchors' own slicing, [1:], at the same read token, and scores them with the configured bank
(seed 47, cached features) and forest, as shipped otherwise. Design fixed before running: same evaluation sample,
same text-disjoint calibration, metrics from release_driver.evaluate. Checks: (i) re-extracting the first 64 test
prompts with the release's own test slicing reproduces its cached features; (ii) scoring the cached features
reproduces the saved configured-seed scores.
"""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import torch

from mtkaudit import release as rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    identity = rd.verify_copy()
    fp = rd.import_release()
    m, seed = "mistral", rd.SHIPPED_SEED["mistral"]
    out = rd.ROOT / "results/release_pipeline/mistral_offset"
    out.mkdir(parents=True, exist_ok=True)
    ext = importlib.import_module("extract_trainset_hiddenstates_mistral")
    runner = rd.ProtocolModel(fp, m, a.device)
    saved, _, _ = runner.test_features()
    prompts = []
    for path in [runner.protocol.benign_test] + runner.protocol.attack_files:
        info = saved["datasets"][fp.method_name(path)]
        assert info["start"] == len(prompts)
        prompts.extend(r["prompt"] for r in fp.load_records(path))
        assert len(prompts) == info["start"] + info["count"]
    assert len(prompts) == saved["features"].shape[0]
    extras = rd.extra_prompts(fp, m)
    lay = json.loads((rd.OUT / m / "rows.json").read_text())["release_cache"]
    ex_prompts = []
    for arm in ("saa850_repaired", "saa850_nosuffix"):
        assert lay["extras"][arm]["start"] == len(prompts) + len(ex_prompts)
        ex_prompts.extend(r["prompt"] for r in extras[arm])

    # (i) ordering and read position: the release's test slicing reproduces its cache
    with rd.quiet():
        chk = ext.mistral_extract("test")(runner.model, runner.tok, prompts[:64], runner.batch, a.device)
    feat_diff = float((chk.float() - saved["features"][:64].float()).abs().max())

    bank = torch.load(rd.CACHE / f"{m}_bank_seed{seed}.pt", weights_only=False)
    labels = bank["labels"]
    bank_gpu = rd.to_gpu(bank["features"], a.device)
    ex_saved = torch.load(rd.CACHE / f"{m}_extras.pt", weights_only=False)
    # (ii) the cached features reproduce the saved configured-seed scores
    q_ship = torch.cat([saved["features"], ex_saved["saa850_repaired"], ex_saved["saa850_nosuffix"]])
    s_ship, _, _ = runner.anomaly(bank_gpu, labels, rd.to_gpu(q_ship, a.device), seed)
    ref = np.load(rd.OUT / m / "scores" / f"as_shipped_release_cache_seed{seed}.npy")
    score_diff = float(np.abs(s_ship[:len(q_ship)].astype(np.float32) - ref[:len(q_ship)]).max())

    with rd.quiet():
        aligned = ext.mistral_extract("training")(runner.model, runner.tok, prompts + ex_prompts, runner.batch, a.device)
    s_al, _, _ = runner.anomaly(bank_gpu, labels, rd.to_gpu(aligned, a.device), seed)
    np.save(out / "scores_aligned.npy", s_al.astype(np.float32))
    np.save(out / "scores_as_shipped.npy", s_ship.astype(np.float32))

    layout = {"datasets": lay["datasets"],
              "extras": {k: v for k, v in lay["extras"].items() if k in ("saa850_repaired", "saa850_nosuffix")}}
    texts, judged = rd.benign_texts(fp, m), rd.load_judged(m)
    res = {k: rd.evaluate(fp, s, layout, seed, set(), judged, texts) for k, s in (("as_shipped", s_ship), ("aligned", s_al))}
    rep = {"release_commit": identity["commit"], "seed": seed, "check_feature_max_abs_diff_64": feat_diff,
           "check_score_max_abs_diff": score_diff, "n_prompts": len(prompts) + len(ex_prompts), **res}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1))
    print(f"checks: features {feat_diff:.2e}, scores {score_diff:.2e}")
    for k, r in res.items():
        f = r["families"]
        print(f"{k:10s} mean AUROC {r['mean_auroc']:.3f} mean TPR@5 {r['mean_tpr_0.05']:.3f} FPR {r['realised_fpr']['0.05']:.3f} "
              f"SAA {f['saa']['auroc']:.3f}/{f['saa']['tpr']['0.05']:.3f} AutoDAN {f['autodan']['auroc']:.3f}/{f['autodan']['tpr']['0.05']:.3f} "
              f"worst {min((v['tpr']['0.05'], n) for n, v in f.items() if v['n'] >= 50)}")


if __name__ == "__main__":
    main()
