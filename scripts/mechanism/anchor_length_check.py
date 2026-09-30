"""Are the anchors that interleave for SAA unusually short? (GPU optional; cached features only)

    CUDA_VISIBLE_DEVICES=0 python scripts/mechanism/anchor_length_check.py      # writes results/rule_test/anchor_length_v2.json

Hypothesis under test (stated in the paper as an untested reading): MTK reads the hidden state at the chat
template's final token, and for a near-empty prompt and a heavily templated one that state is dominated by the
template, so short malicious anchors sit near SAA prompts and raise their interleaving count. Design, fixed before
running: for each model at its configured seed (Vicuna: the primary colon position), recompute with
release_mechanism.interleaving -- unchanged -- the share of each population's query-layers in which each of the
800 malicious anchors precedes the k-th nearest benign anchor. Anchor length is its character count after
stripping. Reported per model, for SAA (released and SAA-850) and, as a control, the evaluated benign prompts:
the Spearman correlation between anchor length and share across the 800 anchors, and the median length of the 20
highest-share anchors with its percentile among all 800. Support for the hypothesis: strongly negative correlation
and short top anchors for SAA, clearly more so than for benign prompts.

v2 (2026-09-26): a share is now divided by the number of query-layers, as defined above
(mtkaudit.mechanism.anchor_shares). v1 (anchor_length.json, kept) divided by the number of anchors (1,600), which
scaled every share of a model by the same constant, so its correlations, orderings and medians -- the numbers the
paper uses -- are unchanged, but its stored shares were too small. v2 also stores the full share arrays and
anchor lengths, so the verifier can recompute every reported number.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import spearmanr

from mtkaudit import release as rd
from mtkaudit.cli import run
from mtkaudit.mechanism import anchor_shares, interleaving, populations
from mtkaudit.texts import text_sha256


def main():
    rd.verify_copy()
    fp = rd.import_release()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = {}
    for model in ("llama2", "llama3", "mistral", "vicuna"):
        seed = rd.SHIPPED_SEED[model]
        extras = rd.extra_prompts(fp, model)
        extra_feats = torch.load(rd.CACHE / f"{model}_extras.pt", weights_only=False)
        if model == "vicuna":
            key = "exact_shipped"
            ts = torch.load(rd.CACHE / "vicuna_test_exact_shipped.pt", weights_only=False)
            test_feats, test_ds = ts["features"]["colon"], ts["datasets"]
            bank = torch.load(rd.CACHE / f"vicuna_bank_seed{seed}.pt", weights_only=False)
            bf, labels, malicious = bank["features"]["colon"], bank["labels"], bank["malicious"]
            extra_list = [extra_feats[n]["colon"] for n in extras]
        else:
            key = "release_cache"
            saved = torch.load(rd.COPY / "canonical_assets" / model / "cache/test_features.pt",
                               map_location="cpu", weights_only=False)
            test_feats = saved["features"]
            test_ds = {n: {"n": i["count"], "feature_row": list(range(i["start"], i["start"] + i["count"])),
                           "source_indices": i["source_indices"]} for n, i in saved["datasets"].items()}
            bank = torch.load(rd.CACHE / f"{model}_bank_seed{seed}.pt", weights_only=False)
            bf, labels = bank["features"], bank["labels"]
            ext = __import__(f"extract_trainset_hiddenstates_{model}")
            proto = fp.FeatureProtocol(model, model, ext.ATTACK_FILES,
                                       [(rd.COPY / p, n) for p, n in rd.BENIGN_LIST],
                                       [(rd.COPY / p, n) for p, n in rd.MALICIOUS_LIST], "x", "x",
                                       **({"training_line_reader": ext.read_mistral_training_lines}
                                          if model == "mistral" else {}))
            benign_prompts, malicious, _ = proto.sample_training(seed)
            extra_list = [extra_feats[n] for n in extras]
        n_test = test_feats.shape[0]
        layout = rd.build_layout(test_ds, extras, n_test)
        stored = json.loads((rd.OUT / model / "rows.json").read_text())[key]
        assert all(stored["datasets"][n]["feature_row"] == layout["datasets"][n]["feature_row"]
                   for n in layout["datasets"])
        queries = torch.cat([test_feats] + extra_list)
        pops = populations(fp, layout, seed, set(), model)
        # (n_queries, 1600) counts: in how many query layers each anchor precedes the k-th nearest benign anchor
        _, _, _, _, ahead = interleaving(queries, bf, labels, device)
        n_layers = queries.shape[1]
        lengths = np.array([len(t.strip()) for t in malicious])
        assert len(lengths) == 800
        res = {}
        for name in ("saa", "saa850_repaired", "benign_eval"):
            share = anchor_shares(ahead, pops[name], n_layers)
            top = np.argsort(-share)[:20]
            med_top = float(np.median(lengths[top]))
            res[name] = {"spearman_length_vs_share": float(spearmanr(lengths, share).correlation),
                         "median_len_top20": med_top,
                         "percentile_of_median_top20": float((lengths < med_top).mean() * 100),
                         "median_len_all": float(np.median(lengths)),
                         "top5": [{"text40_sha256": text_sha256(malicious[i].strip()[:40]), "length": int(lengths[i]),
                                   "share": round(float(share[i]), 3)} for i in top[:5]],
                         "share": [round(float(x), 6) for x in share]}
        res["lengths"] = [int(x) for x in lengths]
        res["n_query_layers"] = int(n_layers)
        out[model] = res
        print(model, {k: (round(v["spearman_length_vs_share"], 2), v["median_len_top20"], round(v["percentile_of_median_top20"], 1))
                      for k, v in res.items() if isinstance(v, dict)}, "| median length all", res["saa"]["median_len_all"], flush=True)
        print("   top SAA anchors:", res["saa"]["top5"], flush=True)
        del ahead
        torch.cuda.empty_cache()
    p = rd.ROOT / "results/rule_test"
    p.mkdir(parents=True, exist_ok=True)
    (p / "anchor_length_v2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
