"""R3 mechanism on the authors' release (commit c5e2f18), at each model's shipped seed.

    python scripts/mechanism/release_mechanism.py --model llama2 --device cuda:0

The release's per-layer feature is the mean position of the k=10 nearest benign anchors in the list of
all 1,600 anchors sorted by distance. If m_j malicious anchors come before the j-th nearest benign
anchor, that position is j + m_j, so

    rank_l(x) = (k+1)/2 + (1/k) * sum_j m_j          (exact)

Two things are measured here, kept apart:

1. The identity, against the feature the release's rank_features actually returns. The loop below
   repeats the release's own loop (same distance expression, dtype, batch size 64 and argsort) and
   also records m_j; the rebuilt feature must equal the release's.
2. What the identity does not give: whether few interleaved malicious anchors actually yields a
   sub-threshold score. This is measured, not inferred: per population, the mean interleaving count,
   the share of query-layers where the k nearest benign anchors come before every malicious anchor,
   the ratio of nearest-malicious to nearest-benign distance, and the driver's shipped-seed score
   against the calibrated 5% threshold.

Populations: the release's evaluated benign prompts and each attack family at the shipped seed, our
calibration benign prompts, the 850-behaviour SAA set with and without its suffix, pseudo-malicious
prompts that are not bank members, and the bank's own benign anchors (leave-one-out, which is what the
forest is fitted on). Vicuna: both read positions are reported separately.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mtkaudit import release as rd
from mtkaudit.texts import text_sha256

K = rd.K
BATCH = 64


def interleaving(queries, background, labels, device, exclude_self=False):
    """Repeat the release's rank loop; return the feature, mean m_j, and nearest distances per class."""
    n_q, n_layers, _ = queries.shape
    background = background.to(device)
    labels = labels.to(device)
    positions = torch.arange(1, len(labels) + 1, device=device, dtype=torch.float32)
    rank = torch.empty((n_q, n_layers))
    mean_m = torch.empty((n_q, n_layers))
    all_benign_first = torch.empty((n_q, n_layers), dtype=torch.bool)
    d_ratio = torch.empty((n_q, n_layers))
    # per query, how many layers each anchor sits ahead of the k-th nearest benign anchor (malicious only)
    ahead = torch.zeros((n_q, len(labels)), dtype=torch.int16)
    for layer in range(n_layers):
        refs = background[:, layer, :]
        for start in range(0, n_q, BATCH):
            stop = min(start + BATCH, n_q)
            query = queries[start:stop, layer, :].to(device=device, dtype=background.dtype)
            distances = (refs.unsqueeze(0) - query.unsqueeze(1)).norm(p=2, dim=2)
            if exclude_self:
                local = torch.arange(stop - start, device=device)
                distances[local, torch.arange(start, stop, device=device)] = torch.inf
            order = distances.argsort(dim=1)
            benign_sorted = labels[order].eq(0)
            ranked = positions.expand(stop - start, -1).masked_fill(~benign_sorted, torch.inf)
            nearest = ranked.topk(K, largest=False, dim=1).values            # j + m_j, j = 1..k
            rank[start:stop, layer] = nearest.mean(dim=1).cpu()
            m = nearest - torch.arange(1, K + 1, device=device, dtype=torch.float32)
            mean_m[start:stop, layer] = m.mean(dim=1).cpu()
            all_benign_first[start:stop, layer] = (m[:, -1] == 0).cpu()
            before = positions.unsqueeze(0) < nearest[:, -1:]               # sorted slots ahead of k-th benign
            hit = torch.zeros((stop - start, len(labels)), dtype=torch.int16, device=device)
            hit.scatter_(1, order, (before & ~benign_sorted).to(torch.int16))
            ahead[start:stop] += hit.cpu()
            d = distances.float()
            d_ben = d.masked_fill(labels.eq(1).unsqueeze(0), torch.inf).min(dim=1).values
            d_mal = d.masked_fill(labels.eq(0).unsqueeze(0), torch.inf).min(dim=1).values
            d_ratio[start:stop, layer] = (d_mal / d_ben).cpu()
    return rank, mean_m, all_benign_first, d_ratio, ahead



def anchor_shares(ahead, rows, n_layers: int, n_benign: int = 800) -> np.ndarray:
    """Per malicious anchor, the share of the given queries' query-layers in which it precedes the k-th nearest
    benign anchor. ``ahead`` is interleaving()'s (n_queries, n_anchors) count of such layers, and the malicious
    anchors follow the ``n_benign`` benign ones. The denominator is queries x layers (not x anchors)."""
    rows = torch.as_tensor(rows)
    share = (ahead[rows].float().sum(dim=0) / (len(rows) * n_layers)).cpu().numpy()[n_benign:]
    assert 0.0 <= share.min() and share.max() <= 1.0, "a share lies outside [0, 1]"
    return share


def populations(fp, layout, seed, bank_pmp, model):
    ds = layout["datasets"]
    b = ds[rd.BENIGN_DATASET]
    b_eval = rd.sampled_positions(fp, b["n"], seed, rd.BENIGN_DATASET)
    b_cal = rd.calibration_positions_text_disjoint(fp, rd.benign_texts(fp, model), b_eval, seed)
    pops = {"benign_eval": rd.rows_of(b, b_eval), "benign_calibration": rd.rows_of(b, b_cal)}
    for name, info in ds.items():
        if name != rd.BENIGN_DATASET:
            pops[name] = rd.rows_of(info, rd.sampled_positions(fp, info["n"], seed, name))
    for arm in ("saa850_repaired", "saa850_nosuffix"):
        info = layout["extras"][arm]
        pops[arm] = list(range(info["start"], info["start"] + info["count"]))
    info = layout["extras"]["pmp"]
    pops["pmp_non_member"] = [info["start"] + i for i, t in enumerate(info["texts_stripped_sha"])
                              if t not in bank_pmp]
    return pops


def summarise(rank, mean_m, first, ratio, rows):
    r = torch.as_tensor(rows)
    traj = rank[r]
    modal = traj.round().mode(dim=0).values
    return {"n": len(rows),
            "share_on_modal_trajectory": float((traj.round() == modal).all(dim=1).float().mean()),
            "mean_between_prompt_sd_of_rank": float(traj.std(dim=0).mean()),
            "mean_interleaving": float(mean_m[r].mean()),
            "median_query_interleaving": float(mean_m[r].mean(dim=1).median()),
            "per_layer_interleaving": [round(float(x), 4) for x in mean_m[r].mean(dim=0)],
            "share_layers_benign_first": float(first[r].float().mean()),
            "median_dist_ratio_mal_over_ben": float(ratio[r].median()),
            "mean_rank_feature": float(rank[r].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    identity = rd.verify_copy()
    fp = rd.import_release()
    seed = rd.SHIPPED_SEED[a.model]
    out_dir = rd.OUT / "mechanism_v2"   # v2: threshold from the text-disjoint calibration set
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = rd.extra_prompts(fp, a.model)
    extra_feats = torch.load(rd.CACHE / f"{a.model}_extras.pt", weights_only=False)
    extra_list = [extra_feats[n] for n in extras]

    if a.model == "vicuna":
        key = "exact_shipped"
        ts = torch.load(rd.CACHE / "vicuna_test_exact_shipped.pt", weights_only=False)
        test_feats, test_ds = ts["features"], ts["datasets"]
        bank_path = rd.CACHE / f"vicuna_bank_seed{seed}.pt"
        if not bank_path.exists():
            runner = rd.VicunaModel(fp, a.device)
            benign, malicious = runner.runner.get_train_dataset(
                [list(x) for x in rd.BENIGN_LIST], [list(x) for x in rd.MALICIOUS_LIST], seed)
            feats, labels, _, benign_check = runner.bank(seed)
            assert benign_check == benign
            torch.save({"features": feats, "labels": labels, "benign": benign, "malicious": malicious},
                       bank_path)
            del runner
        bank = torch.load(bank_path, weights_only=False)
        bank_feats, labels = bank["features"], bank["labels"]
        benign_prompts = bank["benign"]
        malicious_prompts = bank["malicious"]
        release_rank = importlib_rank = __import__("JailbreakDetector_vicuna").rank_features
    else:
        key = "release_cache"
        saved = torch.load(rd.COPY / "canonical_assets" / a.model / "cache/test_features.pt",
                           map_location="cpu", weights_only=False)
        test_feats = saved["features"]
        test_ds = {name: {"n": info["count"],
                          "feature_row": list(range(info["start"], info["start"] + info["count"])),
                          "source_indices": info["source_indices"]}
                   for name, info in saved["datasets"].items()}
        bank = torch.load(rd.CACHE / f"{a.model}_bank_seed{seed}.pt", weights_only=False)
        bank_feats, labels = bank["features"], bank["labels"]
        # the bank's benign prompts, for PMP membership: the release's seeded draw
        proto_ext = __import__(f"extract_trainset_hiddenstates_{a.model}")
        proto = fp.FeatureProtocol(a.model, a.model, proto_ext.ATTACK_FILES,
                                   [(rd.COPY / p, n) for p, n in rd.BENIGN_LIST],
                                   [(rd.COPY / p, n) for p, n in rd.MALICIOUS_LIST], "x", "x",
                                   **({"training_line_reader": proto_ext.read_mistral_training_lines}
                                      if a.model == "mistral" else {}))
        benign_prompts, malicious_prompts, _ = proto.sample_training(seed)
    n_test = (test_feats["colon"] if isinstance(test_feats, dict) else test_feats).shape[0]
    layout = rd.build_layout(test_ds, extras, n_test)
    stored = json.loads((rd.OUT / a.model / "rows.json").read_text())[key]
    assert stored["datasets"].keys() == layout["datasets"].keys()
    assert all(stored["datasets"][n]["feature_row"] == layout["datasets"][n]["feature_row"]
               for n in layout["datasets"])
    queries = rd.cat_queries(test_feats, extra_list)
    bank_pmp = {rd.sha(p.strip()) for p in benign_prompts[600:800]}
    pops = populations(fp, layout, seed, bank_pmp, a.model)

    anomaly = np.load(rd.OUT / a.model / "scores" / f"as_shipped_{key}_seed{seed}.npy")
    thr = float(np.quantile(anomaly[pops["benign_calibration"]], 0.95))
    report = {"model": a.model, "seed": seed, "k": K, "release_commit": identity["commit"],
              "threshold_0.05": thr, "endpoints": {}}
    endpoints = ("colon", "ist") if a.model == "vicuna" else ("final",)
    for ep in endpoints:
        q = queries[ep] if a.model == "vicuna" else queries
        bf = bank_feats[ep] if a.model == "vicuna" else bank_feats
        rank, mean_m, first, ratio, ahead = interleaving(q, bf, labels, a.device)
        if a.model == "vicuna":
            release = release_rank(q, bf, labels, K, a.device, exclude_self=False, batch_size=64)
        else:
            release = fp.rank_features(q, bf, labels, [K], a.device)[K]
        rebuilt = (K + 1) / 2 + mean_m
        diff = (release - rebuilt).abs()
        t_rank, t_m, t_first, t_ratio, _ = interleaving(bf, bf, labels, a.device, exclude_self=True)
        ben_rows = torch.nonzero(labels == 0, as_tuple=True)[0].tolist()
        ep_rep = {"identity": {"max_abs_diff_release_vs_rebuilt": float(diff.max()),
                               "max_abs_diff_ours_vs_release": float((release - rank).abs().max()),
                               "n_entries": int(diff.numel())},
                  "bank_benign_anchors_loo": summarise(t_rank, t_m, t_first, t_ratio, ben_rows),
                  "populations": {}}
        for name, rows in pops.items():
            s = summarise(rank, mean_m, first, ratio, rows)
            if len(endpoints) == 1 or ep == "colon":
                sc = anomaly[rows]
                q25, q75 = np.percentile(sc - thr, [25, 75])
                s.update({"flagged_0.05": float((sc > thr).mean()),
                          "median_margin_to_threshold": float(np.median(sc - thr)),
                          "margin_iqr": [float(q25), float(q75)]})
            # which malicious anchors do the interleaving (share of this population's query-layers)
            share = ahead[torch.as_tensor(rows)].float().sum(dim=0) / (len(rows) * q.shape[1])
            top = torch.argsort(share, descending=True)[:10].tolist()
            s["top_interleaving_anchors"] = [
                {"bank_position": i, "share_of_query_layers": round(float(share[i]), 4),
                 "text80_sha256": text_sha256(malicious_prompts[i - 800][:80])} for i in top if share[i] > 0]
            ep_rep["populations"][name] = s
        report["endpoints"][ep] = ep_rep
        b = ep_rep["populations"]["benign_eval"]["mean_interleaving"]
        print(f"{a.model} [{ep}] identity max diff {ep_rep['identity']['max_abs_diff_release_vs_rebuilt']:.2e}; "
              f"bank benign LOO m {ep_rep['bank_benign_anchors_loo']['mean_interleaving']:.2f}", flush=True)
        for name, s in sorted(ep_rep["populations"].items(), key=lambda kv: kv[1]["mean_interleaving"]):
            print(f"  {name:22s} n={s['n']:4d} m={s['mean_interleaving']:8.2f} ({s['mean_interleaving'] / b:5.2f}x benign) "
                  f"benign-first {s['share_layers_benign_first']:.3f} d_mal/d_ben {s['median_dist_ratio_mal_over_ben']:.3f} "
                  f"modal {s['share_on_modal_trajectory']:.2f} "
                  + (f"flagged {s['flagged_0.05']:.3f} IQR {s['margin_iqr'][0]:+.3f}..{s['margin_iqr'][1]:+.3f}"
                     if "flagged_0.05" in s else ""), flush=True)
        for name in ("saa", "benign_eval"):
            print(f"  top anchors ahead for {name}:", [(t["bank_position"], t["share_of_query_layers"], t["text80_sha256"][:12])
                                                    for t in ep_rep["populations"][name]["top_interleaving_anchors"][:5]])
    report["driver_sha256"] = rd.file_sha(rd.Path(__file__))
    (out_dir / f"{a.model}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
