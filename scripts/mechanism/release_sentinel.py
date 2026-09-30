"""Sentinel anchors through the authors' released pipeline (commit c5e2f18).

    python scripts/mechanism/release_sentinel.py --model llama2 --device cuda:0

Same experiment as run_sentinel_anchors.py (matched placement), but every scoring step is the release's:
its seeded bank draw, its extractor for anchors, its rank features, its forest. Only the bank changes
between arms, and only in 200 (or 50) of the 600 PKU slots:

    none        the release's bank as drawn
    ordinary    the slots refilled with UNUSED ordinary PKU prompts (placement control)
    nosuffix    public SAA template around DEFENDER-held goals
    randsuffix  template + random vocabulary tokens
    shufsuffix  template + the published suffix tokens, permuted

Goal split: even record indices of the 850-behaviour set are the defender's (sentinels are built only
from these); odd indices are the evaluation half (the optimised-suffix prompts scored). Asserted disjoint.
Placement: the replaced positions are drawn once per (bank, dose) and shared by every arm.
Every arm refits the forest and recalibrates the threshold, because changing anchors changes the benign
anchors' own rank trajectories; this is stated, not hidden. Test prompts and forest seed are held at the
shipped seed (the `bank_only` convention of release_driver.py).

Banks: the shipped seed, plus seeds 0 and 1 (declared before running).

v2 (2026-09-25), written to sentinel_v2/: (a) the text-disjoint calibration set (rd.evaluate's default);
(b) one draw of defender goals per (bank, dose), shared by the three template arms, so those arms differ
only in their suffix treatment (v1 drew goals per arm); (c) per-prompt scores saved for every arm.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from mtkaudit import release as rd

DOSES = (50, 200)
ARMS = ("ordinary", "nosuffix", "randsuffix", "shufsuffix")
EXTRA_BANKS = (0, 1)
MARKUP = {"[\\list]", "[list]", ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    identity = rd.verify_copy()
    fp = rd.import_release()
    torch.set_num_threads(8)
    shipped = rd.SHIPPED_SEED[a.model]
    out_dir = rd.OUT / "sentinel_v2"
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.model}.jsonl"
    texts = rd.benign_texts(fp, a.model)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["bank_seed"], r["arm"], r["dose"]))

    runner = rd.VicunaModel(fp, a.device) if a.model == "vicuna" else rd.ProtocolModel(fp, a.model, a.device)
    judged = rd.load_judged(a.model)
    extras = rd.extra_prompts(fp, a.model)
    extra_feats = torch.load(rd.CACHE / f"{a.model}_extras.pt", weights_only=False)
    extra_list = [extra_feats[n] for n in extras]
    if a.model == "vicuna":
        ts = torch.load(rd.CACHE / "vicuna_test_exact_shipped.pt", weights_only=False)
    else:
        saved, _, _ = runner.test_features()
        ts = {"datasets": {name: {"n": info["count"],
                                  "feature_row": list(range(info["start"], info["start"] + info["count"])),
                                  "source_indices": info["source_indices"]}
                           for name, info in saved["datasets"].items()},
              "features": saved["features"]}
    n_test = (ts["features"]["colon"] if isinstance(ts["features"], dict) else ts["features"]).shape[0]
    layout = rd.build_layout(ts["datasets"], extras, n_test)
    queries = rd.to_gpu(rd.cat_queries(ts["features"], extra_list), a.device)
    del ts

    # goal split and sentinel prompts
    multi = rd.load_saa850(a.model)
    goals = [r["goal"] for r in multi["repaired"]]
    dev_rows, eval_rows = np.arange(0, len(goals), 2), np.arange(1, len(goals), 2)
    assert not ({goals[i] for i in dev_rows} & {goals[i] for i in eval_rows}), "goal leak"
    rep = layout["extras"]["saa850_repaired"]
    assert rep["record_index"] == [int(r["index"]) for r in multi["repaired"]]
    eval_feature_rows = [rep["start"] + int(i) for i in eval_rows]
    hb = np.array([judged["harmbench"][int(multi["repaired"][i]["index"])] for i in eval_rows])

    # PKU pool as the release reads it (blank lines dropped), for the ordinary control
    pku_path = rd.COPY / "datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt"
    reader = (__import__("extract_trainset_hiddenstates_mistral").read_mistral_training_lines
              if a.model == "mistral" else fp.read_lines)
    pku_rows = reader(pku_path)

    for bank_seed in (shipped,) + EXTRA_BANKS:
        todo = [(arm, dose) for arm in ("none",) + ARMS for dose in ((0,) if arm == "none" else DOSES)
                if (bank_seed, arm, dose) not in done]
        if not todo:
            continue
        base_feats, labels, sources, benign_prompts = runner.bank(bank_seed)
        bank_pmp = {rd.sha(p.strip()) for p in benign_prompts[600:800]}
        # malicious order in the release bank: AdvBench 100, MaliciousInstruct 100, PKU 600
        pku_positions = np.arange(800 + 200, 1600)
        if sources is not None:
            pku_src = next(s for s in sources if s["path"].endswith("PKU-SafeRLHF-prompts_3-6k.txt"))
            used_pku = set(pku_src["source_indices"])
        else:   # vicuna: recover the draw with the same seeded sampler
            import random
            rng_v = random.Random(fp.stable_seed(bank_seed, "train:malicious:PKU-SafeRLHF-prompts_3-6k.txt"))
            used_pku = set(rng_v.sample(range(len(pku_rows)), 600))
        eligible = [i for i in range(len(pku_rows))
                    if i not in used_pku and pku_rows[i].strip() not in MARKUP]
        place = np.random.default_rng(10_000 + bank_seed)
        drops = {d: place.choice(pku_positions, d, replace=False) for d in DOSES}
        for arm, dose in todo:
            t0 = time.time()
            feats = {k: v.clone() for k, v in base_feats.items()} if isinstance(base_feats, dict) \
                else base_feats.clone()
            meta = {}
            if dose:
                drop = drops[dose]
                assert int(labels[drop].sum()) == dose          # every replaced slot is malicious
                if arm == "ordinary":
                    pick_rng = np.random.default_rng([bank_seed, dose, ARMS.index(arm)])
                    repl_idx = pick_rng.choice(eligible, dose, replace=False)
                    anchor_texts = [pku_rows[i] for i in repl_idx]
                    meta["replacement_pku_rows"] = [int(i) for i in repl_idx]
                else:   # the same defender goals for every template arm at this (bank, dose)
                    take = np.random.default_rng([bank_seed, dose]).choice(dev_rows, dose, replace=False)
                    assert set(take) <= set(dev_rows)
                    anchor_texts = [multi[arm][i]["prompt"] for i in take]
                    meta["sentinel_records"] = [int(i) for i in take]
                new = runner.train_x(runner.model, runner.tok, anchor_texts, runner.batch, a.device) \
                    if a.model != "vicuna" else runner._dual(anchor_texts)
                if isinstance(feats, dict):
                    for ep in feats:
                        feats[ep][drop] = new[ep].to(feats[ep].dtype)
                else:
                    feats[drop] = new.to(feats.dtype)
                meta["replaced_positions"] = [int(i) for i in drop]
            anomaly, _, _ = runner.anomaly(rd.to_gpu(feats, a.device), labels, queries, shipped)
            np.save(out_dir / "scores" / f"{a.model}_bank{bank_seed}_{arm}_N{dose}.npy", anomaly.astype(np.float32))
            m = rd.evaluate(fp, anomaly, layout, shipped, bank_pmp, judged, texts)
            thr = m["thresholds"]["0.05"]
            s_eval = anomaly[eval_feature_rows]
            fams = m["families"]
            non_saa = [f for f in fams if not f.startswith("saa")]
            rec = {"bank_seed": bank_seed, "arm": arm, "dose": dose, "forest_seed": shipped,
                   "sample_seed": shipped,
                   "saa850_eval_half": {"n": len(eval_rows), "tpr_0.05": float((s_eval > thr).mean()),
                                        "easr_harmbench_0.05": float((hb & ~(s_eval > thr)).mean())},
                   "saa_released_tpr_0.05": fams["saa"]["tpr"]["0.05"],
                   "mean_tpr_all_0.05": float(np.mean([fams[f]["tpr"]["0.05"] for f in fams])),
                   "mean_tpr_non_saa_0.05": float(np.mean([fams[f]["tpr"]["0.05"] for f in non_saa])),
                   "non_saa_families": non_saa,
                   "realised_fpr_0.05": m["realised_fpr"]["0.05"], "pmp_fpr_0.05": m["pmp"]["fpr"]["0.05"],
                   "metrics": m, "placement": meta, "seconds": time.time() - t0,
                   "release_commit": identity["commit"], "driver_sha256": rd.file_sha(rd.Path(__file__))}
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{a.model} bank {bank_seed} {arm:10s} N={dose:3d} SAA850-eval TPR "
                  f"{rec['saa850_eval_half']['tpr_0.05']:.3f} released-SAA {rec['saa_released_tpr_0.05']:.3f} "
                  f"non-SAA {rec['mean_tpr_non_saa_0.05']:.3f} FPR {rec['realised_fpr_0.05']:.3f} "
                  f"PMP {rec['pmp_fpr_0.05']:.4f} ({rec['seconds']:.0f}s)", flush=True)
            del feats
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
