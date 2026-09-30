"""Causal test of the markup-row mechanism, through the authors' released pipeline (commit c5e2f18).

    python scripts/mechanism/release_markup.py --model llama2 --device cuda:0

The PKU malicious pool the release samples 600 anchors from holds six formatting rows that are not
prompts ('[\\list]' x5, '[list]' x1). The mechanism analysis (release_mechanism.py) found that at
Llama-2's shipped seed the malicious anchors ranked ahead of SAA's benign neighbours are almost only
the three such rows that draw contains. This script changes only those rows and measures the result.

Design, fixed before running:
  banks   the shipped seed and seeds 0-9 (the release's own seeded draws)
  arms    none            the bank as drawn
          remove          every drawn markup row replaced by an unused ordinary PKU row (n > 0 only)
          insert3         (3 - n) randomly chosen ordinary PKU anchors replaced by undrawn markup rows,
                          so the bank holds three (n < 3 only)
          insert3_control the same positions replaced by unused ordinary PKU rows instead
          remove_control  n randomly chosen ordinary PKU anchors replaced by unused ordinary PKU rows
                          (n > 0 only). Added after the shipped bank's `remove` arm had run, as the
                          placement control for `remove`; it changes no other arm.
  saved   per-prompt scores for every arm run after 2026-09-24 21:40 in markup/scores/ (Llama-2's
          earlier arms recorded metrics only; its configured-bank arms are replayed with scores by
          `--replay`, which must reproduce the recorded metrics)
  held    test sample and forest at the shipped seed (release_driver's bank_only convention); the
          threshold is recalibrated per arm on our 1,000 benign calibration prompts.
Replacements are extracted with the release's own training extractor (runner.train_x / _dual).
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from mtkaudit import release as rd

MARKUP = {"[\\list]", "[list]"}
BANKS = tuple(range(10))
PKU = "datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt"
PKU_START = 1000                       # malicious order: AdvBench 100, MaliciousInstruct 100, PKU 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--replay", action="store_true",
                    help="re-run the configured bank's recorded arms, save their scores, assert same metrics")
    a = ap.parse_args()
    identity = rd.verify_copy()
    fp = rd.import_release()
    torch.set_num_threads(8)
    shipped = rd.SHIPPED_SEED[a.model]
    out_dir = rd.OUT / "markup_v2"   # v2: text-disjoint calibration; every arm's scores saved
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.model}.jsonl"
    done, recorded = set(), {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["bank_seed"], r["arm"]))
            recorded[(r["bank_seed"], r["arm"])] = r
    if a.replay:
        done = {k for k in done if k[0] != shipped}

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

    reader = (__import__("extract_trainset_hiddenstates_mistral").read_mistral_training_lines
              if a.model == "mistral" else fp.read_lines)
    pku_rows = reader(rd.COPY / PKU)
    markup_rows = [i for i, r in enumerate(pku_rows) if r.strip() in MARKUP]
    assert len(markup_rows) == 6, markup_rows

    def extract(texts):
        if a.model == "vicuna":
            return runner._dual(texts)
        return runner.train_x(runner.model, runner.tok, texts, runner.batch, a.device)

    for bank_seed in ((shipped,) if a.replay else (shipped,) + BANKS):
        feats0, labels, sources, benign_prompts = runner.bank(bank_seed)
        if sources is not None:
            used = next(s for s in sources if s["path"].endswith(PKU.split("/")[-1]))["source_indices"]
        else:   # vicuna's sampler, same seeded draw
            used = random.Random(fp.stable_seed(bank_seed, "train:malicious:" + PKU.split("/")[-1])) \
                .sample(range(len(pku_rows)), 600)
        mk_pos = [PKU_START + j for j, i in enumerate(used) if i in markup_rows]
        ordinary_pos = [PKU_START + j for j, i in enumerate(used) if i not in markup_rows]
        unused_ordinary = [i for i in range(len(pku_rows)) if i not in set(used) and i not in markup_rows]
        undrawn_markup = [i for i in markup_rows if i not in set(used)]
        n = len(mk_pos)
        bank_pmp = {rd.sha(p.strip()) for p in benign_prompts[600:800]}
        rng = np.random.default_rng([bank_seed, 7])
        arms = [("none", [], [])]
        if n > 0:
            arms.append(("remove", mk_pos, list(rng.choice(unused_ordinary, n, replace=False))))
        if n < 3:
            pos = list(rng.choice(ordinary_pos, 3 - n, replace=False))
            arms.append(("insert3", pos, undrawn_markup[:3 - n]))
            arms.append(("insert3_control", pos, list(rng.choice(unused_ordinary, 3 - n, replace=False))))
        if n > 0:   # drawn from its own generator so the arms above keep their draws
            rc = np.random.default_rng([bank_seed, 11])
            arms.append(("remove_control", list(rc.choice(ordinary_pos, n, replace=False)),
                         list(rc.choice(unused_ordinary, n, replace=False))))
        for arm, positions, rows in arms:
            if (bank_seed, arm) in done:
                continue
            t0 = time.time()
            feats = {k: v.clone() for k, v in feats0.items()} if isinstance(feats0, dict) else feats0.clone()
            if positions:
                new = extract([pku_rows[i] for i in rows])
                idx = torch.as_tensor([int(p) for p in positions])
                if isinstance(feats, dict):
                    for ep in feats:
                        feats[ep][idx] = new[ep].to(feats[ep].dtype)
                else:
                    feats[idx] = new.to(feats.dtype)
            anomaly, _, _ = runner.anomaly(rd.to_gpu(feats, a.device), labels, queries, shipped)
            (out_dir / "scores").mkdir(exist_ok=True)
            np.save(out_dir / "scores" / f"{a.model}_bank{bank_seed}_{arm}.npy", anomaly.astype(np.float32))
            m = rd.evaluate(fp, anomaly, layout, shipped, bank_pmp, judged, rd.benign_texts(fp, a.model))
            fams = m["families"]
            non_saa = [f for f in fams if f != "saa"]
            n_after = n - (n if arm == "remove" else 0) + (3 - n if arm == "insert3" else 0)
            rec = {"bank_seed": bank_seed, "arm": arm, "n_markup_drawn": n, "n_markup_after": n_after,
                   "positions": [int(p) for p in positions], "replacement_pku_rows": [int(i) for i in rows],
                   "forest_seed": shipped, "sample_seed": shipped,
                   "saa_auroc": fams["saa"]["auroc"], "saa_tpr_0.05": fams["saa"]["tpr"]["0.05"],
                   "saa_tpr_0.1": fams["saa"]["tpr"]["0.1"],
                   "saa850_auroc": m["saa850_repaired"]["auroc"], "saa850_tpr_0.05": m["saa850_repaired"]["tpr"]["0.05"],
                   "saa850_easr_harmbench_0.05": m["saa850_repaired"]["easr_harmbench_0.05"],
                   "mean_auroc": m["mean_auroc"],
                   "mean_auroc_non_saa": float(np.mean([fams[f]["auroc"] for f in non_saa])),
                   "mean_tpr_non_saa_0.05": float(np.mean([fams[f]["tpr"]["0.05"] for f in non_saa])),
                   "realised_fpr_0.05": m["realised_fpr"]["0.05"], "pmp_fpr_0.05": m["pmp"]["fpr"]["0.05"],
                   "metrics": m, "seconds": time.time() - t0, "release_commit": identity["commit"],
                   "driver_sha256": rd.file_sha(rd.Path(__file__))}
            if a.replay:
                old_rec = recorded[(bank_seed, arm)]
                assert old_rec["positions"] == rec["positions"] and old_rec["replacement_pku_rows"] == rec["replacement_pku_rows"]
                assert json.dumps(old_rec["metrics"], sort_keys=True) == json.dumps(m, sort_keys=True), \
                    f"replay of {bank_seed}/{arm} does not reproduce the recorded metrics"
                print(f"replayed {bank_seed}/{arm}: metrics identical", flush=True)
                continue
            with out_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{a.model} bank {bank_seed:3d} n={n} {arm:16s} -> {n_after}: SAA AUROC {rec['saa_auroc']:.3f} "
                  f"TPR@5 {rec['saa_tpr_0.05']:.3f} @10 {rec['saa_tpr_0.1']:.3f} | SAA850 {rec['saa850_tpr_0.05']:.3f} "
                  f"| mean AUROC {rec['mean_auroc']:.4f} non-SAA TPR {rec['mean_tpr_non_saa_0.05']:.3f} "
                  f"FPR {rec['realised_fpr_0.05']:.3f} ({rec['seconds']:.0f}s)", flush=True)
            del feats
            torch.cuda.empty_cache()
        del feats0


if __name__ == "__main__":
    main()
