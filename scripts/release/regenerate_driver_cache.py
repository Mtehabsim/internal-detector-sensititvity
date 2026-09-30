"""Regenerate the driver's transient caches and compare them, tensor by tensor, with the originals.

    CUDA_VISIBLE_DEVICES=0 python scripts/release/regenerate_driver_cache.py --model vicuna [--save] [--reference DIR]

The caches (extras features and the configured seed's bank; for Vicuna also its test features, both the
runner-batched sample and all records) live on scratch space; results/archive/cache_SHA256SUMS records the
originals. This rebuilds every one of them with the release's own extraction code through release_driver
and, when a reference copy exists (default: the current MTK_RELEASE_SCRATCH/driver_cache; override with
--reference to compare a fresh scratch against the original), compares them for exact equality, walking
nested dictionaries (Vicuna's features are keyed by read position). With --save it writes every rebuilt
cache into MTK_RELEASE_SCRATCH/driver_cache. It never writes under results/ except its own report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mtkaudit import release as rd
from mtkaudit.caches import ROOTS, equal
from mtkaudit.cli import abs_path


def check_and_save(name, new, filename, ref_dir: Path, save: bool, report: dict):
    ref = ref_dir / filename
    report[name] = ("identical" if equal(new, torch.load(ref, weights_only=False)) else "DIFFERS") \
        if ref.exists() else "no original to compare"
    if save:
        torch.save(new, rd.CACHE / filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--reference", type=abs_path, default=None,
                    help="directory holding the original caches (default: this scratch's driver_cache)")
    a = ap.parse_args()
    identity = rd.verify_copy()
    fp = rd.import_release()
    ref_dir = a.reference or rd.CACHE
    ROOTS.extend(sorted({str(rd.SCRATCH), str(ref_dir.parent)}, key=len, reverse=True))
    rd.CACHE.mkdir(parents=True, exist_ok=True)
    seed = rd.SHIPPED_SEED[a.model]
    runner = rd.VicunaModel(fp, a.device) if a.model == "vicuna" else rd.ProtocolModel(fp, a.model, a.device)
    report = {"model": a.model, "release": identity["commit"], "reference": str(ref_dir)}

    extras = rd.extra_prompts(fp, a.model)
    feats = {name: runner.extract_queries([r["prompt"] for r in rows]) for name, rows in extras.items()}
    check_and_save("extras", feats, f"{a.model}_extras.pt", ref_dir, a.save, report)

    bank_feats, labels, sources, benign = runner.bank(seed)
    if a.model == "vicuna":
        malicious = runner.runner.get_train_dataset(
            [list(x) for x in rd.BENIGN_LIST], [list(x) for x in rd.MALICIOUS_LIST], seed)[1]
        bank = {"features": bank_feats, "labels": labels, "benign": benign, "malicious": malicious}
        # the cache written by release_mechanism.py holds the same four fields
        check_and_save("bank", bank, f"vicuna_bank_seed{seed}.pt", ref_dir, a.save, report)
        check_and_save("test_exact_shipped", runner.test_features(seed), "vicuna_test_exact_shipped.pt",
                       ref_dir, a.save, report)
        check_and_save("test_allrecords", runner.test_features(None), "vicuna_test_allrecords.pt",
                       ref_dir, a.save, report)
    else:
        bank = {"features": bank_feats, "labels": labels, "sources": sources}
        check_and_save("bank", bank, f"{a.model}_bank_seed{seed}.pt", ref_dir, a.save, report)

    out = rd.OUT / "cache_regeneration"
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.reference is None else "_fresh"
    (out / f"{a.model}{tag}.json").write_text(json.dumps(report, indent=1))
    print(report, flush=True)


if __name__ == "__main__":
    main()
