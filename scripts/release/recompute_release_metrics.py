"""Recompute every release-pipeline run's metrics from its saved per-prompt scores (CPU only).

    python scripts/release/recompute_release_metrics.py      # writes results/release_pipeline/<model>/sweep_v2.jsonl

The anomaly scores do not depend on calibration, so changing the calibration set needs no model. For each
record of <model>/sweep.jsonl this loads the saved score array and:

  1. recomputes the metrics under the original rule ("v1", position-disjoint calibration) and requires them
     to match the recorded metrics, which checks that this recomputation path is the one that produced them;
  2. computes them under the text-disjoint rule ("v2", rd.calibration_positions_text_disjoint) and writes
     the record, unchanged except for its metrics, to sweep_v2.jsonl.

sweep.jsonl is left as it is.
"""
from __future__ import annotations

import importlib
import json
import math

import numpy as np

from mtkaudit import release as rd
from mtkaudit.cli import run

TOL = 1e-9


def same(a, b, path=""):
    """Recursive comparison with a float tolerance; returns the first differing path or None."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            d = same(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            d = same(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(a, b, rel_tol=0, abs_tol=TOL):
            return f"{path}: {a} vs {b}"
        return None
    return None if a == b else f"{path}: {a} vs {b}"


def bank_pmp_texts(fp, model: str, seed: int) -> set[str]:
    """The bank's 200 OR-Bench non-refusal prompts for this seed, as the driver excluded them."""
    if model == "vicuna":
        runner = importlib.import_module("mtk_vicuna")
        benign, _ = runner.get_train_dataset([list(x) for x in rd.BENIGN_LIST],
                                             [list(x) for x in rd.MALICIOUS_LIST], seed)
    else:
        ext = importlib.import_module(f"extract_trainset_hiddenstates_{model}")
        proto = fp.FeatureProtocol(model, model, [], [(rd.COPY / p, n) for p, n in rd.BENIGN_LIST],
                                   [(rd.COPY / p, n) for p, n in rd.MALICIOUS_LIST], "x", "x",
                                   **({"training_line_reader": ext.read_mistral_training_lines}
                                      if model == "mistral" else {}))
        benign, _, _ = proto.sample_training(seed)
    return {rd.sha(p.strip()) for p in benign[600:800]}


def main():
    fp = rd.import_release()
    report = {}
    for model in ("llama2", "llama3", "mistral", "vicuna"):
        base = rd.OUT / model
        recs = [json.loads(l) for l in (base / "sweep.jsonl").read_text().splitlines()]
        layouts = json.loads((base / "rows.json").read_text())
        texts = rd.benign_texts(fp, model)
        judged = rd.load_judged(model)
        pmp_cache, out, mismatches = {}, [], []
        for r in recs:
            f = base / "scores" / f"{r['variant']}_{r['test_features']}_seed{r['seed']}.npy"
            scores = np.load(f)
            if r["seed"] not in pmp_cache:
                pmp_cache[r["seed"]] = bank_pmp_texts(fp, model, r["seed"])
            lay = layouts[r["test_features"]]
            m1 = rd.evaluate(fp, scores, lay, r["sample_seed"], pmp_cache[r["seed"]], judged, cal_rule="v1")
            stored = dict(r["metrics"])
            m1.pop("calibration_rule")
            diff = same(stored, m1)
            if diff:
                mismatches.append({"seed": r["seed"], "variant": r["variant"], "key": r["test_features"],
                                   "first_difference": diff})
            m2 = rd.evaluate(fp, scores, lay, r["sample_seed"], pmp_cache[r["seed"]], judged, texts)
            out.append({**r, "metrics": m2, "recomputed_from": str(f.relative_to(rd.ROOT)),
                        "v1_recompute_matches_record": diff is None})
        (base / "sweep_v2.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
        report[model] = {"records": len(recs), "v1_mismatches": len(mismatches), "examples": mismatches[:3],
                         "calibration_sizes": sorted({x["metrics"]["n_calibration"] for x in out})}
        print(model, report[model], flush=True)
    (rd.OUT / "recompute_v2_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    run(main, __doc__)
