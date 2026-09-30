"""Compare a fresh run of the release harness with the archived run, record by record (CPU).

    python scripts/release/release_driver.py --model llama2 --seeds 0-49 --out rerun/llama2    # GPU: fresh run
    python scripts/release/compare_runs.py --fresh rerun                   # every model found under rerun/
    python scripts/release/compare_runs.py --fresh rerun --models llama2 --tol 1e-6 --metric-tol 0

`make gate`, `make tables` and `make verify` check the archive under results/ and never read a fresh run; this
script is the check for one. For every record (seed, variant, test features) in <fresh>/<model>/sweep.jsonl it
compares the per-prompt scores with the archived scores of the same record, and the record (bank sources,
seeds, metrics) with the archived text-disjoint record in <archived>/<model>/sweep_v2.jsonl, which is the
calibration the driver uses. The row layout (rows.json) must be identical. Records the archive lacks are listed,
not failed. Writes <fresh>/compare_report.json and exits nonzero unless every model passes.

A fresh run of the release's own runners is checked with the gate instead:
    python scripts/release/release_gate.py --driver rerun --official scratch/official_runs
"""
from __future__ import annotations

import argparse
import json
import sys

from mtkaudit.cli import abs_path
from mtkaudit.release import OUT
from mtkaudit.runs import compare_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", type=abs_path, required=True, help="directory holding <model>/ from --out runs")
    ap.add_argument("--archived", type=abs_path, default=OUT, help="default: results/release_pipeline")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tol", type=float, default=1e-6, help="largest allowed per-prompt score difference")
    ap.add_argument("--metric-tol", type=float, default=1e-9, help="largest allowed metric difference")
    a = ap.parse_args()
    models = a.models or sorted(p.name for p in a.fresh.iterdir() if (p / "sweep.jsonl").exists())
    if not models:
        raise SystemExit(f"no <model>/sweep.jsonl under {a.fresh}")
    report = {}
    for model in models:
        r = compare_model(a.fresh / model, a.archived / model, a.tol, a.metric_tol)
        report[model] = r
        print(f"{model}: {'PASS' if r['pass'] else 'FAIL'}  {len(r['records'])} records compared, "
              f"{len(r['missing_in_archive'])} not in the archive; rows identical {r['rows_identical']}; "
              f"max score diff {r['max_abs_score_diff']:.3g}, max metric diff {r['max_abs_metric_diff']:.3g}",
              flush=True)
    (a.fresh / "compare_report.json").write_text(json.dumps(report, indent=1))
    if not all(r["pass"] for r in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
