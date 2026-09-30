"""Bootstrap helpers shared by the uncertainty analyses (moved verbatim from scripts/analysis/bootstrap_ci.py)."""
from __future__ import annotations

import json

import numpy as np

from mtkaudit.paths import ROOT

DETS = {"MTK": "mtk.npy", "HiddenDetect": "hiddendetect.npy", "Windowed PPL": "ppl_windowed.npy"}


def load(model):
    d = ROOT / "results/matched_detectors" / model
    items = json.loads((d / "items.json").read_text())
    dets = {k: np.load(d / f).astype(float) for k, f in DETS.items()}
    gsf = (ROOT / "results/gradsafe/scores_official.jsonl" if model == "llama2"
           else ROOT / f"results/gradsafe/port/scores_{model}.jsonl")
    g = {json.loads(l)["row"]: json.loads(l)["score"] for l in gsf.read_text().splitlines()}
    dets["GradSafe"] = np.array([g[i["row"]] for i in items])
    return items, dets


def auroc_rows(neg: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Row-wise Mann-Whitney AUROC for (B, n_neg) and (B, n_pos) arrays, ties counted one half."""
    out = np.empty(neg.shape[0])
    for b in range(neg.shape[0]):
        ns = np.sort(neg[b])
        lo = np.searchsorted(ns, pos[b], side="left")
        hi = np.searchsorted(ns, pos[b], side="right")
        out[b] = (lo + 0.5 * (hi - lo)).sum() / (len(ns) * len(pos[b]))
    return out


def summarise(point, reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}

