"""Supervised probe on the same data MTK uses (CPU; cached features only).

    python scripts/detectors/probe_baseline.py --model llama2

Question: is the operating-point gap specific to reference-based, unsupervised designs, or do internal signals
fail the same way when used with supervision? The probe gets exactly MTK's inputs: the configured seed's 1,600
labelled reference prompts (800 benign, 800 malicious) as training data, and hidden states at MTK's own read
position. Design, fixed before running:

  * per layer, standardise features on the training prompts and fit L2 logistic regression (C=1.0);
  * choose the layer by 5-fold stratified cross-validated AUROC on the 1,600 training prompts only;
  * refit on all 1,600 at that layer and score the matched evaluation prompts (the same items, calibration set
    and budgets as every other detector); the score is the decision function (log-odds of malicious).
Read positions follow the release. Vicuna uses its primary (colon) position. On Mistral the release's anchors
are read one block deeper than its queries (hidden_states[1:] vs [:-1]); the probe is applied to the query
features of the same transformer block it was trained on, so the offset does not handicap it.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mtkaudit import release as rd

C = 1.0


def cv_auroc(X, y, seed=0):
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=3000).fit((X[tr] - mu) / sd, y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function((X[te] - mu) / sd)))
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    a = ap.parse_args()
    fp = rd.import_release()
    from mtkaudit import matched as md
    seed = rd.SHIPPED_SEED[a.model]
    items = json.loads((rd.ROOT / "results/matched_detectors" / a.model / "items.json").read_text())
    if a.model == "vicuna":
        bank = torch.load(rd.CACHE / f"vicuna_bank_seed{seed}.pt", weights_only=False)
        Xb = bank["features"]["colon"].float().numpy()
        ts = torch.load(rd.CACHE / "vicuna_test_exact_shipped.pt", weights_only=False)
        test = ts["features"]["colon"]
        extras = torch.load(rd.CACHE / "vicuna_extras.pt", weights_only=False)
        extra_list = [extras[n]["colon"] for n in ("saa850_repaired", "saa850_nosuffix", "pmp")]
        key = "exact_shipped"
    else:
        bank = torch.load(rd.CACHE / f"{a.model}_bank_seed{seed}.pt", weights_only=False)
        Xb = bank["features"].float().numpy()
        test = torch.load(rd.COPY / "canonical_assets" / a.model / "cache/test_features.pt",
                          map_location="cpu", weights_only=False)["features"]
        extras = torch.load(rd.CACHE / f"{a.model}_extras.pt", weights_only=False)
        extra_list = [extras[n] for n in ("saa850_repaired", "saa850_nosuffix", "pmp")]
        key = "release_cache"
    y = bank["labels"].numpy()
    queries = torch.cat([test] + extra_list)
    layout = json.loads((rd.OUT / a.model / "rows.json").read_text())[key]
    rows = [md.feature_row(layout, i) for i in items]
    Q = queries[rows].float().numpy()
    offset = 1 if a.model == "mistral" else 0          # same block for anchors and queries (see docstring)
    layers = list(range(Xb.shape[1] - offset))
    cv = Parallel(n_jobs=16)(delayed(cv_auroc)(Xb[:, l, :], y) for l in layers)
    best = int(np.argmax(cv))
    X = Xb[:, best, :]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000).fit((X - mu) / sd, y)
    scores = clf.decision_function((Q[:, best + offset, :] - mu) / sd)
    out = rd.ROOT / "results/matched_detectors" / a.model
    np.save(out / "probe.npy", scores.astype(np.float64))
    (out / "probe_manifest.json").write_text(json.dumps({
        "training": "configured-seed bank: 800 benign + 800 malicious prompts (the release's seeded draw)",
        "read_position": "as the release (vicuna: colon)", "C": C, "cv": "5-fold stratified, bank only",
        "cv_auroc_by_layer": cv, "chosen_bank_layer": best, "query_layer": best + offset,
        "driver_sha256": rd.file_sha(rd.Path(__file__))}, indent=1))
    print(a.model, "layer", best, "CV AUROC", round(cv[best], 4), flush=True)


if __name__ == "__main__":
    main()
