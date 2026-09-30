"""Per-family detection heatmap for the artifact (CPU; reads the saved per-family table only).

    python scripts/paper/per_family_heatmap.py      # writes results/matched_detectors/per_family_heatmap.{pdf,png}

Rows are detector-model pairs, columns the twelve attack families; each cell is TPR at the real-user 5% threshold.
A black outline marks a family of 50 or more prompts ranked well (AUROC >= 0.85) but detected below 0.2, the
criterion behind the paper's count of 11 detector-model pairs. Families under 50 prompts are marked with a dagger.
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from mtkaudit import results as mt
from mtkaudit.cli import run

SRC = mt.RES / "matched_detectors" / "budget_per_family.csv"
OUT = mt.RES / "matched_detectors" / "per_family_heatmap"
FAMS = ["SAA", "Zulu", "AutoDAN", "nanoGCG", "DrAttack", "IJP", "JailJudge", "PAIR", "TAP",
        "PAP (GPT-3.5)", "PAP (GPT-4)", "PAP (Llama-2)"]
DETS = ["MTK", "GradSafe", "HiddenDetect", "Linear probe", "Windowed PPL"]


def main():
    rows = list(csv.DictReader(SRC.open()))
    cell = {(r["model"], r["detector"].replace(" (port)", ""), r["family"]): r for r in rows}
    labels, tpr, au, n = [], [], [], []
    for m in mt.MODELS:
        for d in DETS:
            labels.append(f"{mt.MNAME[m]}  {d}")
            rs = [cell[(m, d, f)] for f in FAMS]
            tpr.append([float(r["tpr_5pct"]) for r in rs])
            au.append([float(r["auroc"]) for r in rs])
            n.append([int(r["n"]) for r in rs])
    tpr, au, n = map(np.array, (tpr, au, n))
    assert tpr.shape == (20, 12)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(tpr, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    hidden = 0
    for i in range(tpr.shape[0]):
        for j in range(tpr.shape[1]):
            ax.text(j, i, f"{tpr[i, j]:.2f}".lstrip("0") + ("†" if n[i, j] < 50 else ""), ha="center",
                    va="center", fontsize=6.5, color="white" if tpr[i, j] > 0.6 else "0.15")
            if n[i, j] >= 50 and au[i, j] >= 0.85 and tpr[i, j] < 0.2:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, lw=1.3, ec="k"))
                hidden += 1
    for k in range(1, len(mt.MODELS)):
        ax.axhline(k * len(DETS) - 0.5, color="white", lw=2.5)
    ax.set_xticks(range(len(FAMS)), FAMS, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Detection (TPR) at the real-user 5% threshold\noutlined: AUROC ≥ 0.85 and TPR < 0.2 "
                 "(n ≥ 50);  † fewer than 50 prompts", fontsize=8, loc="left")
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        p = OUT.with_suffix(f".{ext}")
        if p.exists():
            raise SystemExit(f"{p} exists; write a new version instead of overwriting")
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print("written", OUT.with_suffix(".pdf"), "| outlined cells:", hidden)


if __name__ == "__main__":
    run(main, __doc__)
