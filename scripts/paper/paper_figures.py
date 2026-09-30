"""Figures 2-4 at the size the submitted paper prints them (CPU; archived results only).

    python scripts/paper/paper_figures.py      # writes fig_ci_column.pdf, fig_budget_column.pdf, fig_lottery_v3.pdf
                                               # into paper/latex/generated/

Fig. 2 (fig_ci_column): 95% bootstrap intervals of detection at the real-user 5% threshold, realized FPR and AUROC for
the misses discussed in the text and the matching SAA-850 rows (results/uncertainty/bootstrap.json and saa850.json).
Fig. 3 (fig_budget_column): MTK per-family detection against realized benign FPR at each model's configured seed.
Fig. 4 (fig_lottery_v3): MTK across release seeds; SAA detection at 5% against mean AUROC over the twelve families
(top) and SAA AUROC against the number of formatting rows drawn into the malicious bank (bottom).
The generator (make_final_tables.py) draws full-width versions from the same records; these are the one-column and
printed-size layouts of the submitted paper (paper/submitted/). PDFs carry no creation date, so reruns are identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from mtkaudit import results as mt
from mtkaudit.cli import run

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_final_tables as gen  # noqa: E402  (markup_counts)

BLUE, VERM, INK, MUTED, GRID = "#0072B2", "#D55E00", "#222222", "#6b6b6b", "#e6e6e6"
META = {"CreationDate": None}


def fig_ci_column():
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": INK,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.0, "axes.labelcolor": INK}):
        bs = json.loads((mt.RES / "uncertainty/bootstrap.json").read_text())
        s8 = json.loads((mt.RES / "uncertainty/saa850.json").read_text())
        rows = []
        for det, m, f in (("MTK", "llama2", "saa"), ("MTK", "llama3", "saa"), ("MTK", "vicuna", "saa"),
                          ("MTK", "mistral", "autodan"), ("GradSafe", "llama2", "zulu"),
                          ("HiddenDetect", "vicuna", "saa")):
            c = bs["models"][m]["detectors"][det]
            rows.append({"label": f"{det}, {mt.MNAME[m]}, {mt.FAM[f]}", "auroc": c[f"auroc_{f}"],
                         "tpr": c[f"tpr_{f}_0.05"], "fpr": c["fpr_0.05"], "group": "released"})
        for det, m in (("MTK", "llama2"), ("MTK", "llama3"), ("MTK", "vicuna"), ("HiddenDetect", "vicuna")):
            c, f8 = bs["models"][m]["detectors"][det], s8["models"][m][det]
            rows.append({"label": f"{det}, {mt.MNAME[m]}, SAA-850", "auroc": f8["auroc"], "tpr": f8["tpr_0.05"],
                         "fpr": c["fpr_0.05"], "group": "saa850"})
        ys, y = [], 0.0
        for i, r in enumerate(rows):
            if i and r["group"] != rows[i - 1]["group"]:
                y += 0.8
            ys.append(y)
            y += 1.0
        ys = np.array(ys)

        def err(v):
            return [[v["point"] - v["lo"]], [v["hi"] - v["point"]]]

        abbr = {"GradSafe": "GS", "HiddenDetect": "HD"}
        labs = [f"{abbr.get(r['label'].split(', ', 1)[0], r['label'].split(', ', 1)[0])}, "
                f"{r['label'].split(', ', 1)[1]}" for r in rows]
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(3.45, 2.55), sharey=True,
                                         gridspec_kw={"width_ratios": [1.9, 1, 1], "wspace": 0.34})
        fig.subplots_adjust(left=0.3, right=0.98, bottom=0.2, top=0.92)
        for yy, r in zip(ys, rows):
            a1.errorbar(r["tpr"]["point"], yy, xerr=err(r["tpr"]), fmt="o", ms=3.6, color=VERM, elinewidth=1.4,
                        capsize=1.8)
            a2.errorbar(r["fpr"]["point"], yy, xerr=err(r["fpr"]), fmt="o", ms=3.2, color=INK, elinewidth=1.1,
                        capsize=1.8)
            a3.errorbar(r["auroc"]["point"], yy, xerr=err(r["auroc"]), fmt="o", ms=3.2, color=BLUE, elinewidth=1.1,
                        capsize=1.8)
        gap = (ys[5] + ys[6]) / 2
        a1.set_yticks(ys)
        a1.set_yticklabels(labs)
        a1.set_ylim(ys.max() + 0.7, -0.7)
        for ax in (a1, a2, a3):
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.axhline(gap, color=MUTED, lw=0.5, ls=(0, (2, 2)))
        for ax in (a2, a3):
            ax.tick_params(axis="y", length=0)
        a1.set_xlim(-0.02, 1.02); a1.set_xticks([0, 0.5, 1.0]); a1.set_xticklabels(["0", ".5", "1"])
        a2.set_xlim(0, 0.10); a2.set_xticks([0, 0.05, 0.10]); a2.set_xticklabels(["0", ".05", ".10"])
        a2.axvline(0.05, color=MUTED, lw=0.7, ls="--")
        a3.set_xlim(0.48, 1.02); a3.set_xticks([0.5, 0.75, 1.0]); a3.set_xticklabels([".5", ".75", "1"])
        a1.set_title("Detection at 5%", loc="left")
        a2.set_title("FPR", loc="left")
        a3.set_title("AUROC", loc="left")
        note = fig.text(0.5, 0.03, "95% bootstrap intervals; dashed: the 5% target", ha="center", va="bottom",
                        fontsize=6.3, color=INK)
        for _ in range(3):                      # fit the margins to the rendered labels: exactly one column wide
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            w = fig.get_figwidth() * fig.dpi
            x0 = min(tl.get_window_extent(rend).x0 for tl in a1.get_yticklabels()) / w
            x1 = max([ax.title.get_window_extent(rend).x1 for ax in (a1, a2, a3)]
                     + [a3.get_xticklabels()[-1].get_window_extent(rend).x1]) / w
            fig.subplots_adjust(left=fig.subplotpars.left + (0.008 - x0), right=fig.subplotpars.right - (x1 - 0.992))
        note.set_x((a1.get_position().x0 + a3.get_position().x1) / 2)
        fig.savefig(mt.GEN / "fig_ci_column.pdf", metadata=META)
        plt.close(fig)


def fig_budget_column():
    with plt.rc_context({"pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.5), sharex=True, sharey=True)
        for ax, m in zip(axes.flat, mt.MODELS):
            met = mt.shipped_metrics(mt.load_sweep(m), m)
            ts = sorted(met["realised_fpr"], key=float)
            x = [met["realised_fpr"][t] for t in ts]
            for f, v in met["families"].items():
                y = [v["tpr"][t] for t in ts]
                name = mt.FAM[f]
                if name in ("SAA", "Zulu", "AutoDAN"):
                    ax.plot(x, y, marker="o", ms=2.2, lw=1.3, label=name,
                            color={"SAA": "C3", "Zulu": "C0", "AutoDAN": "C2"}[name])
                else:
                    ax.plot(x, y, lw=0.5, color="0.75", zorder=0)
            ax.axvline(met["realised_fpr"]["0.05"], color="k", lw=0.6, ls="--")
            ax.set_title(f"{mt.MNAME[m]} (AUROC {met['mean_auroc']:.3f})", fontsize=7.5, pad=2)
            ax.tick_params(labelsize=6.5, pad=1.5)
            ax.set_xlim(0, 0.21)
            ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
            ax.set_xticklabels(["0", ".05", ".10", ".15", ".20"])
        for ax in axes[1]:
            ax.set_xlabel("realized benign FPR", fontsize=7, labelpad=1)
        for ax in axes[:, 0]:
            ax.set_ylabel("detection rate (TPR)", fontsize=7, labelpad=1)
        axes[0, 0].legend(fontsize=6.5, loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.2)
        fig.tight_layout(pad=0.25, h_pad=0.6, w_pad=0.5)
        fig.savefig(mt.GEN / "fig_budget_column.pdf", metadata=META)
        plt.close(fig)


def fig_lottery_v3():
    rng = np.random.default_rng(0)                      # horizontal jitter of the integer row counts
    with plt.rc_context({"pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 4, figsize=(5.73, 2.4))
        for j, m in enumerate(mt.MODELS):
            recs = mt.load_sweep(m)
            d = mt.by_seed(recs, m, "as_shipped")
            mk = gen.markup_counts(m, recs)
            seeds = sorted(d)
            ship = mt.SHIPPED[m]
            x = np.array([d[s]["mean_auroc"] for s in seeds])
            tpr = np.array([d[s]["families"]["saa"]["tpr"]["0.05"] for s in seeds])
            au = np.array([d[s]["families"]["saa"]["auroc"] for s in seeds])
            k = np.array([mk.get(s, 0) for s in seeds], float)
            top, bot = axes[0, j], axes[1, j]
            top.scatter(x, tpr, s=9, lw=0, alpha=0.75, color=BLUE)
            top.scatter([d[ship]["mean_auroc"]], [d[ship]["families"]["saa"]["tpr"]["0.05"]], marker="*", s=60,
                        facecolor="none", edgecolor="C3", lw=0.9, clip_on=False, zorder=3)
            top.set_ylim(-0.04, 1.04)
            top.set_title(mt.MNAME[m], fontsize=8, pad=2)
            top.xaxis.set_major_locator(MaxNLocator(3))
            top.margins(x=0.06)
            top.set_yticks([0, 0.5, 1])
            top.set_xlabel("mean AUROC", fontsize=7, labelpad=1)
            kj = k + rng.uniform(-0.12, 0.12, len(k))
            bot.scatter(kj, au, s=9, lw=0, alpha=0.75, color=BLUE)
            bot.scatter([mk.get(ship, 0)], [d[ship]["families"]["saa"]["auroc"]], marker="*", s=60,
                        facecolor="none", edgecolor="C3", lw=0.9, clip_on=False, zorder=3)
            bot.set_xlim(-0.5, 3.5)
            bot.set_xticks([0, 1, 2, 3])
            bot.set_ylim(0.2, 1.0)
            bot.set_yticks([0.2, 0.6, 1.0])
            bot.set_xlabel("formatting rows", fontsize=7, labelpad=1)
            for ax in (top, bot):
                ax.tick_params(labelsize=6.5, pad=1.5, length=2.5)
            if j == 0:
                top.set_ylabel("SAA TPR at 5%", fontsize=7)
                bot.set_ylabel("SAA AUROC", fontsize=7)
        fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.4)
        fig.savefig(mt.GEN / "fig_lottery_v3.pdf", metadata=META)
        plt.close(fig)


def main():
    mt.GEN.mkdir(parents=True, exist_ok=True)
    fig_ci_column()
    fig_budget_column()
    fig_lottery_v3()
    print("written", *(mt.GEN / f for f in ("fig_ci_column.pdf", "fig_budget_column.pdf", "fig_lottery_v3.pdf")),
          sep="\n  ")


if __name__ == "__main__":
    run(main, __doc__)
