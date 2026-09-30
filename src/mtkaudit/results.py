"""Loaders for the archived results under results/: model and family names, sweep records, matched scores.
(Moved verbatim from scripts/paper/make_final_tables.py, which imports them.)
"""
from __future__ import annotations

import json

import numpy as np

from mtkaudit.paths import ROOT
RES = ROOT / "results"
RP = RES / "release_pipeline"
GEN = ROOT / "paper" / "latex" / "generated"
MODELS = ("llama2", "llama3", "mistral", "vicuna")
MNAME = {"llama2": "Llama-2", "llama3": "Llama-3", "mistral": "Mistral", "vicuna": "Vicuna"}
MMAC = {"llama2": "Ltwo", "llama3": "Lthree", "mistral": "Mis", "vicuna": "Vic"}
SHIPPED = {"llama2": 27, "llama3": 143, "mistral": 47, "vicuna": 56}
PRIMARY_KEY = {"llama2": "release_cache", "llama3": "release_cache", "mistral": "release_cache",
               "vicuna": "exact_shipped"}
FAM = {"autodan": "AutoDAN", "drattack": "DrAttack", "ijp": "IJP", "JailJudge": "JailJudge",
       "JailJudge_all": "JailJudge", "nanogcg": "nanoGCG", "nonagcg": "nanoGCG", "pair": "PAIR",
       "pap_gpt3.5": "PAP (GPT-3.5)", "pap_gpt4": "PAP (GPT-4)", "pap_llama2": "PAP (Llama-2)",
       "saa": "SAA", "tap": "TAP", "zulu": "Zulu"}
FAM_ORDER = ["SAA", "Zulu", "AutoDAN", "nanoGCG", "DrAttack", "IJP", "JailJudge", "PAIR", "TAP",
             "PAP (GPT-3.5)", "PAP (GPT-4)", "PAP (Llama-2)"]

def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def load_sweep(model):
    p = RP / model / "sweep_v2.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def by_seed(recs, model, variant):
    """One record per seed for a variant; the shipped seed's as_shipped row stands in for both."""
    ship = SHIPPED[model]
    out = {}
    for r in recs:
        if model == "vicuna":
            want = "allrecords" if variant == "as_shipped" else "exact_shipped"
            ok = (r["variant"] == variant and r["test_features"] == want) or \
                 (r["seed"] == ship and r["variant"] == "as_shipped" and r["test_features"] == "exact_shipped")
        else:
            ok = r["variant"] == variant or (r["seed"] == ship and r["variant"] == "as_shipped")
        if ok and r["seed"] not in out:
            out[r["seed"]] = r["metrics"]
    return out


def shipped_metrics(recs, model):
    for r in recs:
        if r["seed"] == SHIPPED[model] and r["variant"] == "as_shipped" and r["test_features"] == PRIMARY_KEY[model]:
            return r["metrics"]
    return None


# ------------------------------------------------------------------------------------------------ #
DETECTORS = (("MTK", "mtk.npy"), ("GradSafe", None), ("HiddenDetect", "hiddendetect.npy"),
             ("Linear probe", "probe.npy"), ("Windowed PPL", "ppl_windowed.npy"))
DCODE = {"MTK": "MTK", "GradSafe": "GS", "HiddenDetect": "HD", "Linear probe": "LP", "Windowed PPL": "PPL"}
INTERNAL = ("MTK", "GradSafe", "HiddenDetect", "Linear probe")



def matched_scores(model):
    """{detector: per-item scores} on the matched prompt set of one model, or {} if not scored."""
    out_dir = RES / "matched_detectors" / model
    if not (out_dir / "items.json").exists():
        return None, {}
    items = json.loads((out_dir / "items.json").read_text())
    dets = {}
    for name, f in DETECTORS:
        if f is not None and (out_dir / f).exists():
            dets[name] = np.load(out_dir / f).astype(float)
    gs = (RES / "gradsafe/scores_official.jsonl" if model == "llama2"
          else RES / f"gradsafe/port/scores_{model}.jsonl")
    if gs.exists():
        if model != "llama2":
            port_items = json.loads((RES / f"gradsafe/port/items_{model}.json").read_text()) \
                if (RES / f"gradsafe/port/items_{model}.json").exists() else None
            if port_items is None or [p["prompt_sha256"] for p in port_items] != [i["prompt_sha256"] for i in items]:
                gs = None
        if gs is not None:
            sc = {json.loads(l)["row"]: json.loads(l)["score"] for l in gs.read_text().splitlines()}
            if len(sc) == len(items):
                dets["GradSafe"] = np.array([sc[i["row"]] for i in items])
    return items, dets


def matched_scores_one(model, det):
    items, dets = matched_scores(model)
    return items, dets.get(det)


# ---- operating-point metrics per detector and model (moved from scripts/paper/make_final_tables.py, which uses
# them for Table VI; the analysis scripts pauc_vs_tpr.py and gradsafe_refset_v4.py use them too)
BUDGETS = (0.01, 0.02, 0.03, 0.04, 0.05)


def _auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def budget_metrics(model, items, s):
    rows = lambda p, f=None: [i["row"] for i in items if i["population"] == p and (f is None or i["family"] == f)]
    from mtkaudit import release as rd
    from mtkaudit.matched import calibration_rows_v2
    cal, ev = s[calibration_rows_v2(items)], s[rows("benign_eval")]
    thr = {t: float(np.quantile(cal, 1 - t)) for t in BUDGETS + (0.10,)}
    fams = sorted({i["family"] for i in items if i["population"] == "release_attack"})
    per = {}
    for f in fams:
        x = s[rows("release_attack", f)]
        per[f] = {"n": len(x), "auroc": _auroc(ev, x), "tpr": {t: float((x > thr[t]).mean()) for t in BUDGETS}}
    rep = [i for i in items if i["population"] == "saa850_repaired"]
    x850 = s[[i["row"] for i in rep]]
    judged = rd.load_judged(model)
    hb = np.array([judged["harmbench"][int(i["source_index"])] for i in rep], bool)
    return {"mean_auroc": float(np.mean([v["auroc"] for v in per.values()])),
            "mean_tpr": {t: float(np.mean([v["tpr"][t] for v in per.values()])) for t in BUDGETS},
            "saa_auroc": per["saa"]["auroc"], "saa_tpr": per["saa"]["tpr"],
            "saa_tpr10": float((s[rows("release_attack", "saa")] > thr[0.10]).mean()), "n_calibration": len(cal),
            "fpr": {t: float((ev > thr[t]).mean()) for t in BUDGETS},
            "saa850_easr": float((hb & ~(x850 > thr[0.05])).mean()),
            "saa850_easr_sr": float((np.array([judged["strongreject"][int(i["source_index"])] > 0.5 for i in rep], bool)
                                     & ~(x850 > thr[0.05])).mean()), "families": per}

