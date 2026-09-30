"""Text-disjoint Dolly split (rule "v2"; CPU, no rescoring).

    python scripts/calibration/dolly_split_v2.py        # writes results/dolly_calibration/split_v2/<model>.json

dolly_calibration.py drew 700 distinct ROWS of the release's Dolly file; the file repeats some texts, so the
first-200 calibration / last-500 evaluation split can hold a text twice or on both sides. v2 keeps the scored
prompts and drops, with texts compared after lower-casing and collapsing whitespace:
  * calibration (first 200 in draw order): any later copy of a text already kept, and any text that also occurs
    in the model's ToxicChat evaluated 500 or its text-disjoint ToxicChat calibration set;
  * evaluation (the other 500): the same, plus any text kept for calibration.
Indices refer to positions in the 700 saved scores (results/dolly_calibration/<model>_dolly_scores.npy and
other_detectors/<model>_<detector>.npy), so every detector uses the same prompts.
"""
from __future__ import annotations

import json

from mtkaudit import release as rd
from mtkaudit.cli import run
from mtkaudit.texts import norm, text_sha256



def main():
    fp = rd.import_release()
    out = rd.ROOT / "results/dolly_calibration/split_v2"
    out.mkdir(parents=True, exist_ok=True)
    for m, seed in rd.SHIPPED_SEED.items():
        rep = json.loads((rd.ROOT / "results/dolly_calibration" / f"{m}.json").read_text())
        reader = (__import__("extract_trainset_hiddenstates_mistral").read_mistral_training_lines
                  if m == "mistral" else fp.read_lines)
        rows = reader(rd.COPY / "datasets/train_data/databricks-dolly-15k.txt")
        texts = [norm(rows[i]) for i in rep["dolly_prompt_positions"]]
        tt = rd.benign_texts(fp, m)
        key = "exact_shipped" if m == "vicuna" else "release_cache"
        b = json.loads((rd.OUT / m / "rows.json").read_text())[key]["datasets"][rd.BENIGN_DATASET]
        ev = rd.sampled_positions(fp, b["n"], seed, rd.BENIGN_DATASET)
        tc = {norm(tt[p]) for p in ev + rd.calibration_positions_text_disjoint(fp, tt, ev, seed)}
        keep = {"calibration": [], "evaluation": []}
        dropped, seen = [], set()
        for i, t in enumerate(texts):
            part = "calibration" if i < 200 else "evaluation"
            reason = ("in ToxicChat" if t in tc else
                      "repeat" if t in seen else None)
            if reason:
                dropped.append({"index": i, "part": part, "reason": reason, "text80_sha256": text_sha256(t[:80])})
                continue
            seen.add(t)
            keep[part].append(i)
        rec = {"model": m, "rule": "v2: normalised-text dedup within and across the split; no ToxicChat texts",
               "calibration_idx": keep["calibration"], "evaluation_idx": keep["evaluation"],
               "n_calibration": len(keep["calibration"]), "n_evaluation": len(keep["evaluation"]),
               "dropped": dropped}
        (out / f"{m}.json").write_text(json.dumps(rec, indent=1))
        print(m, rec["n_calibration"], rec["n_evaluation"], [(d["part"], d["reason"], d["text80_sha256"][:12]) for d in dropped])


if __name__ == "__main__":
    run(main, __doc__)
