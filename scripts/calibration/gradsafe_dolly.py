"""Dolly calibration for GradSafe (GPU; one model per run).

    CUDA_VISIBLE_DEVICES=0 python scripts/calibration/gradsafe_dolly.py --model llama2
    CUDA_VISIBLE_DEVICES=0 python scripts/calibration/gradsafe_dolly.py --model llama2 --source wildchat   # the WildChat set

--source wildchat (added 2026-09-26; pre-registered design in the research record) scores the 1,500 selected
WildChat prompts instead, with the same code, into results/wildchat/other_detectors/.

Scores the 700 held-out Dolly prompts of dolly_calibration.py with GradSafe exactly as in the matched
evaluation: the official reference and scorer on Llama-2 (gradsafe_driver), the chat-template port elsewhere
(gradsafe_port). The first 40 evaluated benign prompts of the matched set are rescored and compared with the
saved scores, to confirm the reference and scale are the same.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from mtkaudit import release as rd
from mtkaudit.gradsafe import official as gd
from mtkaudit.gradsafe import port as gp


TOLERANCE = 1e-4     # largest allowed |rescored - saved| on the 40 matched benign prompts (added 2026-09-26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--source", choices=("dolly", "wildchat"), default="dolly")
    a = ap.parse_args()
    fp = rd.import_release()
    from mtkaudit import matched as md
    rep = json.loads((rd.ROOT / "results/dolly_calibration" / f"{a.model}.json").read_text())
    reader = (__import__("extract_trainset_hiddenstates_mistral").read_mistral_training_lines
              if a.model == "mistral" else fp.read_lines)
    rows = reader(rd.COPY / "datasets/train_data/databricks-dolly-15k.txt")
    texts = [rows[i] for i in rep["dolly_prompt_positions"]]
    if a.source == "wildchat":
        wc = json.loads((rd.ROOT / "results/wildchat/wildchat_prompts.json").read_text())
        texts = wc["calibration"] + wc["evaluation"]
    items = json.loads((rd.ROOT / "results/matched_detectors" / a.model / "items.json").read_text())
    by_sha = {it["prompt_sha256"]: it["prompt"] for it in md.build_items(fp, a.model)}
    check = [i for i in items if i["population"] == "benign_eval"][:40]
    check_texts = [by_sha[i["prompt_sha256"]] for i in check]

    fcp = gd.import_official()
    if a.model == "llama2":
        ref, mrow, mcol = fcp.find_critical_para(gd.MODEL_PATH)
        fast = gd.FastScorer(ref, mrow, mcol)
        del ref
        torch.cuda.empty_cache()
        model, tok = fcp.load_model(gd.MODEL_PATH)
        score = lambda t: fast(model, tok, t)[0]
        saved = {json.loads(l)["row"]: json.loads(l)["score"]
                 for l in (rd.ROOT / "results/gradsafe/scores_official.jsonl").read_text().splitlines()}
    else:
        model, tok = fcp.load_model(gp.MODEL_PATHS[a.model])
        encode = gp.make_encoder(a.model, tok)
        ref, mrow, mcol = gp.critical_parameters(model, encode)
        fast = gd.FastScorer(ref, mrow, mcol)
        del ref
        torch.cuda.empty_cache()

        def score(text):
            gp.grads(model, encode, text)
            total = torch.zeros((), dtype=torch.float64, device="cuda")
            count = 0
            for name, param in model.named_parameters():
                if param.grad is None or name not in fast.slices:
                    continue
                r_, ref_rows, c_, ref_cols = fast.slices[name]
                g = param.grad
                if ref_rows is not None:
                    total += torch.nan_to_num(F.cosine_similarity(g[r_], ref_rows, dim=1)).double().sum()
                    count += len(r_)
                if ref_cols is not None:
                    total += torch.nan_to_num(F.cosine_similarity(g[:, c_], ref_cols, dim=0)).double().sum()
                    count += len(c_)
            return float(total / count)
        saved = {json.loads(l)["row"]: json.loads(l)["score"]
                 for l in (rd.ROOT / f"results/gradsafe/port/scores_{a.model}.jsonl").read_text().splitlines()}
    dolly = np.array([score(t) for t in texts])
    chk = np.array([score(t) for t in check_texts])
    diff = float(np.abs(chk - np.array([saved[i["row"]] for i in check])).max())
    if diff > TOLERANCE:        # same reference and scorer as the saved scores (archived maximum 5.0e-5)
        raise SystemExit(f"rescored matched prompts differ by {diff} > {TOLERANCE}; nothing written")
    out = rd.ROOT / ("results/dolly_calibration" if a.source == "dolly" else "results/wildchat") / "other_detectors"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{a.model}_GradSafe.npy", dolly)
    (out / f"{a.model}_gradsafe_consistency.json").write_text(json.dumps({"max_abs_diff_40": diff}))
    print(a.model, "GradSafe Dolly scored", len(dolly), "consistency", diff, flush=True)


if __name__ == "__main__":
    main()
