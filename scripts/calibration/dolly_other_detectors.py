"""Dolly calibration for HiddenDetect, windowed perplexity and the supervised probe (GPU; one model per run).

    CUDA_VISIBLE_DEVICES=0 python scripts/calibration/dolly_other_detectors.py --model llama2
    CUDA_VISIBLE_DEVICES=0 python scripts/calibration/dolly_other_detectors.py --model llama2 --source wildchat   # the WildChat set

--source wildchat (added 2026-09-26; pre-registered design in the research record) scores the 1,500 selected
WildChat prompts instead, with the same code, into results/wildchat/other_detectors/.

Scores the same 700 held-out Dolly prompts as dolly_calibration.py (positions read from its report), with each
detector computed exactly as for the matched evaluation (results/matched_detectors):
  * HiddenDetect: Llama-2/3 from the release's own extractor (final rendered token, hidden_states[1:]); Mistral
    and Vicuna from the release chat template's final token, as matched_detectors.stage_score_extract;
  * windowed perplexity: baselines.prompt_ppls, window 16;
  * probe: refit exactly as probe_baseline.py (same layer, C, training data) and applied at MTK's read position.
A consistency check rescores the first 40 evaluated benign prompts of the matched set with the same code and
compares with the saved matched scores.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from mtkaudit import release as rd

LAYERS = list(range(16, 30))
# Largest allowed |rescored - saved| on the 40 matched benign prompts rescored as a consistency check (added
# 2026-09-26; every archived run is within them). HiddenDetect and perplexity recompute the same forward pass; the probe
# is refit, and logistic regression's solver lands on slightly different decision values (archived maximum 0.033).
TOLERANCE = {"HiddenDetect": 1e-3, "Windowed PPL": 1e-6, "Probe": 0.05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(rd.SHIPPED_SEED))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--source", choices=("dolly", "wildchat"), default="dolly")
    a = ap.parse_args()
    rd.verify_copy()
    fp = rd.import_release()
    import sys
    sys.path.insert(0, str(rd.ROOT))
    from mtkaudit.baselines import load_head, refusal_vector, hiddendetect_scores
    from mtkaudit.baselines import prompt_ppls
    seed = rd.SHIPPED_SEED[a.model]
    rep = json.loads((rd.ROOT / "results/dolly_calibration" / f"{a.model}.json").read_text())
    reader = (__import__("extract_trainset_hiddenstates_mistral").read_mistral_training_lines
              if a.model == "mistral" else fp.read_lines)
    rows = reader(rd.COPY / "datasets/train_data/databricks-dolly-15k.txt")
    texts = [rows[i] for i in rep["dolly_prompt_positions"]]
    if a.source == "wildchat":
        wc = json.loads((rd.ROOT / "results/wildchat/wildchat_prompts.json").read_text())
        texts = wc["calibration"] + wc["evaluation"]
    md_dir = rd.ROOT / "results/matched_detectors" / a.model
    items = json.loads((md_dir / "items.json").read_text())
    from mtkaudit import matched as md
    prompts_by_sha = {}
    for it in md.build_items(fp, a.model):
        prompts_by_sha[it["prompt_sha256"]] = it["prompt"]
    check = [i for i in items if i["population"] == "benign_eval"][:40]
    check_texts = [prompts_by_sha[i["prompt_sha256"]] for i in check]

    runner = rd.VicunaModel(fp, a.device) if a.model == "vicuna" else rd.ProtocolModel(fp, a.model, a.device)
    mtk_feats = runner.extract_queries(texts + check_texts)          # MTK's read position

    # probe, refit exactly as probe_baseline.py
    man = json.loads((md_dir / "probe_manifest.json").read_text())
    if a.model == "vicuna":
        bank = torch.load(rd.CACHE / f"vicuna_bank_seed{seed}.pt", weights_only=False)
        Xb, Q = bank["features"]["colon"].float().numpy(), mtk_feats["colon"].float().numpy()
    else:
        bank = torch.load(rd.CACHE / f"{a.model}_bank_seed{seed}.pt", weights_only=False)
        Xb, Q = bank["features"].float().numpy(), mtk_feats.float().numpy()
    X = Xb[:, man["chosen_bank_layer"], :]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=man["C"], max_iter=3000).fit((X - mu) / sd, bank["labels"].numpy())
    probe = clf.decision_function((Q[:, man["query_layer"], :] - mu) / sd)
    del runner
    torch.cuda.empty_cache()

    # HiddenDetect and perplexity, as in the matched evaluation
    model_obj, tok, norm, head, vocab = load_head(a.model)
    model_obj.eval()
    r = refusal_vector(tok, vocab)
    if a.model in ("llama2", "llama3"):
        hd = hiddendetect_scores(mtk_feats, norm, head, r, LAYERS, True, device=str(model_obj.device))
    else:
        tok.padding_side = "left"
        all_texts = texts + check_texts
        rendered = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                            add_generation_prompt=True) for t in all_texts]
        hd = np.empty(len(all_texts))
        order = np.argsort([len(x) for x in rendered])
        for s0 in range(0, len(order), 8):
            idx = order[s0:s0 + 8]
            enc = tok([rendered[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                hs = model_obj(**{k: v.to(model_obj.device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
            states = torch.stack([h[:, -1, :] for h in hs[1:]], dim=1)
            hd[idx] = hiddendetect_scores(states, norm, head, r, LAYERS, True, device=str(model_obj.device))
    ppl, _ = prompt_ppls(model_obj, tok, texts + check_texts, 16, model_obj.device)

    n = len(texts)
    saved = {"HiddenDetect": np.load(md_dir / "hiddendetect.npy"), "Windowed PPL": np.load(md_dir / "ppl_windowed.npy"),
             "Probe": np.load(md_dir / "probe.npy")}
    new = {"HiddenDetect": np.asarray(hd, float), "Windowed PPL": np.asarray(ppl, float), "Probe": np.asarray(probe, float)}
    rows_check = [i["row"] for i in check]
    consistency = {k: float(np.abs(new[k][n:] - saved[k][rows_check]).max()) for k in new}
    too_far = {k: v for k, v in consistency.items() if v > TOLERANCE[k]}
    if too_far:
        raise SystemExit(f"rescored matched prompts differ beyond tolerance {TOLERANCE}: {too_far}; nothing written")
    out = rd.ROOT / ("results/dolly_calibration" if a.source == "dolly" else "results/wildchat") / "other_detectors"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in new.items():
        np.save(out / f"{a.model}_{k.replace(' ', '_')}.npy", v[:n])
    (out / f"{a.model}_consistency.json").write_text(json.dumps(consistency, indent=1))
    print(a.model, "consistency (max |new - saved| on 40 matched prompts):", consistency, flush=True)


if __name__ == "__main__":
    main()
