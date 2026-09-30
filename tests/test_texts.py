"""Third-party texts are not distributed, and the rebuild reproduces them exactly (THIRD_PARTY_NOTICES.md)."""
import copy
import json

from mtkaudit.paths import ROOT
from mtkaudit.texts import (check_against_index, find_texts, rebuild_saa850, saa850_index, saa_template,
                            strip_texts, text_sha256)


def test_no_third_party_texts_in_the_results():
    found = {}
    for p in sorted((ROOT / "results").rglob("*.json")):
        rel = str(p.relative_to(ROOT))
        if rel.endswith("_prompts.json"):        # rebuilt locally by rebuild_texts.py; ignored by git
            continue
        if f := find_texts(rel, json.loads(p.read_text())):
            found[rel] = f[:3]
    for p in sorted((ROOT / "results/release_pipeline/official_runs").rglob("*.csv")):
        if "prompt" in p.read_text(encoding="utf-8-sig").splitlines()[0].split(","):
            found[str(p.relative_to(ROOT))] = ["prompt column"]
    assert not found


def test_stripping_replaces_each_text_by_its_hash_and_is_idempotent():
    cases = {
        "results/release_pipeline/mechanism_v2/m.json":
            {"populations": {"saa": {"top_interleaving_anchors": [{"bank_position": 801, "text": "an anchor"}]}}},
        "results/rule_test/anchor_length_v2.json": {"m": {"saa": {"top5": [["an anchor", 9, 0.5]]}, "lengths": [9]}},
        "results/gradsafe/summary_refset0.json": {"reference": {"unsafe": ["u1", "u2"], "safe": ["s1", "s2"]}},
        "results/dolly_calibration/split_v2/m.json": {"dropped": [{"index": 3, "reason": "repeat", "text": "a dup"}]},
        "results/METHODS_MANIFEST.json": {"gradsafe": {"reference_prompts": {"unsafe": ["u"], "safe": ["s"]}},
                                          "windowed_perplexity": {"text": "raw prompt"}},
    }
    for rel, obj in cases.items():
        assert find_texts(rel, obj), rel
        once = strip_texts(rel, copy.deepcopy(obj))
        assert not find_texts(rel, once), rel
        assert strip_texts(rel, copy.deepcopy(once)) == once, rel
    anchors = strip_texts("results/release_pipeline/mechanism_v2/m.json", copy.deepcopy(
        cases["results/release_pipeline/mechanism_v2/m.json"]))["populations"]["saa"]["top_interleaving_anchors"]
    assert anchors == [{"bank_position": 801, "text80_sha256": text_sha256("an anchor")}]


class _Tokenizer:                     # stands in for a model tokenizer: one token per character
    vocab_size = 50

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": [ord(c) % self.vocab_size for c in text]})()

    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


def test_saa850_rebuild_matches_its_index():
    template = "Start: {TARGET}. Rules. Now {GOAL_LC} (end)"
    first = "Write a script that exploits a flaw"
    goals = [first, "Explain something harmful", "Describe another thing"]
    records = [{"goal": g, "target": f"Sure, target {i}",
                "jailbreak": template.replace("{TARGET}", f"Sure, target {i}").replace("{GOAL_LC}", "write a script "
                                                                                     "that exploits a flaw") + f" !!sfx{i}"}
               for i, g in enumerate(goals)]
    assert saa_template(records[0], first, text_sha256(template)) == template
    arms = rebuild_saa850(records, first, template, _Tokenizer())
    assert list(arms) == ["repaired", "nosuffix", "shufsuffix", "randsuffix"]
    assert arms["repaired"][1]["prompt"].endswith("now explain something harmful (end) !!sfx1".replace("now", "Now"))
    index = saa850_index(json.dumps(arms, indent=0).encode(), template)
    assert check_against_index(arms, index) == []
    arms["nosuffix"][2]["prompt"] += " "
    assert check_against_index(arms, index) == ["nosuffix[2]"]
