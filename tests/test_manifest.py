"""Regenerating the methods manifest keeps the as-run hashes and hashes every distributed file."""
import json
import re

from mtkaudit.manifest import distributed_files, distributed_hashes, merge
from mtkaudit.paths import ROOT


def _tree(root):
    for rel in ("src/pkg/a.py", "src/pkg/t.jinja", "scripts/x/b.py", "scripts/s.sh", "tests/test_c.py",
                "Makefile", "pyproject.toml", "requirements.txt", "results/r.json", "src/pkg/__pycache__/a.pyc"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(rel)


def test_merge_keeps_the_as_run_hashes(tmp_path):
    _tree(tmp_path)
    old = {"scripts_sha256": {"flat_script.py": "0" * 64}, "distributed_scripts_note": "as run", "k": 1}
    new = {"k": 2, "scripts_sha256": {"src/pkg/a.py": "1" * 64}}
    out = merge(new, old, tmp_path)
    assert out["scripts_sha256"] == old["scripts_sha256"]
    assert out["distributed_scripts_note"] == "as run"
    assert out["k"] == 2
    assert set(out["distributed_scripts_sha256"]) == {
        "src/pkg/a.py", "src/pkg/t.jinja", "scripts/x/b.py", "scripts/s.sh", "tests/test_c.py",
        "Makefile", "pyproject.toml", "requirements.txt"}


def test_the_repository_distributes_its_setup_template_and_specs():
    names = {str(p.relative_to(ROOT)) for p in distributed_files(ROOT)}
    for required in ("scripts/setup_release.sh", "src/mtkaudit/chat_template_vicuna_v1_1.jinja", "Makefile",
                     "pyproject.toml", "requirements.txt", "src/mtkaudit/release.py"):
        assert required in names


def test_the_archived_manifest_describes_this_tree():
    man = json.loads((ROOT / "results/METHODS_MANIFEST.json").read_text())
    assert man["scripts_sha256"], "the as-run hashes are missing"
    stale = {k for k, v in distributed_hashes(ROOT).items() if man["distributed_scripts_sha256"].get(k) != v}
    assert not stale, ("results/METHODS_MANIFEST.json does not match these files (after changing code, rerun "
                       f"scripts/paper/write_methods_manifest.py): {sorted(stale)}")


def test_every_input_of_the_manifest_writer_is_distributed():
    script = (ROOT / "scripts/paper/write_methods_manifest.py").read_text()
    paths = set(re.findall(r'ROOT / "(results/[^"{}]+)"', script))
    assert paths, "no inputs found; has the script changed?"
    assert not [p for p in sorted(paths) if not (ROOT / p).exists()]
