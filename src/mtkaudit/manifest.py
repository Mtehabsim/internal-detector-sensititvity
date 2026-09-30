"""Hashes in results/METHODS_MANIFEST.json: the scripts as run for the paper, and the files distributed here.

The as-run hashes (``scripts_sha256``) describe the flat research scripts that produced the archived results; they
cannot be recomputed from this tree, so regenerating the manifest carries them over unchanged. The distributed
hashes cover every file that defines what this repository runs: code, templates, tests, build and dependency
specifications.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

DISTRIBUTED = ("src/**/*.py", "src/**/*.jinja", "scripts/**/*.py", "scripts/**/*.sh", "tests/**/*.py",
               "pyproject.toml", "requirements.txt", "Makefile")
AS_RUN_KEYS = ("scripts_sha256", "distributed_scripts_note")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distributed_files(root: Path) -> list[Path]:
    files = {p for pattern in DISTRIBUTED for p in root.glob(pattern)
             if p.is_file() and "__pycache__" not in p.parts}
    return sorted(files)


def distributed_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in distributed_files(root)}


def merge(new: dict, old: dict | None, root: Path) -> dict:
    """The manifest to write: the regenerated settings in ``new``, the as-run entries carried over from ``old``
    unchanged, and the distributed files' hashes recomputed from ``root``."""
    out = {k: v for k, v in new.items() if k not in AS_RUN_KEYS and k != "distributed_scripts_sha256"}
    for k in AS_RUN_KEYS:
        if old and k in old:
            out[k] = old[k]
    out["distributed_scripts_sha256"] = distributed_hashes(root)
    return out
