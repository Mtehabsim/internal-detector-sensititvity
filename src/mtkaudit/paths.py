"""Where everything lives. One place for the repository root and the environment variables.

    MTKAUDIT_ROOT         repository root (default: two levels above this file; use `pip install -e .`)
    MTK_RELEASE_CHECKOUT  git clone of github.com/Rookie143/mtk, its llm/ directory (default: third_party/mtk/llm)
    MTK_RELEASE_SCRATCH   working copy of the release and the feature caches (default: scratch/)
    MTK_MODEL_DIR         model checkpoints: llama2, llama3, mistral_7b, vicuna-7b-v1_5 (default: models/)
    GRADSAFE_DIR          git clone of github.com/xyq7/GradSafe (default: third_party/GradSafe)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MTKAUDIT_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results"
PAPER_GENERATED = ROOT / "paper" / "latex" / "generated"
RELEASE_CHECKOUT = Path(os.environ.get("MTK_RELEASE_CHECKOUT", ROOT / "third_party" / "mtk" / "llm"))
SCRATCH = Path(os.environ.get("MTK_RELEASE_SCRATCH", ROOT / "scratch"))
MODEL_DIR = Path(os.environ.get("MTK_MODEL_DIR", ROOT / "models"))
GRADSAFE_DIR = Path(os.environ.get("GRADSAFE_DIR", ROOT / "third_party" / "GradSafe"))


def display(path: Path) -> str:
    """A path as recorded in archived reports: relative to the repository when inside it, so that no local
    directory is written into results/."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
