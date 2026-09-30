"""Command-line helpers shared by the scripts."""
from __future__ import annotations

import argparse
from pathlib import Path


def run(main, doc: str) -> None:
    """Run ``main`` after parsing the command line, so ``--help`` prints the script's docstring instead of
    starting the computation, and a mistyped option is rejected rather than ignored."""
    argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()


def abs_path(value: str) -> Path:
    """argparse type for directory and file options: resolved against the directory the command was run from,
    at parse time, because mtkaudit.release.import_release() later changes the working directory."""
    return Path(value).expanduser().resolve()
