"""Exact comparison of feature caches, tensors and nested containers (moved verbatim from
scripts/release/regenerate_driver_cache.py)."""
from __future__ import annotations

import torch

ROOTS: list[str] = []   # scratch roots; path strings are compared with the root replaced


def _norm(x: str) -> str:
    for r in ROOTS:
        x = x.replace(r, "<SCRATCH>")
    return x


def equal(a, b) -> bool:
    """Exact equality for tensors, recursively for dicts / lists / tuples of them; strings compared with
    the scratch root normalised (the release's manifests store absolute paths)."""
    if isinstance(a, str) and isinstance(b, str):
        return _norm(a) == _norm(b)
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b
