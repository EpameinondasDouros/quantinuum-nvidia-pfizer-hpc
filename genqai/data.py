"""Geometry grids, geometry-level splits, and the Hamiltonian dataset (Workstream A5).

Splits are assigned per unique geometry BEFORE any labels exist:
  test_extrap   - beyond the training range (largest bond lengths)
  test_interval - one contiguous held-out interval inside the range
  test_interp   - isolated points interleaved with training points
  val           - isolated points for checkpoint selection
  train         - everything else
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from genqai.chemistry import SYSTEMS, build_record
from genqai.schema import GeneratedCircuit, HamiltonianRecord, dump_jsonl, load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def geometry_grid(system: str) -> list[tuple[float, ...]]:
    if system == "h2":
        return [(r,) for r in np.round(np.linspace(0.5, 2.5, 21), 4)]
    if system == "lih4":
        return [(r,) for r in np.round(np.linspace(1.0, 4.0, 76), 4)]
    if system == "lih6":
        return [(r,) for r in np.round(np.linspace(1.0, 4.0, 46), 4)]
    if system == "h4":
        return [(a, b) for a in np.round(np.linspace(0.8, 1.6, 8), 4) for b in np.round(np.linspace(0.8, 2.0, 8), 4)]
    raise ValueError(system)


def assign_split(system: str, params: tuple[float, ...], idx: int) -> str:
    if system == "h4":
        a, b = params
        if b > 1.85:
            return "test_extrap"
        if 1.15 < a < 1.3:
            return "test_interval"
    else:
        (r,) = params
        hi = {"h2": 2.2, "lih4": 3.6, "lih6": 3.6}[system]
        lo, up = {"h2": (1.2, 1.45), "lih4": (2.3, 2.65), "lih6": (2.3, 2.65)}[system]
        if r > hi:
            return "test_extrap"
        if lo <= r <= up:
            return "test_interval"
    if idx % 6 == 3:
        return "test_interp"
    if idx % 6 == 5:
        return "val"
    return "train"


def build_dataset(system: str) -> list[HamiltonianRecord]:
    recs = []
    for i, p in enumerate(geometry_grid(system)):
        r = build_record(SYSTEMS[system], p)
        r.split = assign_split(system, p, i)
        recs.append(r)
    return recs


def hamiltonians_path(system: str) -> Path:
    return DATA_DIR / system / "hamiltonians.jsonl"


def load_hamiltonians(system: str, split: str | None = None) -> list[HamiltonianRecord]:
    recs = load_jsonl(hamiltonians_path(system), HamiltonianRecord)
    if split is None:
        return recs
    return [r for r in recs if r.split == split or (split == "test" and r.split.startswith("test"))]


def load_teacher(system: str) -> dict[str, GeneratedCircuit]:
    """ADAPT teacher circuits (data/<system>/teacher.jsonl) keyed by geometry_id."""
    return {c.geometry_id: c for c in load_jsonl(DATA_DIR / system / "teacher.jsonl", GeneratedCircuit)}
