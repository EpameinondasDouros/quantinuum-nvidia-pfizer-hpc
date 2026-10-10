"""Geometry grids, geometry-level splits, and the Hamiltonian dataset (Workstream A5).

Splits are assigned per unique geometry BEFORE any labels exist:
  test_extrap   - beyond the training range (largest bond lengths)
  test_interval - one contiguous held-out interval inside the range
  test_interp   - isolated points interleaved with training points
  val           - isolated points for checkpoint selection
  train         - everything else

A separate, never-tuned-on test set lives in data/<system>/hamiltonians_fresh.jsonl (split
"test_fresh", see fresh_geometries). It is only returned by load_hamiltonians(system, "test_fresh");
every other call (None, "test", ...) reads the original grid file unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from genqai.chemistry import SYSTEMS, build_record
from genqai.schema import GeneratedCircuit, HamiltonianRecord, dump_jsonl, load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def results_dir(fresh: bool = False) -> Path:
    """results/ for the grid splits; results_fresh/ for test_fresh, so old aggregates never mix with it."""
    return DATA_DIR.parent / ("results_fresh" if fresh else "results")


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


FRESH_SPLIT = "test_fresh"
FRESH_SEED = 20261010
FRESH_MIN_GRID_DIST = 0.01  # Angstrom; a fresh geometry must be at least this far from every grid point
# Rule fixed before any fresh result was computed. 1-D: n_in uniform r in [grid min, test_extrap
# threshold] (the range the train/val/test_interp/test_interval geometries span) + n_out uniform r in
# the extrapolation box beyond the grid edge. h4: n_in uniform (a, b) in a x [0.8, 1.85] + n_out with
# b beyond the grid edge 2.0.
FRESH_RULE = {
    "h2": {"n_in": 10, "in": [(0.5, 2.2)], "n_out": 4, "out": [(2.5, 2.9)]},
    "lih4": {"n_in": 10, "in": [(1.0, 3.6)], "n_out": 4, "out": [(4.0, 4.6)]},
    "lih6": {"n_in": 10, "in": [(1.0, 3.6)], "n_out": 4, "out": [(4.0, 4.6)]},
    "h4": {"n_in": 14, "in": [(0.8, 1.6), (0.8, 1.85)], "n_out": 6, "out": [(0.8, 1.6), (2.0, 2.4)]},
}


def fresh_geometries(system: str) -> list[tuple[tuple[float, ...], str]]:
    """[(params, "interp" | "extrap")] drawn with a fixed seed per FRESH_RULE, rounded to 1e-4 and
    resampled whenever a point lies within FRESH_MIN_GRID_DIST of a grid geometry or another fresh one."""
    rule = FRESH_RULE[system]
    rng = np.random.default_rng([FRESH_SEED, sum(map(ord, system))])
    taken = [np.array(g) for g in geometry_grid(system)]
    out = []
    for regime, n, box in (("interp", rule["n_in"], rule["in"]), ("extrap", rule["n_out"], rule["out"])):
        lo, hi = np.array(box).T
        while sum(r == regime for _, r in out) < n:
            p = np.round(rng.uniform(lo, hi), 4)
            if all(np.linalg.norm(p - q) >= FRESH_MIN_GRID_DIST for q in taken):
                taken.append(p)
                out.append((tuple(float(x) for x in p), regime))
    return out


def fresh_regime(system: str, params: dict[str, float]) -> str:
    """"extrap" if a fresh geometry lies in the FRESH_RULE extrapolation box, else "interp"."""
    lo, hi = np.array(FRESH_RULE[system]["out"]).T
    x = np.array(list(params.values()))
    return "extrap" if bool(np.all((x >= lo) & (x <= hi))) else "interp"


def build_dataset(system: str, fresh: bool = False) -> list[HamiltonianRecord]:
    if fresh:
        geoms = [(p, FRESH_SPLIT) for p, _ in fresh_geometries(system)]
    else:
        geoms = [(p, assign_split(system, p, i)) for i, p in enumerate(geometry_grid(system))]
    recs = []
    for p, split in geoms:
        r = build_record(SYSTEMS[system], p)
        r.split = split
        recs.append(r)
    return recs


def hamiltonians_path(system: str, fresh: bool = False) -> Path:
    return DATA_DIR / system / ("hamiltonians_fresh.jsonl" if fresh else "hamiltonians.jsonl")


def load_hamiltonians(system: str, split: str | None = None) -> list[HamiltonianRecord]:
    """Grid geometries (all, one split, or "test" = all test_*), or the fresh set for split "test_fresh"."""
    if split == FRESH_SPLIT:
        return load_jsonl(hamiltonians_path(system, fresh=True), HamiltonianRecord)
    recs = load_jsonl(hamiltonians_path(system), HamiltonianRecord)
    if split is None:
        return recs
    return [r for r in recs if r.split == split or (split == "test" and r.split.startswith("test"))]


def train_subset(records: list[HamiltonianRecord], n: int | None, seed: int = 0) -> list[HamiltonianRecord]:
    """n TRAIN records drawn uniformly without replacement (grid order kept). For a fixed seed the
    subsets are nested in n (prefixes of one permutation). n None or >= len(records): all of them."""
    if any(r.split != "train" for r in records):
        raise ValueError("train_subset draws from training-split geometries only")
    if n is None or n >= len(records):
        return list(records)
    keep = np.random.default_rng(seed).permutation(len(records))[:n]
    return [records[i] for i in sorted(keep)]


def load_teacher(system: str, variant: str = "teacher") -> dict[str, GeneratedCircuit]:
    """Teacher circuits keyed by geometry_id: "teacher" (raw ADAPT), "teacher_canonical", or
    "teacher_fresh" (fresh ADAPT on the test_fresh geometries; for scoring ADAPT only, never training).
    An absolute variant path (without .jsonl) reads labels stored outside data/, e.g. under runs/."""
    return {c.geometry_id: c for c in load_jsonl(DATA_DIR / system / f"{variant}.jsonl", GeneratedCircuit)}
