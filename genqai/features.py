"""Conditioning features for the generative model (Workstream C).

Two featurizers, both fitted per system and serialisable into a checkpoint:

- ``hamiltonian``: canonical Pauli vocabulary = sorted union of Pauli labels over ALL geometries of
  the system (structure only, no coefficients/energies leak). A record becomes its coefficient
  vector on that vocabulary (absent terms = 0), standardised with mean/std from the TRAIN split.
- ``geometry``: the raw geometry parameters, standardised with TRAIN mean/std (ablation baseline).

Budget conditioning is a fixed 3-vector [has_budget, max_ops / MAX_OPS_NORM, cnot / CNOT_NORM].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from genqai.data import load_hamiltonians
from genqai.schema import HamiltonianRecord

STD_FLOOR = 1e-4  # Ha; terms that barely move across geometries are not blown up into noise
MAX_OPS_NORM = 32.0
CNOT_NORM = 500.0
BUDGET_DIM = 3


@dataclass
class Featurizer:
    kind: str  # "hamiltonian" | "geometry"
    keys: list[str]  # Pauli labels or geometry parameter names
    mean: list[float]
    std: list[float]

    @property
    def dim(self) -> int:
        return len(self.keys)

    def raw(self, record: HamiltonianRecord) -> np.ndarray:
        if self.kind == "geometry":
            return np.array([record.geometry_params[k] for k in self.keys], float)
        coef = {t["pauli"]: t["coefficient"] for t in record.pauli_terms}
        return np.array([coef.get(k, 0.0) for k in self.keys], float)

    def __call__(self, records: list[HamiltonianRecord]) -> np.ndarray:
        x = np.stack([self.raw(r) for r in records])
        return ((x - np.array(self.mean)) / np.array(self.std)).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_featurizer(system: str, kind: str = "hamiltonian") -> Featurizer:
    recs = load_hamiltonians(system)
    if kind == "hamiltonian":
        keys = sorted({t["pauli"] for r in recs for t in r.pauli_terms})
    elif kind == "geometry":
        keys = list(recs[0].geometry_params)
    else:
        raise ValueError(kind)
    f = Featurizer(kind, keys, [0.0] * len(keys), [1.0] * len(keys))
    x = np.stack([f.raw(r) for r in recs if r.split == "train"])
    f.mean = x.mean(0).tolist()
    f.std = np.maximum(x.std(0), STD_FLOOR if kind == "hamiltonian" else 1e-8).tolist()
    return f


def budget_vector(max_ops: int | None = None, cnot_budget: int | None = None) -> np.ndarray:
    """[has_budget, max_ops/MAX_OPS_NORM, cnot/CNOT_NORM]; all zeros means unconstrained."""
    if max_ops is None and cnot_budget is None:
        return np.zeros(BUDGET_DIM, np.float32)
    return np.array(
        [1.0, (max_ops or 0) / MAX_OPS_NORM, (cnot_budget or 0) / CNOT_NORM], np.float32
    )
