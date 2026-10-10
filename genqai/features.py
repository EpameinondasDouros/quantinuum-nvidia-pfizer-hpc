"""Conditioning features for the generative model (Workstream C).

Two featurizers, both fitted per system and serialisable into a checkpoint:

- ``hamiltonian``: canonical Pauli vocabulary = sorted union of Pauli labels over ALL geometries of
  the system (structure only, no coefficients/energies leak). A record becomes its coefficient
  vector on that vocabulary (absent terms = 0), standardised with mean/std from the TRAIN split.
- ``geometry``: the raw geometry parameters, standardised with TRAIN mean/std (ablation baseline).

Orbital-phase gauge (``hamiltonian`` kind, on by default). PySCF molecular orbitals have arbitrary
signs, so between neighbouring geometries the qubit Hamiltonian can jump to H' = U H U with
U = prod_p (-1)^{n_p} over some spatial orbitals p (a Pauli string after mapping). That flips the
sign of every Pauli term / pool generator that anticommutes with U, and the optimal angles flip
with them: the labels are discontinuous in geometry although the physics is smooth. We pick, per
record, the gauge g in {0,1}^n_orb whose transformed coefficient vector best aligns with a fixed
reference (fitted on TRAIN only) and present the model with gauge-fixed coefficients. Angles are
modelled in the gauge-fixed frame; ``op_signs(records)`` gives the per-operator signs s_k = ±1 with
theta_original = s_k * theta_canonical (energy is exactly invariant under this map).

Budget conditioning is a fixed 3-vector [has_budget, max_ops / MAX_OPS_NORM, cnot / CNOT_NORM].
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from functools import cached_property, lru_cache

import numpy as np
from qiskit.quantum_info import Pauli

from genqai.data import load_hamiltonians
from genqai.schema import HamiltonianRecord

STD_FLOOR = 1e-4  # Ha; terms that barely move across geometries are not blown up into noise
MAX_OPS_NORM = 32.0
CNOT_NORM = 500.0
BUDGET_DIM = 3


@lru_cache(maxsize=None)
def orbital_parity_paulis(system: str) -> tuple[str, ...]:
    """Qubit images of (-1)^(n_p,alpha + n_p,beta) for each active spatial orbital p."""
    from qiskit_nature.second_q.operators import FermionicOp

    from genqai.chemistry import SYSTEMS, mapper_for

    spec = SYSTEMS[system]
    n = spec.num_active_orbitals
    out = []
    for p in range(n):
        par = [FermionicOp({"": 1.0, f"+_{q} -_{q}": -2.0}, num_spin_orbitals=2 * n) for q in (p, p + n)]
        q = mapper_for(spec).map(par[0] @ par[1]).simplify()
        if len(q) != 1:
            raise RuntimeError(f"orbital parity of {system} orbital {p} is not a single Pauli: {q}")
        out.append(q.paulis.to_labels()[0])
    return tuple(out)


def _anticommute_mask(labels, gauge_paulis) -> np.ndarray:
    U = [Pauli(u) for u in gauge_paulis]
    return np.array([[not Pauli(l).commutes(u) for u in U] for l in labels], dtype=np.int64)


@dataclass
class Featurizer:
    kind: str  # "hamiltonian" | "geometry"
    system: str
    keys: list[str]  # Pauli labels or geometry parameter names
    mean: list[float]
    std: list[float]
    gauge_paulis: list[str] = field(default_factory=list)  # empty = no gauge fixing
    ref: list[float] = field(default_factory=list)  # reference raw coefficient vector (TRAIN)

    @property
    def dim(self) -> int:
        return len(self.keys)

    @cached_property
    def _gauges(self) -> tuple[np.ndarray, np.ndarray]:
        """(gauge bits [G, n_orb], sign patterns over keys [G, dim]) for every gauge (identity first)."""
        mask = _anticommute_mask(self.keys, self.gauge_paulis)
        gs = np.array(list(itertools.product([0, 1], repeat=len(self.gauge_paulis))), dtype=np.int64)
        return gs, 1.0 - 2.0 * ((mask @ gs.T) % 2).T

    @cached_property
    def _op_mask(self) -> np.ndarray:
        from genqai.pool import get_pool

        masks = [_anticommute_mask(g.paulis.to_labels(), self.gauge_paulis) for g in get_pool(self.system).generators]
        if any((m != m[0]).any() for m in masks):
            raise RuntimeError("a pool generator mixes gauge parities")
        return np.stack([m[0] for m in masks])

    def _raw(self, record: HamiltonianRecord) -> np.ndarray:
        if self.kind == "geometry":
            return np.array([record.geometry_params[k] for k in self.keys], float)
        coef = {t["pauli"]: t["coefficient"] for t in record.pauli_terms}
        return np.array([coef.get(k, 0.0) for k in self.keys], float)

    def _gauge_index(self, raw: np.ndarray) -> int:
        _, signs = self._gauges
        return int(np.argmax(signs @ (raw * np.array(self.ref))))

    def raw(self, record: HamiltonianRecord) -> np.ndarray:
        """Unstandardised (gauge-fixed, if enabled) feature vector."""
        x = self._raw(record)
        if not self.gauge_paulis:
            return x
        return x * self._gauges[1][self._gauge_index(x)]

    def op_signs(self, records: list[HamiltonianRecord]) -> np.ndarray:
        """[B, n_ops] signs s_k with theta_original = s_k * theta_canonical."""
        if not self.gauge_paulis:
            from genqai.pool import get_pool

            return np.ones((len(records), len(get_pool(self.system))), np.float32)
        gs, _ = self._gauges
        g = np.stack([gs[self._gauge_index(self._raw(r))] for r in records])
        return (1.0 - 2.0 * ((g @ self._op_mask.T) % 2)).astype(np.float32)

    def __call__(self, records: list[HamiltonianRecord]) -> np.ndarray:
        x = np.stack([self.raw(r) for r in records])
        return ((x - np.array(self.mean)) / np.array(self.std)).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_featurizer(system: str, kind: str = "hamiltonian", gauge: bool = True) -> Featurizer:
    recs = load_hamiltonians(system)
    train = [r for r in recs if r.split == "train"]
    if kind == "hamiltonian":
        keys = sorted({t["pauli"] for r in recs for t in r.pauli_terms})
    elif kind == "geometry":
        keys = list(recs[0].geometry_params)
    else:
        raise ValueError(kind)
    f = Featurizer(kind, system, keys, [0.0] * len(keys), [1.0] * len(keys))
    if kind == "hamiltonian" and gauge:
        # reference = TRAIN geometry closest to the TRAIN parameter centroid, then refined twice to
        # the mean of the gauge-aligned TRAIN vectors
        P = np.array([list(r.geometry_params.values()) for r in train])
        centre = train[int(np.argmin(np.linalg.norm((P - P.mean(0)) / (P.std(0) + 1e-12), axis=1)))]
        f.gauge_paulis = list(orbital_parity_paulis(system))
        f.ref = f._raw(centre).tolist()
        for _ in range(2):
            f.ref = np.mean([f.raw(r) for r in train], 0).tolist()
    x = np.stack([f.raw(r) for r in train])
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
