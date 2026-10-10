"""Fast exact statevector evaluation for <= ~12 qubits (research backend).

Fermionic excitation generators satisfy G^3 = G, so
    exp(-i t G) = I + (cos t - 1) G^2 - i sin t G
which lets us apply each gate with two sparse mat-vecs. Gradients use the adjoint method.
This is an exact ideal-expectation evaluator; it is cross-checked against Qiskit's
Statevector in tests/test_core.py. It has no shot noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from genqai.pool import get_pool
from genqai.schema import HamiltonianRecord

_GEN_CACHE: dict[str, tuple[list, list]] = {}


def _pool_mats(system: str):
    if system not in _GEN_CACHE:
        pool = get_pool(system)
        G = [sp.csr_matrix(g.to_matrix(sparse=True)) for g in pool.generators]
        G2 = [(g @ g).tocsr() for g in G]
        for g, g2 in zip(G, G2):
            if abs(g2 @ g - g).max() > 1e-10:
                raise RuntimeError("generator does not satisfy G^3 = G")
        _GEN_CACHE[system] = (G, G2)
    return _GEN_CACHE[system]


@dataclass
class Evaluator:
    """Exact energies for one Hamiltonian. Counts every call for cost accounting."""

    record: HamiltonianRecord
    energy_evals: int = 0
    gradient_evals: int = 0  # full-parameter gradient calls
    pool_gradient_evals: int = 0  # number of individual pool-operator gradients computed
    H: sp.csr_matrix = field(init=False, repr=False)
    psi0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.H = sp.csr_matrix(self.record.sparse_pauli_op().to_matrix(sparse=True))
        dim = 2**self.record.num_qubits
        self.psi0 = np.zeros(dim, dtype=complex)
        self.psi0[int(self.record.hf_state, 2)] = 1.0
        self.G, self.G2 = _pool_mats(self.record.system)

    @property
    def shift(self) -> float:
        return self.record.energy_shift

    def _apply(self, k: int, t: float, psi: np.ndarray) -> np.ndarray:
        return psi + (np.cos(t) - 1.0) * (self.G2[k] @ psi) - 1j * np.sin(t) * (self.G[k] @ psi)

    def state(self, ops, angles) -> np.ndarray:
        psi = self.psi0
        for k, t in zip(ops, angles):
            psi = self._apply(k, t, psi)
        return psi

    def energy(self, ops, angles) -> float:
        self.energy_evals += 1
        psi = self.state(ops, angles)
        return float(np.real(np.vdot(psi, self.H @ psi))) + self.shift

    def energy_and_grad(self, ops, angles) -> tuple[float, np.ndarray]:
        self.energy_evals += 1
        self.gradient_evals += 1
        states = [self.psi0]
        for k, t in zip(ops, angles):
            states.append(self._apply(k, t, states[-1]))
        psi = states[-1]
        lam = self.H @ psi
        e = float(np.real(np.vdot(psi, lam)))
        grad = np.zeros(len(ops))
        for i in range(len(ops) - 1, -1, -1):
            k, t = ops[i], angles[i]
            grad[i] = 2.0 * np.imag(np.vdot(lam, self.G[k] @ states[i + 1]))
            lam = self._apply(k, -t, lam)  # U^dagger
        return e + self.shift, grad

    def pool_gradients(self, ops, angles) -> np.ndarray:
        """dE/dtheta for appending each pool operator at theta=0 (ADAPT selection)."""
        psi = self.state(ops, angles)
        hpsi = self.H @ psi
        self.pool_gradient_evals += len(self.G)
        return np.array([2.0 * np.imag(np.vdot(hpsi, g @ psi)) for g in self.G])

    def optimize(self, ops, init_angles=None, gtol: float = 1e-8, maxiter: int = 1000):
        """BFGS over all angles. Returns (angles, energy, scipy result)."""
        x0 = np.zeros(len(ops)) if init_angles is None else np.asarray(init_angles, float)
        if len(ops) == 0:
            return x0, self.energy(ops, x0), None
        res = minimize(
            lambda x: self.energy_and_grad(ops, x),
            x0,
            jac=True,
            method="BFGS",
            options={"gtol": gtol, "maxiter": maxiter},
        )
        return res.x, float(res.fun), res
