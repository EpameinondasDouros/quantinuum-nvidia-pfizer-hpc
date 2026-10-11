"""Deterministic fermionic singles+doubles operator pool, mapped to qubits.

Operator k has a Hermitian qubit generator G_k (Qiskit Nature UCCSD convention), and the gate it
contributes is exp(-i * theta * G_k). Operator IDs depend only on the SystemSpec, never on
geometry, so the same ID means the same excitation for every Hamiltonian of a system.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import UCCSD

from genqai.chemistry import SYSTEMS, mapper_for


@dataclass(frozen=True)
class OperatorPool:
    system: str
    excitations: tuple  # Qiskit Nature excitation tuples, index == operator ID
    generators: tuple[SparsePauliOp, ...]  # Hermitian

    def __len__(self) -> int:
        return len(self.generators)

    def describe(self, k: int) -> str:
        occ, virt = self.excitations[k]
        kind = "single" if len(occ) == 1 else "double"
        return f"{kind} {list(occ)}->{list(virt)}"


@lru_cache(maxsize=None)
def get_pool(system: str) -> OperatorPool:
    spec = SYSTEMS[system]
    u = UCCSD(spec.num_active_orbitals, spec.nelec, mapper_for(spec))
    gens = []
    for op in u.operators:
        op = op.simplify(atol=1e-12)
        if np.abs(np.imag(op.coeffs)).max() > 1e-12:
            raise RuntimeError("expected real-coefficient (Hermitian) generators")
        gens.append(SparsePauliOp(op.paulis, np.real(op.coeffs)))
    return OperatorPool(system, tuple(u.excitation_list), tuple(gens))
