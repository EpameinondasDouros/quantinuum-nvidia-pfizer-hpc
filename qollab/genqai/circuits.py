"""Qiskit circuit construction and gate accounting for operator-angle sequences."""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

from genqai.pool import get_pool
from genqai.schema import HamiltonianRecord

BASIS = ["cx", "rz", "sx", "x"]


def build_circuit(record: HamiltonianRecord, ops, angles) -> QuantumCircuit:
    """HF state then prod_t exp(-i angle_t G_{op_t}).

    Each generator's Pauli terms commute (single/double excitations), so one Lie-Trotter step
    is exact. PauliEvolutionGate(G, time=t) implements exp(-i t G).
    """
    pool = get_pool(record.system)
    n = record.num_qubits
    qc = QuantumCircuit(n)
    for q, b in enumerate(reversed(record.hf_state)):
        if b == "1":
            qc.x(q)
    for k, t in zip(ops, angles):
        qc.append(PauliEvolutionGate(pool.generators[k], time=float(t), synthesis=LieTrotter()), range(n))
    return qc


def gate_stats(qc: QuantumCircuit, optimization_level: int = 1) -> dict:
    """Compiled counts in a generic CX basis (all-to-all connectivity, like trapped ions).

    These are Qiskit-basis CNOT counts, NOT IonQ-native MS/ZZ counts.
    """
    t = transpile(qc, basis_gates=BASIS, optimization_level=optimization_level, seed_transpiler=0)
    ops = t.count_ops()
    return {"cnot_count": int(ops.get("cx", 0)), "depth": int(t.depth()), "num_gates": int(sum(ops.values()))}
