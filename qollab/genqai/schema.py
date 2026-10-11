"""Shared integration contracts. Every workstream reads and writes these records.

Conventions (do not change without updating every consumer):
- Pauli labels use Qiskit order: qubit 0 is the RIGHTMOST character.
- All energies are total energies in Hartree (qubit expectation + energy_shift).
- An operator is identified by its integer index into genqai.pool.OperatorPool for the system.
- A circuit is HF state followed by prod_t exp(-i * angle_t * G_{op_t}), applied in list order.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def software_versions() -> dict[str, str]:
    import numpy, pyscf, qiskit, qiskit_nature, scipy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pyscf": pyscf.__version__,
        "qiskit": qiskit.__version__,
        "qiskit_nature": qiskit_nature.__version__,
    }


@dataclass
class HamiltonianRecord:
    geometry_id: str
    system: str
    molecule: str
    geometry_params: dict[str, float]
    coordinates: list
    basis: str
    active_electrons: int
    active_orbitals: list[int]
    mapping: str
    num_qubits: int
    pauli_terms: list[dict[str, Any]]  # [{"pauli": "IZXY", "coefficient": float}], sorted by label
    energy_shift: float
    hf_state: str
    hf_energy: float
    ccsd_energy: float
    fci_energy: float
    software_versions: dict[str, str] = field(default_factory=dict)
    split: str = ""  # "train" | "val" | "test_interp" | "test_interval" | "test_extrap"
    charge: int = 0
    multiplicity: int = 1
    num_alpha: int = -1  # -1 = closed shell: active_electrons / 2 each (see nelec)
    num_beta: int = -1
    problem_key: str = ""  # genqai.molecule.ProblemConfig.key when built through the workbench
    reference_method: str = "FCI"  # "FCI" | "CCSD" | "HF". fci_energy is NaN unless "FCI" (sector too big
    # for exact diagonalisation); only an "FCI" reference may be called exact

    @property
    def nelec(self) -> tuple[int, int]:
        """Active (n_alpha, n_beta)."""
        if self.num_alpha < 0:
            return self.active_electrons // 2, self.active_electrons // 2
        return self.num_alpha, self.num_beta

    @property
    def reference_energy(self) -> float:
        """Best stored reference energy, named by reference_method (FCI, else CCSD, else HF)."""
        return {"FCI": self.fci_energy, "CCSD": self.ccsd_energy, "HF": self.hf_energy}[self.reference_method]

    def sparse_pauli_op(self):
        from qiskit.quantum_info import SparsePauliOp

        return SparsePauliOp.from_list([(t["pauli"], t["coefficient"]) for t in self.pauli_terms])


@dataclass
class GeneratedCircuit:
    """An ordered operator-angle sequence for one Hamiltonian."""

    geometry_id: str
    system: str
    operators: list[int]
    angles: list[float]
    valid: bool = True
    source: str = ""  # "adapt", "sft", "grpo", "nn_reuse", ...
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    method: str
    system: str
    geometry_id: str
    split: str
    seed: int
    energy: float  # total energy, Hartree
    error_mha: float  # 1000 * (energy - fci_energy)
    num_operators: int
    cnot_count: int | None = None
    depth: int | None = None
    pool_gradient_evals: int = 0
    energy_evals: int = 0
    optimizer_iterations: int = 0
    model_forward_passes: int = 0
    converged: bool | None = None
    runtime_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def dump_jsonl(records, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r) + "\n")


def load_jsonl(path: str | Path, cls=None) -> list:
    out = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(cls(**d) if cls else d)
    return out
