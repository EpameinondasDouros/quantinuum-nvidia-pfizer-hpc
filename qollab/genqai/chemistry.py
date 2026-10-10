"""Workstream A: molecular systems, active spaces, qubit Hamiltonians and classical references.

Every qubit Hamiltonian is built from frozen-core active-space integrals of one PySCF RHF, so the
Hamiltonian, the HF/CCSD/FCI references and the orbital choice all come from one SCF. Only
gto/scf/ao2mo are used for the Hamiltonian and FCI (exact sector diagonalisation), because those are
the PySCF parts that run in Qollab's browser runtime; tests check equality with PySCF CASCI.
All energies stored in a HamiltonianRecord are TOTAL energies in Hartree:
    E_total = <psi| H_qubit |psi> + energy_shift
where energy_shift = nuclear repulsion + frozen-core energy.
PySCF is imported only inside build_record, so stored records (data/<system>/hamiltonians.jsonl),
pools and circuits work in runtimes where PySCF is not installed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper

from genqai.schema import HamiltonianRecord, software_versions


@dataclass(frozen=True)
class SystemSpec:
    """A fixed active-space problem. One model is trained per SystemSpec."""

    name: str  # e.g. "lih4"
    molecule: str  # "H2", "LiH", "H4"
    geometry_params: tuple[str, ...]  # names of the geometry parameters
    active_electrons: int
    num_active_orbitals: int
    mapping: str  # "parity" (two-qubit reduced) or "jordan_wigner"
    basis: str = "sto-3g"

    def atoms(self, params: tuple[float, ...]) -> list[tuple[str, tuple[float, float, float]]]:
        if self.molecule == "H2":
            (r,) = params
            return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r))]
        if self.molecule == "LiH":
            (r,) = params
            return [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r))]
        if self.molecule == "H4":
            a, b = params  # linear chain, adjacent distances (a, b, a)
            return [("H", (0.0, 0.0, z)) for z in (0.0, a, a + b, 2 * a + b)]
        raise ValueError(self.molecule)


SYSTEMS: dict[str, SystemSpec] = {
    "h2": SystemSpec("h2", "H2", ("r",), 2, 2, "parity"),
    "lih4": SystemSpec("lih4", "LiH", ("r",), 2, 3, "parity"),
    "lih6": SystemSpec("lih6", "LiH", ("r",), 2, 4, "parity"),
    "h4": SystemSpec("h4", "H4", ("a", "b"), 4, 4, "jordan_wigner"),
}


def _select_active_orbitals(mol, mf, spec: SystemSpec) -> list[int]:
    """Deterministic active-orbital choice (0-based MO indices).

    Frozen core = lowest (n_elec - active_electrons)/2 orbitals. Active orbitals are the next
    orbitals by energy, restricted to sigma symmetry for linear molecules along z, except that
    lih6 adds exactly one pi_x orbital (identified by its px AO weight) so the choice never
    depends on how the degenerate pi pair happens to be rotated.
    """
    ncore = (mol.nelectron - spec.active_electrons) // 2
    ao_labels = mol.ao_labels()
    px = np.array(["px" in l for l in ao_labels])
    py = np.array(["py" in l for l in ao_labels])
    C = mf.mo_coeff
    w_px = (C[px, :] ** 2).sum(axis=0) if px.any() else np.zeros(C.shape[1])
    w_py = (C[py, :] ** 2).sum(axis=0) if py.any() else np.zeros(C.shape[1])
    sigma = [i for i in range(ncore, C.shape[1]) if w_px[i] < 1e-6 and w_py[i] < 1e-6]
    pix = [i for i in range(ncore, C.shape[1]) if w_px[i] > 0.5]
    if spec.name == "lih6":
        act = sigma[:3] + pix[:1]
    else:
        act = sigma[: spec.num_active_orbitals]
    if len(act) != spec.num_active_orbitals:
        raise RuntimeError(f"could not select {spec.num_active_orbitals} active orbitals: {act}")
    return sorted(act)


def mapper_for(spec: SystemSpec) -> QubitMapper:
    if spec.mapping == "parity":
        n = spec.active_electrons // 2
        return ParityMapper(num_particles=(n, n))
    return JordanWignerMapper()


def hf_bitstring(spec: SystemSpec) -> str:
    """HF reference as a Qiskit bitstring (qubit 0 is the RIGHTMOST character)."""
    from qiskit_nature.second_q.circuit.library import HartreeFock

    n = spec.active_electrons // 2
    hf = HartreeFock(spec.num_active_orbitals, (n, n), mapper_for(spec))
    bits = ["0"] * hf.num_qubits
    for inst in hf.data:
        assert inst.operation.name == "x"
        bits[hf.find_bit(inst.qubits[0]).index] = "1"
    return "".join(reversed(bits))


def geometry_id(spec: SystemSpec, params: tuple[float, ...]) -> str:
    key = json.dumps([spec.name, spec.basis, [round(p, 6) for p in params]])
    return f"{spec.name}-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _active_space_integrals(mol, mf, act: list[int], nelecas: int):
    """Frozen-core active-space (h1, h2 chemists' order, ecore) from RHF quantities only.

    Same result as PySCF's CASCI.get_h1eff / get_h2eff, but uses only gto/scf/ao2mo, which are
    the PySCF parts present in Qollab's WebAssembly build (no libmcscf / libfci / libcc).
    Core orbitals are the lowest non-active MOs, matching CASCI.sort_mo.
    """
    from pyscf import ao2mo

    C = mf.mo_coeff
    ncore = (mol.nelectron - nelecas) // 2
    core = [i for i in range(C.shape[1]) if i not in act][:ncore]
    hcore = mf.get_hcore()
    dm_core = 2.0 * C[:, core] @ C[:, core].T
    veff = mf.get_veff(mol, dm_core) if core else np.zeros_like(hcore)
    ecore = mol.energy_nuc() + np.einsum("ij,ji", dm_core, hcore) + 0.5 * np.einsum("ij,ji", dm_core, veff)
    c_act = C[:, act]
    h1 = c_act.T @ (hcore + veff) @ c_act
    eri = ao2mo.incore.full(mf._eri, c_act) if mf._eri is not None else ao2mo.kernel(mol, c_act)
    return h1, ao2mo.restore(1, eri, len(act)), float(ecore)


def _sector_ground_energy(energy: ElectronicEnergy, ncas: int, nelecas: int) -> float:
    """Exact (FCI) electronic energy in the (n_alpha, n_beta) sector, by dense diagonalisation.

    Jordan-Wigner basis index bit j = occupation of spin orbital j (alpha 0..ncas-1, beta after).
    """
    op = JordanWignerMapper().map(energy.second_q_op())
    n = nelecas // 2
    idx = [i for i in range(2 ** (2 * ncas))
           if bin(i & ((1 << ncas) - 1)).count("1") == n and bin(i >> ncas).count("1") == n]
    H = op.to_matrix(sparse=True)[idx][:, idx].toarray()
    return float(np.linalg.eigvalsh(H)[0])


def _ccsd_energy(mf, frozen: list[int]) -> float:
    """Frozen-core CCSD total energy; NaN where PySCF's CC module is unavailable (Qollab/Pyodide)."""
    try:
        from pyscf import cc
    except OSError:
        return float("nan")
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return float(mycc.e_tot) if mycc.converged else float("nan")


def build_record(spec: SystemSpec, params: tuple[float, ...]) -> HamiltonianRecord:
    from pyscf import gto, scf

    params = tuple(float(p) for p in params)
    atoms = spec.atoms(params)
    mol = gto.M(
        atom=[(s, xyz) for s, xyz in atoms], basis=spec.basis, unit="Angstrom", verbose=0
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF not converged for {spec.name} {params}")

    act = _select_active_orbitals(mol, mf, spec)
    ncas, nelecas = spec.num_active_orbitals, spec.active_electrons
    h1, h2, ecore = _active_space_integrals(mol, mf, act, nelecas)
    energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    e_fci = _sector_ground_energy(energy, ncas, nelecas) + ecore
    qubit_op: SparsePauliOp = mapper_for(spec).map(energy.second_q_op()).simplify(atol=1e-12)
    qubit_op = qubit_op.sort()  # canonical lexicographic Pauli order

    e_ccsd = _ccsd_energy(mf, [i for i in range(mf.mo_coeff.shape[1]) if i not in act])

    terms = [
        {"pauli": lbl, "coefficient": float(np.real(c))}
        for lbl, c in zip(qubit_op.paulis.to_labels(), qubit_op.coeffs)
    ]
    assert max(abs(np.imag(c)) for c in qubit_op.coeffs) < 1e-10

    return HamiltonianRecord(
        geometry_id=geometry_id(spec, params),
        system=spec.name,
        molecule=spec.molecule,
        geometry_params=dict(zip(spec.geometry_params, params)),
        coordinates=[[s, list(xyz)] for s, xyz in atoms],
        basis=spec.basis,
        active_electrons=nelecas,
        active_orbitals=act,
        mapping=spec.mapping,
        num_qubits=qubit_op.num_qubits,
        pauli_terms=terms,
        energy_shift=float(ecore),
        hf_state=hf_bitstring(spec),
        hf_energy=float(mf.e_tot),
        ccsd_energy=e_ccsd,
        fci_energy=e_fci,
        software_versions=software_versions(),
    )
