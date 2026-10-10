"""Workstream A: molecular systems, active spaces, qubit Hamiltonians and classical references.

Every qubit Hamiltonian is built from PySCF's own CASCI active-space integrals, so the
Hamiltonian, the HF/CCSD/FCI references and the orbital choice all come from one SCF.
All energies stored in a HamiltonianRecord are TOTAL energies in Hartree:
    E_total = <psi| H_qubit |psi> + energy_shift
where energy_shift = nuclear repulsion + frozen-core energy (PySCF CASCI ecore).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, cc, gto, mcscf, scf
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


def _select_active_orbitals(mol: gto.Mole, mf: scf.hf.RHF, spec: SystemSpec) -> list[int]:
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


def build_record(spec: SystemSpec, params: tuple[float, ...]) -> HamiltonianRecord:
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
    cas = mcscf.CASCI(mf, ncas, nelecas)
    cas.fcisolver.conv_tol = 1e-12
    mo = cas.sort_mo(act, base=0)
    cas.kernel(mo)
    e_fci = float(cas.e_tot)

    h1, ecore = cas.get_h1eff(mo)
    h2 = ao2mo.restore(1, cas.get_h2eff(mo), ncas)
    energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    qubit_op: SparsePauliOp = mapper_for(spec).map(energy.second_q_op()).simplify(atol=1e-12)
    qubit_op = qubit_op.sort()  # canonical lexicographic Pauli order

    frozen = [i for i in range(mf.mo_coeff.shape[1]) if i not in act]
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    e_ccsd = float(mycc.e_tot) if mycc.converged else float("nan")

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
