"""Workstream A: molecular systems, active spaces, qubit Hamiltonians and classical references.

Every qubit Hamiltonian is built from frozen-core active-space integrals of one PySCF RHF (ROHF for
open shells, with a doubly occupied core), so the Hamiltonian, the HF/CCSD/FCI references and the
orbital choice all come from one SCF. Presets live in SYSTEMS; custom molecules (genqai.molecule)
are added at run time with register_system, so pools, simulators and circuits work unchanged.
Only gto/scf/ao2mo are used for the Hamiltonian and FCI (exact sector diagonalisation), because those are
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
from math import comb

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper

from genqai.schema import HamiltonianRecord, software_versions


@dataclass(frozen=True)
class SystemSpec:
    """A fixed active-space problem. One model is trained per SystemSpec."""

    name: str  # e.g. "lih4"
    molecule: str  # "H2", "LiH", "H4", or a formula for custom molecules
    geometry_params: tuple[str, ...]  # names of the geometry parameters
    active_electrons: int
    num_active_orbitals: int
    mapping: str  # "parity" (two-qubit reduced) or "jordan_wigner"
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1  # 2S+1; PySCF spin = multiplicity - 1; > 1 uses ROHF
    fixed_atoms: tuple | None = None  # custom molecule: ((symbol, (x, y, z)), ...) Angstrom; no geometry params
    active_orbital_indices: tuple[int, ...] | None = None  # explicit 0-based MO indices
    frozen_core: bool = True  # chemical core kept out of the active space (enforced by genqai.molecule)

    @property
    def nelec(self) -> tuple[int, int]:
        """Active (n_alpha, n_beta). Every unpaired electron is active (the ROHF core is doubly occupied)."""
        n_alpha = (self.active_electrons + self.multiplicity - 1) // 2
        return n_alpha, self.active_electrons - n_alpha

    def atoms(self, params: tuple[float, ...]) -> list[tuple[str, tuple[float, float, float]]]:
        if self.fixed_atoms is not None:
            return [(sym, tuple(xyz)) for sym, xyz in self.fixed_atoms]
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
PRESETS = tuple(SYSTEMS)

# Exact sector diagonalisation is dense: above this many determinants the record's reference is CCSD
# (or HF where PySCF's CC module is missing) and fci_energy is NaN.
FCI_MAX_DIM = 2500


def register_system(spec: SystemSpec) -> SystemSpec:
    """Add a run-time SystemSpec (custom molecule) so get_pool / Evaluator / build_circuit see it."""
    old = SYSTEMS.get(spec.name)
    if old is not None and old != spec:
        raise ValueError(f"system {spec.name!r} is already registered with different settings")
    SYSTEMS[spec.name] = spec
    return spec


def sector_dim(ncas: int, nelec: tuple[int, int]) -> int:
    """Number of determinants in the (n_alpha, n_beta) sector of ncas spatial orbitals."""
    return comb(ncas, nelec[0]) * comb(ncas, nelec[1])


def _select_active_orbitals(mol, mf, spec: SystemSpec) -> list[int]:
    """Deterministic active-orbital choice (0-based MO indices).

    Explicit spec.active_orbital_indices win. Custom molecules (spec.fixed_atoms) take the orbitals
    around the Fermi level by energy: the highest n_beta doubly occupied, every singly occupied, and
    the lowest virtuals. Presets: frozen core = lowest (n_elec - active_electrons)/2 orbitals.
    Active orbitals are the next orbitals by energy, restricted to sigma symmetry for linear
    molecules along z, except that lih6 adds exactly one pi_x orbital (identified by its px AO
    weight) so the choice never depends on how the degenerate pi pair happens to be rotated.
    """
    if spec.active_orbital_indices is not None:
        act = sorted(spec.active_orbital_indices)
        if len(act) != spec.num_active_orbitals or len(set(act)) != len(act) or not 0 <= act[0] <= act[-1] < mf.mo_coeff.shape[1]:
            raise ValueError(f"invalid active orbital indices {act} for {spec.num_active_orbitals} orbitals")
        return act
    if spec.fixed_atoms is not None:
        n_alpha, n_beta = spec.nelec
        order = np.argsort(mf.mo_energy, kind="stable")
        by_occ = {k: [int(i) for i in order if round(mf.mo_occ[i]) == k] for k in (2, 1, 0)}
        n_virt = spec.num_active_orbitals - n_alpha
        if len(by_occ[1]) != n_alpha - n_beta or n_beta > len(by_occ[2]) or not 0 <= n_virt <= len(by_occ[0]):
            raise ValueError(
                f"cannot place {spec.active_electrons} electrons in {spec.num_active_orbitals} orbitals around the Fermi "
                f"level ({len(by_occ[2])} doubly, {len(by_occ[1])} singly occupied, {len(by_occ[0])} virtual orbitals)")
        return sorted(by_occ[2][len(by_occ[2]) - n_beta:] + by_occ[1] + by_occ[0][:n_virt])
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
        return ParityMapper(num_particles=spec.nelec)
    return JordanWignerMapper()


def hf_bitstring(spec: SystemSpec) -> str:
    """HF reference as a Qiskit bitstring (qubit 0 is the RIGHTMOST character)."""
    from qiskit_nature.second_q.circuit.library import HartreeFock

    hf = HartreeFock(spec.num_active_orbitals, spec.nelec, mapper_for(spec))
    bits = ["0"] * hf.num_qubits
    for inst in hf.data:
        assert inst.operation.name == "x"
        bits[hf.find_bit(inst.qubits[0]).index] = "1"
    return "".join(reversed(bits))


def geometry_id(spec: SystemSpec, params: tuple[float, ...]) -> str:
    key = json.dumps([spec.name, spec.basis, [round(p, 6) for p in params]])
    return f"{spec.name}-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _active_space_integrals(mol, mf, act: list[int], nelecas: int):
    """Frozen-core active-space (h1, h2 chemists' order, ecore) from RHF/ROHF quantities only.

    Same result as PySCF's CASCI.get_h1eff / get_h2eff, but uses only gto/scf/ao2mo, which are
    the PySCF parts present in Qollab's WebAssembly build (no libmcscf / libfci / libcc).
    Core orbitals are the lowest non-active MOs, matching CASCI.sort_mo. They are doubly occupied
    (also for ROHF), so the closed-shell core potential J - K/2 applies.
    """
    from pyscf import ao2mo, scf

    C = mf.mo_coeff
    ncore = (mol.nelectron - nelecas) // 2
    core = [i for i in range(C.shape[1]) if i not in act][:ncore]
    hcore = mf.get_hcore()
    dm_core = 2.0 * C[:, core] @ C[:, core].T
    # RHF.get_veff (= J - K/2 of the core density) also on an ROHF object, whose own get_veff wants (dma, dmb)
    veff = scf.hf.RHF.get_veff(mf, mol, dm_core) if core else np.zeros_like(hcore)
    ecore = mol.energy_nuc() + np.einsum("ij,ji", dm_core, hcore) + 0.5 * np.einsum("ij,ji", dm_core, veff)
    c_act = C[:, act]
    h1 = c_act.T @ (hcore + veff) @ c_act
    eri = ao2mo.incore.full(mf._eri, c_act) if mf._eri is not None else ao2mo.kernel(mol, c_act)
    return h1, ao2mo.restore(1, eri, len(act)), float(ecore)


def _sector_ground_energy(energy: ElectronicEnergy, ncas: int, nelec: tuple[int, int]) -> float:
    """Exact (FCI) electronic energy in the (n_alpha, n_beta) sector, by dense diagonalisation.

    Jordan-Wigner basis index bit j = occupation of spin orbital j (alpha 0..ncas-1, beta after).
    """
    op = JordanWignerMapper().map(energy.second_q_op())
    n_alpha, n_beta = nelec
    idx = [i for i in range(2 ** (2 * ncas))
           if bin(i & ((1 << ncas) - 1)).count("1") == n_alpha and bin(i >> ncas).count("1") == n_beta]
    H = op.to_matrix(sparse=True)[idx][:, idx].toarray()
    return float(np.linalg.eigvalsh(H)[0])


def _ccsd_energy(mf, frozen: list[int]) -> float:
    """Frozen-core CCSD total energy (PySCF runs UCCSD on an ROHF reference).

    NaN where PySCF's CC module is unavailable (Qollab/Pyodide) or CCSD does not converge.
    """
    try:
        from pyscf import cc
    except (ImportError, OSError):
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
        atom=[(s, xyz) for s, xyz in atoms], basis=spec.basis, unit="Angstrom", verbose=0,
        charge=spec.charge, spin=spec.multiplicity - 1,
    )
    mf = scf.RHF(mol) if spec.multiplicity == 1 else scf.ROHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{type(mf).__name__} not converged for {spec.name} {params}")

    act = _select_active_orbitals(mol, mf, spec)
    ncas, nelecas, nelec = spec.num_active_orbitals, spec.active_electrons, spec.nelec
    h1, h2, ecore = _active_space_integrals(mol, mf, act, nelecas)
    energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    exact = sector_dim(ncas, nelec) <= FCI_MAX_DIM
    e_fci = _sector_ground_energy(energy, ncas, nelec) + ecore if exact else float("nan")
    qubit_op: SparsePauliOp = mapper_for(spec).map(energy.second_q_op()).simplify(atol=1e-12)
    qubit_op = qubit_op.sort()  # canonical lexicographic Pauli order

    e_ccsd = _ccsd_energy(mf, [i for i in range(mf.mo_coeff.shape[1]) if i not in act])

    terms = [
        {"pauli": lbl, "coefficient": float(np.real(c))}
        for lbl, c in zip(qubit_op.paulis.to_labels(), qubit_op.coeffs)
    ]
    assert max(abs(np.imag(c)) for c in qubit_op.coeffs) < 1e-10
    open_shell = nelec[0] != nelec[1]

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
        charge=spec.charge,
        multiplicity=spec.multiplicity,
        num_alpha=nelec[0] if open_shell else -1,
        num_beta=nelec[1] if open_shell else -1,
        reference_method="FCI" if exact else "CCSD" if np.isfinite(e_ccsd) else "HF",
    )
