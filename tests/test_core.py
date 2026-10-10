import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from genqai.chemistry import SYSTEMS, build_record
from genqai.circuits import build_circuit, gate_stats
from genqai.pool import get_pool
from genqai.sim import Evaluator

CASES = [("h2", (0.74,)), ("lih4", (1.6,)), ("lih4", (3.0,)), ("lih6", (1.6,)), ("h4", (1.0, 1.5))]


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"{c[0]}-{c[1]}")
def rec(request):
    return build_record(SYSTEMS[request.param[0]], request.param[1])


def test_hf_energy_matches_pyscf(rec):
    ev = Evaluator(rec)
    assert abs(ev.energy([], []) - rec.hf_energy) < 1e-7


def test_fci_matches_sector_diagonalization(rec):
    """Acceptance gate A3: qubit-Hamiltonian ground state in the HF sector == PySCF CASCI."""
    ev = Evaluator(rec)
    # full UCCSD-like product with many repeats spans the sector; instead diagonalize the
    # Hamiltonian restricted to the Krylov space generated from HF by H and the pool.
    H = ev.H.toarray()
    basis = [ev.psi0]
    for _ in range(4):
        new = [H @ v for v in basis] + [g @ v for g in ev.G for v in basis]
        M = np.array(basis + new).T
        q, r = np.linalg.qr(M)
        keep = np.abs(np.diag(r)) > 1e-10
        basis = list(q[:, keep].T)
    Q = np.array(basis).T
    e0 = np.linalg.eigvalsh(Q.conj().T @ H @ Q)[0] + rec.energy_shift
    assert abs(e0 - rec.fci_energy) < 1e-7
    assert rec.fci_energy <= rec.ccsd_energy + 1e-7 <= rec.hf_energy + 2e-7


def test_simulator_matches_qiskit_and_vqe_reaches_fci(rec):
    pool = get_pool(rec.system)
    ev = Evaluator(rec)
    rng = np.random.default_rng(0)
    ops = list(range(len(pool)))
    ang = rng.normal(0, 0.3, len(ops))
    qc = build_circuit(rec, ops, ang)
    e_qiskit = float(np.real(Statevector(qc).expectation_value(rec.sparse_pauli_op()))) + rec.energy_shift
    assert abs(e_qiskit - ev.energy(ops, ang)) < 1e-9
    # adjoint gradient vs finite differences
    e, g = ev.energy_and_grad(ops, ang)
    fd = np.array([(ev.energy(ops, ang + 1e-6 * np.eye(len(ops))[i]) - e) / 1e-6 for i in range(len(ops))])
    assert np.allclose(g, fd, atol=1e-5)
    _, e_opt, _ = ev.optimize(ops * 2)
    assert e_opt - rec.fci_energy < 1.6e-3
    assert gate_stats(qc)["cnot_count"] > 0


def test_build_record_matches_pyscf_casci_and_stored_data(rec):
    """RHF-only active-space build == PySCF CASCI, and reproduces the committed dataset."""
    from pyscf import gto, mcscf, scf

    from genqai.data import load_hamiltonians

    mol = gto.M(atom=[(s, xyz) for s, xyz in rec.coordinates], basis=rec.basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    cas = mcscf.CASCI(mf, len(rec.active_orbitals), rec.active_electrons)
    cas.fcisolver.conv_tol = 1e-12
    mo = cas.sort_mo(rec.active_orbitals, base=0)
    assert abs(cas.kernel(mo)[0] - rec.fci_energy) < 1e-8
    assert abs(cas.get_h1eff(mo)[1] - rec.energy_shift) < 1e-8
    stored = {r.geometry_id: r for r in load_hamiltonians(rec.system)}.get(rec.geometry_id)
    if stored is not None:
        a, b = rec.sparse_pauli_op(), stored.sparse_pauli_op()
        assert (a - b).simplify(atol=1e-9).size <= 1 and abs(rec.fci_energy - stored.fci_energy) < 1e-9
