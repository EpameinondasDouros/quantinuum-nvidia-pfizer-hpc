"""genqai.molecule: input parsing, validation, preset routing, and real custom-molecule Hamiltonians."""
import ast
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.operators import FermionicOp

from genqai import chemistry
from genqai.adapt import run_adapt
from genqai.chemistry import SYSTEMS, build_record, mapper_for
from genqai.circuits import build_circuit
from genqai.molecule import (
    MoleculeError, ProblemConfig, assess, build_problem, default_config, from_atoms, from_preset, from_smiles,
    from_xyz, packed_width, preset_family,
)
from genqai.schema import HamiltonianRecord, dump_jsonl, load_jsonl
from genqai.sim import Evaluator

ROOT = Path(__file__).resolve().parent.parent


def _same_record(a: HamiltonianRecord, b: HamiltonianRecord) -> bool:
    da, db = asdict(a), asdict(b)
    for d in (da, db):
        d.pop("problem_key")
    return da == db


@pytest.mark.parametrize("system,params", [("h2", (0.74,)), ("lih4", (1.6,)), ("lih6", (2.2,)), ("h4", (1.0, 1.5))])
def test_preset_build_problem_is_identical_to_build_record(system, params):
    mol = from_preset(system, params)
    cfg = default_config(mol)
    spec = SYSTEMS[system]
    assert (cfg.active_electrons, cfg.active_orbitals, cfg.mapping) == (spec.active_electrons, spec.num_active_orbitals, spec.mapping)
    rec = build_problem(cfg)
    assert rec.system == system and rec.problem_key == cfg.key
    assert _same_record(rec, build_record(spec, params))


@pytest.mark.parametrize("text,system,params", [
    ("2\nH2 at 0.9\nH 0.0 0.0 0.0\nH 0.0 0.0 0.9\n", "h2", (0.9,)),
    ("Li 0 0 0\nH 0 0 2.0", "lih4", (2.0,)),
    ("H 1.0 2.0 3.0\nLi 1.0 4.0 3.0", "lih4", (2.0,)),  # other orientation and atom order: same family
    ("H 0 0 0\nH 0 0 1.0\nH 0 0 2.5\nH 0 0 3.5", "h4", (1.0, 1.5)),
])
def test_presets_entered_as_xyz_route_to_the_preset_record(text, system, params):
    mol = from_xyz(text)
    assert mol.source == "xyz" and preset_family(mol) == (system, params)
    rec = build_problem(default_config(mol))
    assert rec.system == system
    assert _same_record(rec, build_record(SYSTEMS[system], params))
    # a non-preset setting (other mapping) on the same molecule is a custom system, never the preset
    other = build_problem(ProblemConfig(mol, mapping="jordan_wigner" if SYSTEMS[system].mapping == "parity" else "parity"))
    assert other.system.startswith("mol-")


def test_keys_hash_physical_content():
    a = from_xyz("H 0 0 0\nH 0 0 0.74")
    b = from_atoms([("h", (0, 0, 1e-8)), ("H", (0, 0, 0.7400000004))], name="other name")
    assert a.key == b.key and a.formula == "H2" and a.n_electrons == 2
    assert from_xyz("H 0 0 0\nH 0 0 0.74", charge=1, multiplicity=2).key != a.key
    assert from_atoms(a.atoms, basis="6-31g").key != a.key
    assert ProblemConfig(a).key == ProblemConfig(b).key != ProblemConfig(a, mapping="jordan_wigner").key
    assert from_xyz("He 0 0 0\nH 0 0 0.772", charge=1).atoms[1][1] == (0.0, 0.0, 0.772)  # verbatim


def _sector_fci(rec: HamiltonianRecord) -> float:
    """Independent check: dense diagonalisation of the stored qubit H in the (n_alpha, n_beta) sector.

    Number operators are diagonal in both Jordan-Wigner and parity encodings."""
    spec = SYSTEMS[rec.system]
    n = spec.num_active_orbitals
    mapper = mapper_for(spec)
    occ = [np.real(mapper.map(FermionicOp({f"+_{p} -_{p}": 1.0 for p in rng}, num_spin_orbitals=2 * n))
                   .to_matrix(sparse=True).diagonal()) for rng in (range(n), range(n, 2 * n))]
    idx = np.flatnonzero((np.abs(occ[0] - rec.nelec[0]) < 1e-9) & (np.abs(occ[1] - rec.nelec[1]) < 1e-9))
    H = rec.sparse_pauli_op().to_matrix(sparse=True)[idx][:, idx].toarray()
    return float(np.linalg.eigvalsh(H)[0]) + rec.energy_shift


def _casci(rec: HamiltonianRecord) -> tuple[float, float]:
    from pyscf import gto, mcscf, scf

    mol = gto.M(atom=[(s, xyz) for s, xyz in rec.coordinates], basis=rec.basis, unit="Angstrom", verbose=0,
                charge=rec.charge, spin=rec.multiplicity - 1)
    mf = (scf.RHF if rec.multiplicity == 1 else scf.ROHF)(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    cas = mcscf.CASCI(mf, len(rec.active_orbitals), rec.nelec)
    cas.fcisolver.conv_tol = 1e-12
    mo = cas.sort_mo(rec.active_orbitals, base=0)
    return float(cas.kernel(mo)[0]), float(cas.get_h1eff(mo)[1])


ACCEPTANCE = {
    # name: (config factory, qubits, (n_alpha, n_beta), active orbitals)
    "HeH+": (lambda: default_config(from_xyz("2\nHeH+\nHe 0 0 0\nH 0 0 0.772", charge=1)), 2, (1, 1), 2),
    "H2O": (lambda: ProblemConfig(from_smiles("O")[0], active_electrons=4, active_orbitals=4), 6, (2, 2), 4),
    "OH": (lambda: ProblemConfig(from_smiles("[OH]")[0], active_orbitals=4), 6, (3, 2), 4),
}


@pytest.fixture(scope="module", params=list(ACCEPTANCE))
def acceptance(request):
    make, q, nelec, norb = ACCEPTANCE[request.param]
    cfg = make()
    return request.param, cfg, build_problem(cfg), (q, nelec, norb)


def test_acceptance_molecules_references(acceptance):
    name, cfg, rec, (q, nelec, norb) = acceptance
    rep = assess(cfg, rec)
    assert not rep.errors and rep.exact_feasible
    assert (rep.num_qubits, rep.active_electrons, rep.active_orbitals) == (q, nelec, norb)
    assert (rec.num_qubits, rec.nelec, len(rec.active_orbitals)) == (q, nelec, norb)
    assert rec.system.startswith("mol-") and rec.reference_method == "FCI" and rec.problem_key == cfg.key
    assert rec.multiplicity == cfg.molecule.multiplicity and rec.charge == cfg.molecule.charge
    assert rep.est_pauli_terms >= len(rec.pauli_terms)
    assert rep.ionq_packed_qubits == packed_width(rec) and rep.fits_ionq == (packed_width(rec) <= 29)
    e_cas, ecore = _casci(rec)
    assert abs(rec.fci_energy - e_cas) < 1e-8 and abs(rec.energy_shift - ecore) < 1e-8
    assert abs(_sector_fci(rec) - rec.fci_energy) < 1e-8
    assert abs(Evaluator(rec).energy([], []) - rec.hf_energy) < 1e-8  # HF = <HF|H|HF> + shift
    assert rec.fci_energy <= rec.ccsd_energy + 1e-6 and rec.ccsd_energy <= rec.hf_energy + 1e-6


def test_acceptance_molecules_adapt_reaches_fci(acceptance, tmp_path):
    _, _, rec, _ = acceptance
    circ = run_adapt(rec, max_ops=30)
    assert circ.meta["error_mha"] < 1.6
    qc = build_circuit(rec, circ.operators, circ.angles)  # pools/circuits see the registered system
    e = float(np.real(Statevector(qc).expectation_value(rec.sparse_pauli_op()))) + rec.energy_shift
    assert abs(e - Evaluator(rec).energy(circ.operators, circ.angles)) < 1e-9
    dump_jsonl([rec], tmp_path / "r.jsonl")
    assert _same_record(load_jsonl(tmp_path / "r.jsonl", HamiltonianRecord)[0], rec)


def test_explicit_orbital_indices_override():
    cfg = ACCEPTANCE["H2O"][0]()
    rec = build_problem(cfg)
    same = build_problem(ProblemConfig(cfg.molecule, active_orbital_indices=tuple(rec.active_orbitals)))
    assert same.system != rec.system and abs(same.fci_energy - rec.fci_energy) < 1e-10
    with pytest.raises(MoleculeError, match="core orbital"):
        build_problem(ProblemConfig(cfg.molecule, active_orbital_indices=(0, 4, 5)))


def test_reference_falls_back_when_fci_is_infeasible(monkeypatch):
    monkeypatch.setattr(chemistry, "FCI_MAX_DIM", 1)
    cfg = default_config(from_xyz("He 0 0 0\nH 0 0 0.9", charge=1))
    rep = assess(cfg)
    assert not rep.exact_feasible and any("not exact" in w for w in rep.warnings)
    rec = build_problem(cfg)
    assert np.isnan(rec.fci_energy) and rec.reference_method == "CCSD" and rec.reference_energy == rec.ccsd_energy


def test_auto_active_space_and_resource_errors():
    ethanol = from_smiles("CCO")[0]
    rep = assess(default_config(ethanol))
    assert not rep.errors and rep.num_qubits <= 12 and any("auto active space" in w for w in rep.warnings)
    big = ProblemConfig(ethanol, active_orbitals=9)
    rep = assess(big)
    assert rep.errors and "active_orbitals=7" in rep.errors[0] and rep.active_orbitals == 9  # reported, not shrunk
    with pytest.raises(MoleculeError, match="exceeds the limit"):
        build_problem(big)
    with pytest.raises(MoleculeError, match="exceed the 7 orbitals"):
        build_problem(ProblemConfig(from_smiles("O")[0], active_orbitals=8))
    with pytest.raises(MoleculeError, match="must differ by 2S"):
        build_problem(ProblemConfig(from_smiles("[OH]")[0], active_electrons=(2, 2), active_orbitals=3))


def test_smiles_conformers_and_metadata():
    a = from_smiles("CCO", optimizer="MMFF", num_conformers=2)
    b = from_smiles("CCO", optimizer="MMFF", num_conformers=2)
    assert len(a) == 2 and [m.key for m in a] == [m.key for m in b]  # recorded seed -> reproducible
    c = a[1].conformer
    assert c["method"] == "ETKDGv3" and c["seed"] == 20261010 and c["optimizer"] == "MMFF" and c["conformer_id"] == 1
    assert a[0].formula == "C2H6O" and a[0].smiles == "CCO" and a[0].source == "smiles"
    assert from_smiles("[NH4+]")[0].charge == 1 and from_smiles("[CH2]")[0].multiplicity == 3
    pair = [("H", (0, 0, 0)), ("H", (0, 0, 0.74)), ("H", (0, 5, 0)), ("H", (0, 5, 0.74))]
    assert any("disconnected" in w for w in assess(ProblemConfig(from_atoms(pair, smiles="[H][H].[H][H]"))).warnings)
    assert any("non-bonded fragments" in w for w in assess(ProblemConfig(from_atoms(pair))).warnings)


@pytest.mark.parametrize("make,match", [
    (lambda: from_smiles("C1CC"), "invalid SMILES"),
    (lambda: from_smiles("[Og]"), "no functions for element Og"),
    (lambda: from_smiles("O", multiplicity=2), "multiplicity 2 is impossible"),
    (lambda: from_xyz("H 0 0 0"), "multiplicity 1 is impossible"),
    (lambda: from_xyz("3\n\nH 0 0 0\nH 0 0 0.7\n"), "declares 3 atoms"),
    (lambda: from_xyz("H 0 0 0\nH 0 0 0.74 1"), "malformed XYZ line 2"),
    (lambda: from_xyz("H 0 0 0\nH 0 0 nan"), "malformed XYZ line 2"),
    (lambda: from_xyz("__import__('os') 0 0 0"), "malformed XYZ"),
    (lambda: from_xyz("Xx 0 0 0"), "unknown element"),
    (lambda: from_xyz("H 0 0 0\nH 0 0 0.05"), "overlapping"),
    (lambda: from_xyz("H 0 0 0\n" * 40), "exceeds the limit of 30"),
    (lambda: from_xyz("H 0 0 0\n" + " " * 70000), "larger than 64 KB"),
    (lambda: from_smiles("C" * 12), "over the limit of 30"),
    (lambda: from_smiles("CCO", optimizer="xtb"), "optimizer must be"),
    (lambda: from_preset("benzene", (1.0,)), "unknown preset"),
])
def test_invalid_inputs_raise_molecule_error(make, match):
    with pytest.raises(MoleculeError, match=match):
        make()


def test_pyodide_safe_imports():
    """No module-level PySCF fci/mcscf/cc or RDKit import (Qollab's runtime lacks them)."""
    for mod in ("chemistry", "molecule", "pool", "schema"):
        tree = ast.parse((ROOT / "genqai" / f"{mod}.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
        assert not [x for x in names if "rdkit" in x or x in ("fci", "mcscf", "cc") or "pyscf" in x], (mod, names)
