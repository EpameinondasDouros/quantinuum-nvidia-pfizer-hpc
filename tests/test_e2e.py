"""End-to-end acceptance: user-supplied molecule -> Hamiltonian -> routing -> circuit -> independent check.

Also guards the validated benchmarks: the generic molecule pathway must reproduce the preset numbers.
"""
import json

import numpy as np
import pytest

from genqai import molecule as M
from genqai.data import load_hamiltonians
from genqai.execution import execute
from genqai.generate import GenerationError, generate
from genqai.registry import route
from genqai.sim import Evaluator

HEH_XYZ = "2\nHeH+ cation\nHe 0.0 0.0 0.0\nH 0.0 0.0 0.772\n"


def _pipeline(mol, **cfg):
    config = M.ProblemConfig(**{**M.default_config(mol).__dict__, **cfg}) if cfg else M.default_config(mol)
    record = M.build_problem(config)
    report = M.assess(config, record)
    return config, record, report, route(config, record, report)


def test_custom_molecule_xyz_to_validated_circuit():
    """HeH+ is not a preset: parsed from XYZ, validated, routed to a conventional solver, solved, checked."""
    mol = M.from_xyz(HEH_XYZ, charge=1)
    assert M.validate_molecule(mol) == [] and mol.n_electrons == 2 and mol.charge == 1
    config, record, report, decision = _pipeline(mol)
    assert report.errors == [] and report.num_qubits == record.num_qubits == 2 and report.exact_feasible
    assert record.reference_method == "FCI" and np.isfinite(record.fci_energy)
    assert decision.path == "C" and decision.compatible == []  # no trained generator for HeH+
    with pytest.raises(GenerationError):  # unsupported neural inference is refused, not disguised
        generate(record, "krr", decision)
    circ, log = generate(record, "adapt", decision)
    assert log.quantum_evals_to_propose > 0 and not log.experimental
    e = Evaluator(record).energy(circ.operators, circ.angles)
    assert abs(1000 * (e - record.fci_energy)) < 1.6
    run = execute(record, circ, "statevector", shots=0)
    assert run.status == "ok" and abs(run.energy - e) < 1e-9
    assert json.loads(run.to_json())["energy"] == pytest.approx(e)


def test_smiles_molecule_routes_to_conventional_solver():
    (mol,) = M.from_smiles("O")  # water, ETKDG conformer with a recorded seed
    assert mol.conformer["seed"] == 20261010 and mol.formula == "H2O" and mol.n_electrons == 10
    config, record, report, decision = _pipeline(mol, active_electrons=4, active_orbitals=4)
    assert record.num_qubits == 6 and decision.path == "C"
    assert not report.fits_ionq  # packed single-circuit width exceeds IonQ's 29 qubits; must not be submitted
    refused = execute(record, ([], []), "ionq_simulator_forte-1", shots=100, dry_run=True)
    assert refused.status in ("refused_width", "unavailable")


def test_preset_typed_as_xyz_uses_trained_model_and_matches_benchmark():
    """H2 at 1.33 A entered by hand (rotated, along x) routes to the validated h2 models and reproduces
    the same Hamiltonian as the preset used in the Qollab run."""
    mol = M.from_xyz("2\nH2\nH 1.0 2.0 3.0\nH 2.33 2.0 3.0\n")
    config, record, report, decision = _pipeline(mol)
    preset = M.build_problem(M.default_config(M.from_preset("h2", (1.33,))))
    assert record.sparse_pauli_op().equiv(preset.sparse_pauli_op()) and record.fci_energy == preset.fci_energy
    assert decision.path == "A" and decision.system == "h2" and decision.recommended == "krr"
    circ, log = generate(record, decision.recommended, decision)
    assert log.quantum_evals_to_propose == 0 and log.forward_passes >= 0 and not log.experimental
    assert abs(1000 * (Evaluator(record).energy(circ.operators, circ.angles) - record.fci_energy)) < 1.6


def test_extrapolation_is_labelled_experimental():
    rec = next(r for r in load_hamiltonians("lih4", "test_fresh") if r.geometry_params["r"] > 4.0)
    decision = route(None, rec, None)
    assert decision.path == "B" and "refine" in decision.recommended
    _, log = generate(rec, "transformer_sft", decision)
    assert log.experimental


@pytest.mark.parametrize("system", ["h2", "h4"])
def test_generic_pathway_reproduces_committed_benchmark_hamiltonians(system):
    for stored in load_hamiltonians(system, "test")[:3]:
        params = tuple(stored.geometry_params.values())
        rec = M.build_problem(M.default_config(M.from_preset(system, params)))
        assert rec.sparse_pauli_op().equiv(stored.sparse_pauli_op(), atol=1e-12)
        assert abs(rec.fci_energy - stored.fci_energy) < 1e-12 and rec.hf_state == stored.hf_state


def test_invalid_and_oversized_inputs_are_rejected():
    for bad in (lambda: M.from_smiles("C1CC"), lambda: M.from_xyz("2\nx\nH 0 0 0\n"),
                lambda: M.validate_molecule(M.from_xyz("1\nH atom\nH 0 0 0\n", multiplicity=1))):
        with pytest.raises(M.MoleculeError):
            bad()
    (big,) = M.from_smiles("CCCCCCCC")
    report = M.assess(M.ProblemConfig(big, active_orbitals=20))
    assert report.errors  # path D: reported, never silently shrunk
