import csv
import importlib.util
import io
import json

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from genqai.circuits import build_circuit
from genqai.data import load_hamiltonians, load_teacher
from genqai.export import (
    benchmark_csv,
    hamiltonian_json,
    measurements_csv,
    metadata_json,
    qasm,
    qiskit_code,
)
from genqai.schema import HamiltonianRecord


@pytest.fixture(scope="module", params=["h2", "lih4", "h4"])
def case(request):
    rec = load_hamiltonians(request.param, "test")[0]
    return rec, load_teacher(request.param)[rec.geometry_id]


def _same_up_to_phase(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return abs(abs(np.vdot(a, b)) - 1.0) < tol


def test_qiskit_code_reproduces_statevector(case):
    rec, circ = case
    ns: dict = {}
    exec(compile(qiskit_code(rec, circ), "<export>", "exec"), ns)
    want = Statevector(build_circuit(rec, circ.operators, circ.angles)).data
    got = Statevector(ns["circuit"]).data
    assert np.max(np.abs(got - want)) < 1e-10


def test_qasm2_roundtrip(case):
    from qiskit import qasm2

    rec, circ = case
    text = qasm(rec, circ, version=2)
    assert "OPENQASM 2.0" in text
    loaded = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)  # qelib1 sx
    want = Statevector(build_circuit(rec, circ.operators, circ.angles)).data
    assert _same_up_to_phase(Statevector(loaded).data, want, 1e-8)


def test_qasm3(case):
    from qiskit import qasm3

    rec, circ = case
    text = qasm(rec, circ, version=3)
    assert text.startswith("OPENQASM 3")
    if importlib.util.find_spec("qiskit_qasm3_import") is None:
        pytest.skip("qiskit_qasm3_import not installed: QASM3 text produced but cannot be loaded back")
    want = Statevector(build_circuit(rec, circ.operators, circ.angles)).data
    assert _same_up_to_phase(Statevector(qasm3.loads(text)).data, want, 1e-8)


def test_qasm_rejects_bad_version(case):
    rec, circ = case
    with pytest.raises(ValueError):
        qasm(rec, circ, version=1)


def test_hamiltonian_json_roundtrip(case):
    rec, _ = case
    assert HamiltonianRecord(**json.loads(hamiltonian_json(rec))) == rec


def test_metadata_hashes_are_content_hashes(case):
    rec, circ = case
    a = json.loads(metadata_json({"config": {"system": rec.system}, "record": rec, "circuit": circ}))
    assert {"config_sha256", "record_sha256", "hamiltonian_sha256", "circuit_sha256"} <= set(a["hashes"])
    assert "qiskit" in a["software_versions"]
    circ2 = type(circ)(**{**circ.__dict__, "source": "relabelled"})
    b = json.loads(metadata_json({"config": {"system": rec.system}, "record": rec, "circuit": circ2}))
    assert a["hashes"] == b["hashes"]


def test_measurements_csv_rows():
    ex = {"execution_id": "x1", "backend": "local_shots", "shots": 100, "energy": -1.1, "stderr": 0.01,
          "error_mha": float("nan"), "job_id": None, "status": "done",
          "group_details": [{"basis": "ZZ", "expectation": 0.5}, {"basis": "XX", "expectation": -0.2}]}
    rows = list(csv.DictReader(io.StringIO(measurements_csv(ex))))
    assert len(rows) == 2 and rows[1]["group.basis"] == "XX" and rows[0]["backend"] == "local_shots"
    assert rows[0]["error_mha"] == "nan" and rows[0]["job_id"] == ""
    single = list(csv.DictReader(io.StringIO(measurements_csv({**ex, "group_details": []}))))
    assert len(single) == 1


def test_benchmark_csv_flattens():
    grades = {"sft": {"status": "valid_meets_target", "energy": {"error_mha": 0.1, "thresholds": {"1.6": True}},
                      "resources": {"cx_qiskit": 4}, "cost": {"quantum_evals_to_propose": 0}, "weighted_score": None},
              "hf": {"status": "valid_misses_target", "energy": {"error_mha": 20.0}, "resources": {"cx_qiskit": 0}}}
    rows = list(csv.DictReader(io.StringIO(benchmark_csv(grades))))
    assert [r["name"] for r in rows] == ["sft", "hf"]
    assert rows[0]["energy.thresholds.1.6"] == "True" and rows[0]["resources.cx_qiskit"] == "4"
    assert rows[1]["cost.quantum_evals_to_propose"] == ""
