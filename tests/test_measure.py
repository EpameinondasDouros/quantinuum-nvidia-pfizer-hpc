import json

import numpy as np
import pytest

from genqai.circuits import build_circuit
from genqai.data import DATA_DIR, load_hamiltonians
from genqai.measure import (
    MeasurementPlan,
    collect_ionq,
    compile_state,
    exact_probabilities,
    measure_energy,
    submit_ionq,
)
from genqai.schema import GeneratedCircuit, load_jsonl
from genqai.sim import Evaluator


@pytest.fixture(scope="module", params=["h2", "lih4"])
def case(request):
    rec = load_hamiltonians(request.param, "test")[0]
    teacher = {c.geometry_id: c for c in load_jsonl(DATA_DIR / request.param / "teacher.jsonl", GeneratedCircuit)}
    return rec, teacher[rec.geometry_id]


def test_plan_covers_hamiltonian(case):
    rec, _ = case
    plan = MeasurementPlan.from_record(rec)
    total = sum(len(g) for g in plan.groups)
    assert total == sum(1 for t in rec.pauli_terms if set(t["pauli"]) != {"I"})
    for g, basis in zip(plan.groups, plan.bases):
        for label in g.paulis.to_labels():
            assert all(c == "I" or c == b for c, b in zip(label, basis))


def test_exact_probabilities_reconstruct_energy(case):
    """Basis rotations + qubit ordering + estimator: exact probs reproduce the exact energy."""
    rec, c = case
    plan = MeasurementPlan.from_record(rec)
    circuits = plan.circuits(compile_state(build_circuit(rec, c.operators, c.angles)))
    probs = exact_probabilities(circuits)
    est = plan.estimate([dict(enumerate(p)) for p in probs], exact=True)
    assert abs(est.energy - Evaluator(rec).energy(c.operators, c.angles)) < 1e-9
    # also via bitstring keys (the format samplers/IonQ return)
    n = rec.num_qubits
    est2 = plan.estimate([{format(i, f"0{n}b"): v for i, v in enumerate(p)} for p in probs], exact=True)
    assert abs(est2.energy - est.energy) < 1e-12


@pytest.mark.parametrize("allocation", ["uniform", "weighted"])
def test_local_shots_converge(case, allocation):
    rec, c = case
    exact = Evaluator(rec).energy(c.operators, c.angles)
    est = measure_energy(rec, c.operators, c.angles, "local_shots", shots=200_000, allocation=allocation, seed=7)
    assert est.total_shots == 200_000
    assert 0 < est.stderr < 0.005
    assert abs(est.energy - exact) < 3.5 * est.stderr


def test_stderr_matches_prediction(case):
    """Reported sample stderr agrees with the stderr predicted from exact probabilities."""
    rec, c = case
    pred = measure_energy(rec, c.operators, c.angles, "exact_probs", shots=50_000)
    est = measure_energy(rec, c.operators, c.angles, "local_shots", shots=50_000, seed=3)
    assert est.stderr == pytest.approx(pred.meta["predicted_stderr_at_shots"]["stderr"], rel=0.1)


def test_allocation_sums():
    rec = load_hamiltonians("lih4", "test")[0]
    plan = MeasurementPlan.from_record(rec)
    for m in ("uniform", "weighted"):
        a = plan.allocate_shots(10_007, m)
        assert sum(a) == 10_007 and min(a) >= 1 and len(a) == plan.num_groups


def test_ionq_job_construction_offline(case, monkeypatch):
    """One job carrying every group circuit, noise model and shots -- no network."""
    from qiskit_ionq import ionq_client
    from qiskit_ionq.helpers import qiskit_to_ionq

    from genqai.measure import ionq_backend

    sent = {}

    def fake_submit(self, job):
        sent["payload"] = json.loads(qiskit_to_ionq(job.circuit, job.backend(), job._passed_args,
                                                    job.extra_query_params, job.extra_metadata))
        return {"id": "fake-job"}

    monkeypatch.setattr(ionq_client.IonQClient, "submit_job", fake_submit)
    rec, c = case
    plan = MeasurementPlan.from_record(rec)
    circuits = plan.circuits(compile_state(build_circuit(rec, c.operators, c.angles)))
    job = submit_ionq(circuits, 500, "forte-1", backend=ionq_backend(token="fake"))
    p = sent["payload"]
    assert job.job_id() == "fake-job"
    assert p["shots"] == 500 and p["noise"]["model"] == "forte-1"
    assert p["backend"] == "simulator"
    body = p["input"]
    assert len(body["circuits"]) == plan.num_groups


def test_collect_ionq_parses_result(case):
    rec, c = case
    plan = MeasurementPlan.from_record(rec)
    probs = exact_probabilities(plan.circuits(compile_state(build_circuit(rec, c.operators, c.angles))))
    n = rec.num_qubits

    class FakeResult:
        def get_probabilities(self, i):
            return {format(k, f"0{n}b"): v for k, v in enumerate(probs[i]) if v > 0}

        def get_counts(self, i):
            return {format(k, f"0{n}b"): int(round(v * 1000)) for k, v in enumerate(probs[i]) if round(v * 1000)}

    exact = Evaluator(rec).energy(c.operators, c.angles)
    est = plan.estimate(collect_ionq(FakeResult(), plan.num_groups, exact=True), exact=True)
    assert abs(est.energy - exact) < 1e-9
    noisy = plan.estimate(collect_ionq(FakeResult(), plan.num_groups, exact=False))
    assert np.isfinite(noisy.stderr) and noisy.total_shots > 0
