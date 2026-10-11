import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from genqai.circuits import build_circuit
from genqai.data import DATA_DIR, load_hamiltonians, load_teacher
from genqai.execution import execute
from genqai.measure import (
    MeasurementPlan,
    _outcome_int,
    collect_ionq,
    compile_state,
    exact_probabilities,
    measure_energy,
    measure_on_backend,
    packed_width,
    result_counts,
    submit_ionq,
)
from genqai.schema import GeneratedCircuit, HamiltonianRecord, load_jsonl
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


def test_result_counts_list_form():
    class ListResult:  # provider whose get_counts(i) fails and get_counts() returns a list
        def get_counts(self, i=None):
            if i is not None:
                raise KeyError(i)
            return [{"00": 3}, {"11": 5}]

    assert result_counts(ListResult(), 2) == [{"00": 3}, {"11": 5}]
    with pytest.raises(ValueError):
        result_counts(ListResult(), 3)


def test_measure_on_injected_backend(case):
    """The Qollab path: one backend.run(list) on a ready-made backend (BasicSimulator here)."""
    from qiskit.providers.basic_provider import BasicSimulator

    rec, c = case
    est = measure_on_backend(rec, c.operators, c.angles, BasicSimulator(), shots_per_group=20_000)
    exact = Evaluator(rec).energy(c.operators, c.angles)
    assert est.backend == "basic_simulator" and est.total_shots == 20_000 * est.num_groups
    assert abs(est.energy - exact) < 5 * est.stderr + 1e-9


def test_packed_single_circuit_matches_exact():
    """Qollab's IonQ gateway takes one single-circuit job: all QWC groups packed on disjoint qubits."""
    from qiskit_aer import AerSimulator

    from genqai.measure import MeasurementPlan, measure_on_backend

    rec = load_hamiltonians("h2", "test")[0]
    circ = load_teacher("h2")[rec.geometry_id]
    plan = MeasurementPlan.from_record(rec)
    assert plan.packed_circuit(QuantumCircuit(rec.num_qubits)).num_qubits == plan.num_groups * rec.num_qubits
    ev = Evaluator(rec)
    exact = ev.energy(circ.operators, circ.angles)
    est = measure_on_backend(rec, circ.operators, circ.angles, AerSimulator(seed_simulator=1), 200_000, packed=True)
    assert est.meta["packed"] and abs(est.energy - exact) < 4 * est.stderr


# ---------------------------------------------------------------- measurement validation (analytic)

TWO_Q = [("ZI", 0.5), ("XX", 0.3), ("YY", -0.2), ("IZ", 0.1)]  # Qiskit order: qubit 0 rightmost


def analytic_record(terms, n, fci=float("nan")) -> HamiltonianRecord:
    nan = float("nan")
    return HamiltonianRecord(
        geometry_id="analytic", system="analytic", molecule="", geometry_params={}, coordinates=[], basis="",
        active_electrons=0, active_orbitals=[], mapping="", num_qubits=n,
        pauli_terms=[{"pauli": p, "coefficient": c} for p, c in terms], energy_shift=0.0, hf_state="0" * n,
        hf_energy=nan, ccsd_energy=nan, fci_energy=fci)


def _states() -> dict[str, tuple[QuantumCircuit, float | None]]:
    """name -> (2-qubit state, analytic <H> for TWO_Q or None)."""
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    asym = QuantumCircuit(2)
    asym.x(0)  # |q1 q0> = |01>: <ZI> = +1, <IZ> = -1
    prod = QuantumCircuit(2)
    prod.ry(0.3, 0)
    prod.ry(1.1, 1)
    prod.rz(0.4, 1)
    ent = QuantumCircuit(2)
    ent.ry(0.7, 0)
    ent.ry(-1.3, 1)
    ent.cx(0, 1)
    ent.rz(0.9, 0)
    ent.ry(0.5, 1)
    ent.sx(0)
    return {"bell": (bell, 0.5), "asym": (asym, 0.4), "product": (prod, None), "entangled": (ent, None)}


def _direct(terms, state) -> float:
    return float(np.real(Statevector(state).expectation_value(SparsePauliOp.from_list(terms))))


def _grouped_exact(plan, state):
    probs = exact_probabilities(plan.circuits(state))
    return plan.estimate([dict(enumerate(p)) for p in probs], exact=True)


def _packed_exact(plan, state):
    (p,) = exact_probabilities([plan.packed_circuit(state)])
    return plan.estimate(plan.unpack_counts(dict(enumerate(p))), exact=True), p


@pytest.mark.parametrize("name", ["bell", "asym", "product", "entangled"])
def test_analytic_direct_grouped_packed_agree(name):
    state, known = _states()[name]
    rec = analytic_record(TWO_Q, 2)
    plan = MeasurementPlan.from_record(rec)
    assert plan.num_groups == 3 and packed_width(plan) == 6
    direct = _direct(TWO_Q, state)
    if known is not None:
        assert direct == pytest.approx(known, abs=1e-12)
    assert abs(_grouped_exact(plan, state).energy - direct) < 1e-12
    assert abs(_packed_exact(plan, state)[0].energy - direct) < 1e-12
    for packed in (False, True):  # same through the execution layer
        ex = execute(rec, state, "statevector", shots=0, packed=packed)
        assert ex.status == "ok" and abs(ex.energy - direct) < 1e-12 and ex.stderr == 0
        assert ex.error_mha is None  # no finite reference


def test_endianness_asymmetric_state():
    """X on qubit 0 only: Z on qubit 0 (label 'IZ') reads -1, Z on qubit 1 ('ZI') reads +1."""
    state = _states()["asym"][0]
    for label, want in (("IZ", -1.0), ("ZI", 1.0)):
        plan = MeasurementPlan.from_record(analytic_record([(label, 1.0)], 2))
        assert _grouped_exact(plan, state).energy == pytest.approx(want, abs=1e-12)
        assert _packed_exact(plan, state)[0].energy == pytest.approx(want, abs=1e-12)
    plan = MeasurementPlan.from_record(analytic_record(TWO_Q, 2))
    groups = plan.unpack_counts(dict(enumerate(_packed_exact(plan, state)[1])))
    zz = {k: v for k, v in groups[plan.bases.index("ZZ")].items() if v > 1e-12}
    assert zz == pytest.approx({0b01: 1.0})  # qubit 0 = bit 0 in every copy


@pytest.mark.parametrize("sign", [1, -1])
def test_single_qubit_x_eigenstates(sign):
    qc = QuantumCircuit(1)
    if sign < 0:
        qc.x(0)
    qc.h(0)  # |+> or |->
    terms = [("X", 0.7), ("Z", 0.2)]
    rec = analytic_record(terms, 1)
    plan = MeasurementPlan.from_record(rec)
    assert _grouped_exact(plan, qc).energy == pytest.approx(0.7 * sign, abs=1e-12)
    assert _packed_exact(plan, qc)[0].energy == pytest.approx(0.7 * sign, abs=1e-12)
    ex = execute(rec, qc, "local_shots", shots=4000, seed=5)
    xg = ex.group_details[plan.bases.index("X")]
    assert xg["mean"] == pytest.approx(0.7 * sign) and xg["stderr"] == 0  # eigenstate: no spread


def test_outcome_key_formats():
    """int, binary, 0b, hex and space-separated register keys give the same estimate; too wide raises."""
    plan = MeasurementPlan.from_record(analytic_record(TWO_Q, 2))
    probs = exact_probabilities(plan.circuits(_states()["entangled"][0]))
    counts = [{i: int(round(v * 10_000)) for i, v in enumerate(p) if round(v * 10_000)} for p in probs]
    ref = plan.estimate(counts)
    fmts = [lambda i: format(i, "02b"), lambda i: "0b" + format(i, "b"), hex,
            lambda i: f"{i >> 1} {i & 1}"]  # two 1-bit classical registers: last register leftmost
    for f in fmts:
        est = plan.estimate([{f(i): c for i, c in d.items()} for d in counts])
        assert est.energy == ref.energy and est.stderr == ref.stderr
    with pytest.raises(ValueError):
        _outcome_int("101", 2)
    with pytest.raises(ValueError):
        plan.estimate([{"0x4": 1}, {"00": 1}, {"00": 1}])


def _multinomial(rng, p, shots) -> dict:
    p = np.clip(p, 0, None)
    return {i: int(c) for i, c in enumerate(rng.multinomial(shots, p / p.sum())) if c}


@pytest.mark.parametrize("packed", [False, True])
def test_stderr_calibration_coverage(packed):
    """Over 400 independent finite-shot estimates, |E - exact| < 1 stderr ~68%, < 2 stderr ~95%."""
    state = _states()["entangled"][0]
    plan = MeasurementPlan.from_record(analytic_record(TWO_Q, 2))
    exact = _direct(TWO_Q, state)
    probs = exact_probabilities(plan.circuits(state))
    (pp,) = exact_probabilities([plan.packed_circuit(state)])
    rng = np.random.default_rng(20261010)
    z = []
    for _ in range(400):
        if packed:
            est = plan.estimate(plan.unpack_counts(_multinomial(rng, pp, 1000)))
        else:
            est = plan.estimate([_multinomial(rng, p, 1000) for p in probs])
        assert est.shots_per_group == [1000] * 3
        z.append(abs(est.energy - exact) / est.stderr)
    z = np.array(z)
    assert 0.61 < (z < 1).mean() < 0.75
    assert 0.92 < (z < 2).mean() < 0.98
    assert (z < 4).all()
    assert est.stderr == pytest.approx(plan.predicted_stderr(probs, [1000] * 3), rel=0.1)


@pytest.mark.parametrize("packed", [False, True])
def test_finite_shots_within_4_stderr_and_shot_accounting(packed):
    state = _states()["entangled"][0]
    rec = analytic_record(TWO_Q, 2)
    ex = execute(rec, state, "local_shots", shots=20_000, seed=11, packed=packed)
    assert ex.status == "ok" and ex.packed == packed and ex.shots == 20_000
    assert abs(ex.energy - _direct(TWO_Q, state)) < 4 * ex.stderr
    assert [g["shots"] for g in ex.group_details] == [20_000] * 3
    assert ex.executions == (20_000 if packed else 3 * 20_000)
    assert ex.width == (6 if packed else 2)


def test_measure_on_backend_shot_accounting():
    from qiskit_aer import AerSimulator

    rec = load_hamiltonians("h2", "test")[0]
    c = load_teacher("h2")[rec.geometry_id]
    for packed in (False, True):
        est = measure_on_backend(rec, c.operators, c.angles, AerSimulator(seed_simulator=2), 500, packed=packed)
        assert est.shots_per_group == [500, 500] and est.total_shots == 1000
        assert est.meta["executions"] == (500 if packed else 1000)
        assert est.meta["width"] == (4 if packed else 2)


def test_execute_matches_evaluator_on_records(case):
    rec, c = case
    exact = Evaluator(rec).energy(c.operators, c.angles)
    ex = execute(rec, c, "statevector", shots=0)
    assert ex.status == "ok" and abs(ex.energy - exact) < 1e-9
    assert ex.error_mha == pytest.approx(1000 * (exact - rec.fci_energy), abs=1e-6)
    if packed_width(MeasurementPlan.from_record(rec)) <= 24:
        assert abs(execute(rec, c, "statevector", shots=0, packed=True).energy - exact) < 1e-9
    shots = execute(rec, c, "local_shots", shots=20_000, seed=4)
    assert abs(shots.energy - exact) < 4 * shots.stderr
