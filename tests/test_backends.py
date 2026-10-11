import json
import os
from pathlib import Path

import pytest
from qiskit.providers.basic_provider import BasicSimulator

import genqai.backends as backends
from genqai.backends import IONQ_SIM_MAX_QUBITS, get_backend_info, list_backends
from genqai.data import load_hamiltonians, load_teacher
from genqai.execution import ExecutionRecord, execute
from genqai.sim import Evaluator

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def h2():
    rec = load_hamiltonians("h2", "test")[0]
    return rec, load_teacher("h2")[rec.geometry_id]


@pytest.fixture(scope="module")
def lih4():
    rec = load_hamiltonians("lih4", "test")[0]
    return rec, load_teacher("lih4")[rec.geometry_id]


class CountingBackend(BasicSimulator):
    """BasicSimulator that counts run() calls and can pose as another (e.g. IonQ) backend."""

    def __init__(self, cap=None, name=None):
        super().__init__()
        self.cap, self.runs = cap, 0
        if name:
            self.name = name

    @property
    def num_qubits(self):
        return self.cap

    def run(self, *args, **kwargs):
        self.runs += 1
        return super().run(*args, **kwargs)


def test_list_backends_table():
    infos = {b.name: b for b in list_backends()}
    assert {"statevector", "local_shots", "local_noisy", "ionq_simulator", "ionq_simulator_forte-1",
            "ionq_simulator_aria-1", "injected"} <= set(infos)
    assert all(not b.available for b in infos.values() if b.paid)
    assert any(b.paid for b in infos.values())  # hardware is listed, never available
    for name in ("ionq_simulator", "ionq_simulator_forte-1", "ionq_simulator_aria-1"):
        assert infos[name].max_qubits == IONQ_SIM_MAX_QUBITS and infos[name].needs_key and infos[name].packed_default
    assert infos["ionq_simulator_forte-1"].noise_model == "forte-1"
    assert "NOT IonQ" in infos["local_noisy"].description
    assert not infos["injected"].available


def test_ionq_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(backends, "ionq_available", lambda: False)
    for b in list_backends():
        if b.kind.startswith("ionq"):
            assert not b.available


def test_injected_info():
    info = get_backend_info("injected", injected=BasicSimulator())
    assert info.available and not info.paid and not info.packed_default
    qpu = backends.injected_info(CountingBackend(name="ionq_qpu.forte-1"))
    assert qpu.paid and not qpu.available and qpu.packed_default


def test_execute_injected_backend(h2):
    rec, c = h2
    b = CountingBackend()
    ex = execute(rec, c, b, shots=20_000)
    exact = Evaluator(rec).energy(c.operators, c.angles)
    assert ex.status == "ok" and ex.backend == "injected" and ex.backend_detail == "basic_simulator"
    assert b.runs == 1 and not ex.packed and abs(ex.energy - exact) < 4 * ex.stderr
    assert ex.compiled_qasm_kind == "state_preparation" and all("compiled_qasm" in g for g in ex.group_details)


def test_paid_backends_refused(h2):
    rec, c = h2
    assert execute(rec, c, "ionq_qpu_forte-1", shots=100).status == "refused_paid"
    b = CountingBackend(name="ionq_qpu.forte-1")
    ex = execute(rec, c, b, shots=100)
    assert ex.status == "refused_paid" and b.runs == 0


def test_width_refused_before_submission(h2):
    rec, c = h2
    b = CountingBackend(cap=3, name="ionq_fake_simulator")  # IonQ-like -> packed: 2 groups x 2 qubits = 4 > 3
    ex = execute(rec, c, b, shots=100)
    assert ex.status == "refused_width" and ex.packed and ex.width == 4 and b.runs == 0
    assert "local simulator" in ex.error and ex.energy is None
    ok = execute(rec, c, b, shots=1000, packed=False)  # 2 qubits fit
    assert ok.status == "ok" and b.runs == 1


def test_lih4_on_ionq_refused_with_group_batch_plan(lih4, monkeypatch):
    monkeypatch.setattr(backends, "make_backend", lambda *a, **k: pytest.fail("must not build a backend"))
    rec, c = lih4
    for dry in (False, True):
        ex = execute(rec, c, "ionq_simulator_forte-1", shots=1000, dry_run=dry)
        assert ex.status == "refused_width" and ex.width == 25 * 4 and ex.job_id is None
        batches = ex.planned["group_batch"]
        assert sum(map(len, batches)) == 25 and max(map(len, batches)) * 4 <= IONQ_SIM_MAX_QUBITS
        assert any("NOT run" in n for n in ex.notes)


def test_dry_run_ionq_offline(h2, monkeypatch):
    """Dry run without a key: transpile locally, fill the plan, submit nothing."""
    monkeypatch.setattr(backends, "ionq_available", lambda: False)
    rec, c = h2
    ex = execute(rec, c, "ionq_simulator_forte-1", shots=1000, dry_run=True)
    assert ex.status == "dry_run" and ex.job_id is None and ex.energy is None
    assert ex.packed and ex.width == 4 and ex.noise_model == "forte-1" and ex.executions == 1000
    assert ex.compiled_qasm.startswith("OPENQASM 2.0") and ex.compiled_qasm_kind == "submitted_circuit"
    assert ex.planned["qubits_per_circuit"] == [4] and len(ex.circuits) == 1
    assert execute(rec, c, "ionq_simulator_forte-1", shots=1000).status == "unavailable"


def test_dry_run_local_does_not_run(h2, monkeypatch):
    rec, c = h2
    b = CountingBackend()
    ex = execute(rec, c, b, shots=500, dry_run=True)
    assert ex.status == "dry_run" and b.runs == 0 and ex.planned["circuits"] == 2 and ex.executions == 1000


def test_execution_record_json_round_trip(h2):
    rec, c = h2
    for ex in (execute(rec, c, "local_noisy", shots=2000, seed=3),
               execute(rec, c, "statevector", shots=0, packed=True)):
        assert ex.status == "ok" and ex.software_versions["qiskit"]
        back = ExecutionRecord.from_json(ex.to_json())
        assert back == ex and back.to_dict() == ex.to_dict()
        assert json.loads(ex.to_json())["group_details"][0]["basis"] in ("XX", "ZZ")
    assert ex.noise_model is None and ex.shots == 0 and ex.compiled_qasm


def test_local_noisy_labelled(h2):
    rec, c = h2
    ex = execute(rec, c, "local_noisy", shots=2000, seed=1)
    assert ex.noise_model == "genqai_trapped_ion_depolarizing"
    assert any("NOT IonQ" in n for n in ex.notes)


@pytest.mark.skipif(os.getenv("GENQAI_LIVE_IONQ") != "1", reason="live IonQ simulator run: set GENQAI_LIVE_IONQ=1")
def test_live_ionq_forte1_h2(h2):
    """ONE packed h2 job on IonQ's cloud simulator with the Forte 1 noise model (free, never hardware)."""
    rec, c = h2
    assert get_backend_info("ionq_simulator_forte-1").available
    ex = execute(rec, c, "ionq_simulator_forte-1", shots=2000)
    assert ex.status == "ok", ex.error
    assert ex.packed and ex.width == 4 and ex.job_id
    text = ex.to_json()
    for k in ("IONQ_API_KEY", "QISKIT_IONQ_API_TOKEN", "IONQ_API_TOKEN"):
        if os.getenv(k):
            assert os.environ[k] not in text
    out = ROOT / "results" / "h2" / f"execution_ionq_forte1_{ex.job_id}.json"
    out.write_text(text)
    print(f"\njob {ex.job_id}: E = {ex.energy:.6f} +- {ex.stderr:.6f} Ha, error {ex.error_mha:+.2f} mHa -> {out}")
