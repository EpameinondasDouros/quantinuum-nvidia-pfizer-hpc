import datetime as dt
import math
import threading
import time

import pytest

from genqai.data import load_hamiltonians, load_teacher
from genqai.jobs import JobRunner
from genqai.store import ExperimentStore, experiment_id


# ---------- store ----------

@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "experiments")


def test_store_roundtrip(store):
    rec = load_hamiltonians("h2", "test")[0]
    circ = load_teacher("h2")[rec.geometry_id]
    cfg = {"system": "h2", "params": [0.74]}
    eid = store.create(cfg, "transformer_sft+refine", record=rec, meta={"formula": "H2"})
    store.add_circuit(eid, "teacher", circ, gen_log={"forward_passes": 0})
    store.save_grades(eid, {"teacher": {"status": "valid_meets_target"}})
    store.save_grades(eid, {"hf": {"status": "valid_misses_target"}})
    store.add_execution(eid, {"execution_id": "e1", "energy": -1.1})
    store.save_export(eid, "circuit.qasm", "OPENQASM 3.0;")
    got = store.load(eid)
    assert got["record"] == rec and got["circuits"][0]["circuit"] == circ
    assert set(got["grades"]) == {"teacher", "hf"} and got["executions"][0]["execution_id"] == "e1"
    assert got["exports"] == ["circuit.qasm"] and store.read_export(eid, "circuit.qasm") == "OPENQASM 3.0;"
    assert got["method"] == "transformer_sft+refine" and got["config"] == cfg
    (summary,) = store.list()
    assert summary["experiment_id"] == eid and summary["num_circuits"] == 1 and summary["num_executions"] == 1
    store.delete(eid)
    assert store.list() == []


def test_experiment_id_format():
    now = dt.datetime(2026, 10, 10, 12, 0, 0, 123456, tzinfo=dt.timezone.utc)
    eid = experiment_id({"a": 1}, "KRR+refine", now)
    assert eid.startswith("20261010T120000123456Z-krr_refine-") and len(eid.rsplit("-", 1)[1]) == 12
    assert experiment_id({"a": 1}, "x", now) != experiment_id({"a": 2}, "x", now)


@pytest.mark.parametrize("bad", ["../etc", "20261010T120000123456Z-x-../../a", "", "a/b", None])
def test_store_rejects_bad_ids(store, bad):
    with pytest.raises(ValueError):
        store.load(bad)


@pytest.mark.parametrize("bad", ["../x", ".hidden", "a/b", "a" * 100, ""])
def test_store_rejects_bad_export_names(store, bad):
    eid = store.create({"a": 1}, "m")
    with pytest.raises(ValueError):
        store.save_export(eid, bad, "x")


def test_store_ignores_foreign_dirs(store):
    (store.root / "not-an-experiment").mkdir(parents=True)
    assert store.list() == []


# ---------- jobs ----------

def test_jobs_dedupe_and_progress():
    runner = JobRunner()
    gate, calls = threading.Event(), []

    def work(ctx, x):
        calls.append(x)
        ctx.progress("started")
        gate.wait(5)
        return x * 2

    a = runner.submit("k", work, 3)
    b = runner.submit("k", work, 3)
    assert a is b
    gate.set()
    assert a.wait(5).status == "done" and a.result == 6 and calls == [3] and a.messages == ["started"]
    c = runner.submit("k", work, 4)  # finished keys can be re-run
    assert c is not a and c.wait(5).result == 8
    runner.shutdown()


def test_jobs_cooperative_cancel_and_queued_cancel():
    runner = JobRunner(max_workers=1)

    def loop(ctx):
        while True:
            ctx.check()
            time.sleep(0.01)

    running = runner.submit("loop", loop)
    queued = runner.submit("other", lambda ctx: 1)
    while running.status != "running":
        time.sleep(0.01)
    assert queued.status == "queued"
    assert runner.cancel("other") and queued.status == "cancelled"
    assert runner.cancel("loop")
    assert running.wait(5).status == "cancelled"
    assert not runner.cancel("loop")
    runner.shutdown()


def test_jobs_failure_and_timeout():
    runner = JobRunner()
    bad = runner.submit("bad", lambda ctx: 1 / 0).wait(5)
    assert bad.status == "failed" and "ZeroDivisionError" in bad.error

    def slow(ctx):
        while True:
            ctx.check()
            time.sleep(0.01)

    t = runner.submit("slow", slow, timeout=0.1).wait(5)
    assert t.status == "failed" and "timed out" in t.error
    runner.shutdown()


def test_jobs_isolated_process_result_error_and_hard_cancel():
    runner = JobRunner(max_workers=2)
    ok = runner.submit("sqrt", math.sqrt, 16.0, isolate=True)
    err = runner.submit("neg", math.sqrt, -1.0, isolate=True)
    assert ok.wait(30).status == "done" and ok.result == 4.0
    assert err.wait(30).status == "failed" and "ValueError" in err.error
    hang = runner.submit("hang", time.sleep, 60, isolate=True)
    while not hang.messages:
        time.sleep(0.01)
    t0 = time.time()
    runner.cancel("hang")
    assert hang.wait(10).status == "cancelled" and time.time() - t0 < 5
    runner.shutdown()
