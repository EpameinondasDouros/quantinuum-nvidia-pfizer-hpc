import math
from pathlib import Path

import pytest

from genqai.adapt import run_adapt
from genqai.baselines import NN_METHODS, nearest_train, run_method
from genqai.chemistry import SYSTEMS, mapper_for
from genqai.data import load_hamiltonians, load_teacher
from genqai.evaluate import score_circuit, summarize
from genqai.schema import EvalRecord, GeneratedCircuit, load_jsonl

RESULTS = Path(__file__).resolve().parent.parent / "results"


@pytest.mark.parametrize("system,r", [("h2", 0.7), ("h2", 2.1), ("lih4", 1.6), ("lih4", 3.0)])
def test_adapt_matches_qiskit_adaptvqe(system, r):
    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms import VQE, AdaptVQE
    from qiskit_algorithms.optimizers import L_BFGS_B
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

    rec = min(load_hamiltonians(system), key=lambda h: abs(h.geometry_params["r"] - r))
    spec = SYSTEMS[system]
    n, m = spec.active_electrons // 2, mapper_for(spec)
    hf = HartreeFock(spec.num_active_orbitals, (n, n), m)
    ansatz = UCCSD(spec.num_active_orbitals, (n, n), m, initial_state=hf)
    adapt = AdaptVQE(VQE(StatevectorEstimator(), ansatz, L_BFGS_B()), gradient_threshold=1e-5, max_iterations=30)
    e_qiskit = adapt.compute_minimum_eigenvalue(rec.sparse_pauli_op()).eigenvalue.real + rec.energy_shift
    ours = run_adapt(rec, max_ops=32, grad_tol=1e-5).meta["energy"]
    assert abs(e_qiskit - ours) < 1e-5


class _LoggingBank(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.accessed = set()

    def __getitem__(self, k):
        self.accessed.add(k)
        return super().__getitem__(k)


def test_nn_baselines_only_read_train_labels():
    recs = load_hamiltonians("lih4")
    teacher = load_teacher("lih4")
    train = [r for r in recs if r.split == "train"]
    train_ids = {r.geometry_id for r in train}
    bank = _LoggingBank({g: teacher[g] for g in train_ids})
    for rec in recs[::5]:
        for m in NN_METHODS:
            ev = run_method(m, rec, train, bank)
            assert ev.extra["neighbor_geometry_id"] in train_ids
            assert ev.extra["neighbor_geometry_id"] != rec.geometry_id  # leave-one-out on train
    assert bank.accessed and bank.accessed <= train_ids
    with pytest.raises(ValueError):
        nearest_train(recs[0], recs)  # non-train candidates are refused


@pytest.mark.parametrize("system", ["h2", "lih4", "lih6", "h4"])
def test_committed_nn_results_use_train_neighbors(system):
    train_ids = {r.geometry_id for r in load_hamiltonians(system, "train")}
    for m in NN_METHODS:
        path = RESULTS / system / f"{m}.jsonl"
        if not path.exists():
            pytest.skip(f"{path} not generated")
        for e in load_jsonl(path, EvalRecord):
            assert e.extra["neighbor_geometry_id"] in train_ids


def test_score_circuit_and_summarize():
    rec = load_hamiltonians("lih4", "test_interp")[0]
    t = load_teacher("lih4")[rec.geometry_id]
    hf = score_circuit(rec, GeneratedCircuit(rec.geometry_id, rec.system, [], []), "hf")
    assert abs(hf.energy - rec.hf_energy) < 1e-7 and hf.cnot_count == 0
    zero = GeneratedCircuit(rec.geometry_id, rec.system, t.operators, [0.0] * len(t.operators))
    raw = score_circuit(rec, zero, "m", model_forward_passes=9)
    ref = score_circuit(rec, zero, "m", refine=True, model_forward_passes=9)
    assert raw.energy_evals == 0 and raw.model_forward_passes == ref.model_forward_passes == 9
    assert ref.energy_evals > 0 and ref.optimizer_iterations > 0 and ref.converged
    assert abs(ref.error_mha) < 1e-3 < abs(raw.error_mha)
    bad = score_circuit(rec, GeneratedCircuit(rec.geometry_id, rec.system, [99], [0.1], valid=False), "m")
    assert math.isnan(bad.error_mha) and bad.converged is False
    with pytest.raises(TypeError):
        score_circuit(rec, zero, "m", bogus_cost=1)
    s = summarize([raw, ref, bad])["m"]["test_interp"]
    assert s["n"] == 3 and abs(s["valid_frac"] - 2 / 3) < 1e-12 and abs(s["frac_under_1.6mha"] - 1 / 3) < 1e-12
