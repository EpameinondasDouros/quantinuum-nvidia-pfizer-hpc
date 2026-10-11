"""Lane C: grading status logic, symmetry checks, exact simplification, pruning guarantees, Pareto."""
import dataclasses

import numpy as np
import pytest

from genqai import grading
from genqai.data import load_hamiltonians, load_teacher
from genqai.grading import BELOW_EXACT, grade, pareto_front, shots_for_stderr, symmetry_checks
from genqai.measure import MeasurementPlan
from genqai.optimize import optimize, simplify
from genqai.schema import GeneratedCircuit
from genqai.sim import Evaluator

FAST = {"ionq_native": False}


@pytest.fixture(scope="module")
def lih6():
    r = load_hamiltonians("lih6", "val")[0]
    return r, load_teacher("lih6")[r.geometry_id]


def _hf(r):
    return GeneratedCircuit(r.geometry_id, r.system, [], [], source="hf")


def test_status_valid_and_misses_target(lih6):
    r, t = lih6
    g = grade(r, t, **FAST)
    assert g.status == "valid_meets_target" and g.checks["variational_bound"] is True
    assert g.energy["reference_kind"] == "FCI" and abs(g.energy["correlation_recovered"] - 1) < 1e-3
    assert g.weighted_score is not None and set(g.weights) == set(g.score_components)
    h = grade(r, _hf(r), **FAST)
    assert h.status == "valid_misses_target" and h.weighted_score is None
    assert abs(h.energy["correlation_recovered"]) < 1e-9


def test_below_exact_reference_is_invalid_never_better(lih6):
    r, t = lih6
    wrong = dataclasses.replace(r, fci_energy=r.fci_energy + 0.010)  # deliberately wrong-shifted reference
    g = grade(wrong, t, **FAST)
    assert g.status == "invalid" and g.checks["variational_bound"] == BELOW_EXACT
    assert g.energy["error_mha"] < 0 and g.weighted_score is None


def test_no_exact_reference_is_unverified(lih6):
    r, t = lih6
    g = grade(dataclasses.replace(r, fci_energy=float("nan")), t, **FAST)
    assert g.status == "unverified_no_reference"
    assert g.energy["reference_kind"] == "CCSD" and "approximate" in g.energy["limitations"]
    assert g.energy["correlation_recovered"] is None


def test_structural_failures_are_invalid(lih6):
    r, t = lih6
    assert grade(r, dataclasses.replace(t, operators=[10_000]), **FAST).status == "invalid"
    assert grade(r, dataclasses.replace(t, angles=[np.nan] * len(t.angles)), **FAST).status == "invalid"
    other = load_hamiltonians("lih6", "val")[1]
    assert grade(other, t, **FAST).checks["same_geometry"] is False


def test_particle_number_violation_detected():
    r = load_hamiltonians("h4", "val")[0]  # Jordan-Wigner: X on qubit 0 changes the electron count
    t = load_teacher("h4")[r.geometry_id]
    psi = Evaluator(r).state(t.operators, t.angles)
    ok = symmetry_checks(r, psi)
    assert ok["particle_number"] and ok["sz"] and abs(ok["particle_number_value"] - 4) < 1e-8
    flipped = psi[np.arange(psi.size) ^ 1]  # X on qubit 0 (bit 0 of the basis index)
    bad = symmetry_checks(r, flipped)
    assert not bad["particle_number"] and not bad["sz"]


def test_exact_merge_and_zero_drop(lih6):
    r, _ = lih6
    ev = Evaluator(r)
    ops, angles = [3, 3, 5, 5, 5, 1, 2, 2], [0.1, -0.05, 0.2, 0.3, -0.1, 0.0, 0.4, -0.4]
    o, a, info = simplify(ops, angles)
    assert o == [3, 5] and np.allclose(a, [0.05, 0.4]) and info == {"merged": 4, "dropped_zero": 2}
    assert abs(ev.energy(ops, angles) - ev.energy(o, a)) < 1e-12


@pytest.mark.parametrize("target", [1.6, 0.3])
def test_pruning_never_returns_a_circuit_failing_target(lih6, target):
    r, t = lih6
    res = optimize(r, t, "min_2q", target_mha=target, max_evals=3000, grade_kw=FAST)
    assert res.after.status == "valid_meets_target" and abs(res.after.energy["error_mha"]) <= target
    assert res.after.resources["cx_qiskit"] <= res.before.resources["cx_qiskit"]
    assert res.same_hamiltonian_check["ok"] and res.evals_used > 0
    for c in res.candidates.values():
        assert abs(1000 * (Evaluator(r).energy(c.operators, c.angles) - r.fci_energy)) <= target


def test_min_cost_uses_no_evaluations(lih6):
    r, t = lih6
    res = optimize(r, t, "min_cost", grade_kw=FAST)
    assert res.evals_used == 1  # the simplify step's single bookkeeping energy, no optimiser
    assert abs(res.after.energy["energy"] - res.before.energy["energy"]) < 1e-12


def test_pareto_excludes_ineligible_and_shallow_never_beats_accurate(lih6):
    r, t = lih6
    shallow = GeneratedCircuit(r.geometry_id, r.system, t.operators[:1], t.angles[:1], source="shallow")
    reports = {"accurate": grade(r, t, **FAST), "shallow": grade(r, shallow, **FAST), "hf": grade(r, _hf(r), **FAST),
               "below": grade(dataclasses.replace(r, fci_energy=r.fci_energy + 0.01), t, **FAST)}
    assert reports["shallow"].resources["cx_qiskit"] < reports["accurate"].resources["cx_qiskit"]
    assert reports["shallow"].status == "valid_misses_target"
    assert pareto_front(reports) == ["accurate"]
    assert reports["shallow"].weighted_score is None


def test_shots_estimate_monotone_in_target(lih6):
    r, t = lih6
    plan = MeasurementPlan.from_record(r)
    probs = grading.group_probabilities(plan, Evaluator(r).state(t.operators, t.angles))
    shots = [shots_for_stderr(plan, probs, s / 1000) for s in (5.0, 1.6, 0.5)]
    assert shots[0] < shots[1] < shots[2]
    assert abs(plan.predicted_stderr(probs, plan.allocate_shots(shots[1])) - 1.6e-3) < 0.05e-3


def test_ionq_native_failure_reported_not_raised(lih6, monkeypatch):
    r, t = lih6

    def boom(qc):
        raise ImportError("qiskit-ionq missing")

    monkeypatch.setattr(grading, "ionq_native_stats", boom)
    g = grade(r, t)
    assert g.resources["ionq_native_2q"] is None and "qiskit-ionq missing" in g.resources["ionq_native_reason"]


def test_precomputed_execution_hook(lih6):
    r, t = lih6
    ex = {"backend": "ionq_simulator", "energy": r.fci_energy + 0.002, "stderr": 0.001, "noise_model": "aria-1",
          "shots": 1000}
    g = grade(r, t, executions=[ex], **FAST)
    (row,) = [v for k, v in g.noise.items() if k.startswith("execution_0")]
    assert abs(row["error_mha"] - 2.0) < 1e-9 and "precomputed" in row["source"]
    assert g.resources["executions"] == 1
