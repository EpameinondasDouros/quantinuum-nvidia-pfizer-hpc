"""Lane C: two-stage circuit grading (docs/workbench_contracts.md).

    grade(record, circuit, ...) -> GradeReport
    pareto_front(reports, objectives) -> [ids]

Stage 1 (validity) decides whether a circuit may be ranked at all: right qubit count, operator ids in
the system's pool, finite angles, a normalised state, the reference particle number <N> and <S_z>
(number / magnetisation operators mapped with the system's own qubit mapper), and the variational
bound against the exact energy of the IDENTICAL Hamiltonian. An energy below E_FCI - 1e-8 Ha is never
"better": it means something is inconsistent (wrong Hamiltonian, wrong reference, a bug) and the
circuit is marked invalid. Stage 2 (efficiency: CX, depth, evaluations) only ranks circuits that
pass Stage 1 AND meet the accuracy target, so a shallow inaccurate circuit can never outrank an
accurate one.

Every energy here is an exact statevector expectation (genqai.sim.Evaluator) unless a block says
otherwise; the local noisy energies use GenQAI's own depolarising approximation
(genqai.measure.trapped_ion_noise_model), NOT IonQ's noise model. IonQ-noise numbers only enter
through precomputed execution records (lane D's execute()), passed via `executions=`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from genqai.chemistry import SYSTEMS, mapper_for
from genqai.circuits import build_circuit, resource_stats
from genqai.evaluate import THRESHOLDS_MHA
from genqai.measure import MeasurementPlan, ionq_native_stats, measure_energy
from genqai.pool import get_pool
from genqai.schema import GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator

STATUSES = ("valid_meets_target", "valid_misses_target", "unverified_no_reference", "invalid")
SYMMETRY_TOL = 1e-8  # |<N> - N_ref|, |<S_z> - S_z,ref|
NORM_TOL = 1e-8
VARIATIONAL_TOL_HA = 1e-8
BELOW_EXACT = "below exact ground state of identical Hamiltonian — investigate"
IONQ_MAX_QUBITS = 29  # packed single-circuit width limit used by the Qollab IonQ path
# weighted_score = sum_k w_k * component_k (lower is better); only for Stage-1 eligible circuits.
DEFAULT_WEIGHTS = {"error_over_target": 1.0, "cx_per_100": 0.1, "log10_quantum_evals": 0.1}


@dataclass
class GradeReport:
    status: str
    checks: dict
    energy: dict
    resources: dict
    cost: dict
    noise: dict | None = None
    weighted_score: float | None = None
    weights: dict | None = None
    score_components: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------ references

def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def reference_energy(record: HamiltonianRecord) -> tuple[float | None, str | None, str | None]:
    """(energy, kind, limitation). FCI when the record carries a finite exact energy, else the
    strongest available approximate reference (CCSD, then HF), which is never called exact."""
    method = getattr(record, "reference_method", "FCI") or "FCI"
    if _finite(record.fci_energy) and method.upper() == "FCI":
        return float(record.fci_energy), "FCI", None
    for kind, e in (("CCSD", record.ccsd_energy), ("HF", record.hf_energy)):
        if _finite(e):
            return float(e), kind, (f"No exact (FCI) energy for this Hamiltonian; errors are relative to {kind}, "
                                    f"which is approximate. A circuit can legitimately lie below a {kind} reference.")
    return None, None, "No reference energy available."


def active_space_caveat(record: HamiltonianRecord) -> str:
    return (f"Exact reference = FCI of this {record.basis} active-space Hamiltonian "
            f"({record.active_electrons} electrons in {len(record.active_orbitals)} orbitals, "
            f"{record.mapping} mapping), not the real molecule: basis-set, frozen-core and active-space "
            f"errors are not included in error_mha.")


def reference_spin_sector(record: HamiltonianRecord) -> tuple[int, int]:
    """(n_alpha, n_beta) of the reference state; lane A's -1 default means closed shell."""
    na, nb = getattr(record, "num_alpha", -1), getattr(record, "num_beta", -1)
    if na is None or na < 0 or nb is None or nb < 0:
        na = nb = record.active_electrons // 2
    return int(na), int(nb)


@lru_cache(maxsize=None)
def _symmetry_ops(system: str):
    """Sparse <N> and <S_z> qubit operators, mapped with the system's own mapper."""
    from qiskit_nature.second_q.properties import Magnetization, ParticleNumber

    spec = SYSTEMS[system]
    norb, mapper = spec.num_active_orbitals, mapper_for(spec)
    out = []
    for prop in (ParticleNumber(norb), Magnetization(norb)):
        (fop,) = prop.second_q_ops().values()
        out.append(mapper.map(fop).to_matrix(sparse=True).tocsr())
    return tuple(out)


def symmetry_checks(record: HamiltonianRecord, psi: np.ndarray, tol: float = SYMMETRY_TOL) -> dict:
    """<N> and <S_z> of a prepared statevector vs the record's reference sector."""
    N, Sz = _symmetry_ops(record.system)
    na, nb = reference_spin_sector(record)
    n_val = float(np.real(np.vdot(psi, N @ psi)))
    sz_val = float(np.real(np.vdot(psi, Sz @ psi)))
    return {"particle_number": abs(n_val - (na + nb)) <= tol, "particle_number_value": n_val,
            "particle_number_ref": na + nb, "sz": abs(sz_val - 0.5 * (na - nb)) <= tol, "sz_value": sz_val,
            "sz_ref": 0.5 * (na - nb)}


# ----------------------------------------------------------------------------------- resources

def shots_for_stderr(plan: MeasurementPlan, probs: list[np.ndarray], target_stderr_ha: float) -> int:
    """Total shots (uniform over QWC groups, as measure_energy's default) for an ideal stderr of
    target_stderr_ha: Var = k * sum_g Var_g / N_total, so N_total = k * sum_g Var_g / target^2."""
    k = plan.num_groups
    if k == 0:
        return 0
    s1 = plan.predicted_stderr(probs, [1] * k)  # sqrt(sum_g Var_g), one shot per group
    return int(max(k, math.ceil(k * s1**2 / target_stderr_ha**2)))


def group_probabilities(plan: MeasurementPlan, psi: np.ndarray) -> list[np.ndarray]:
    """Exact outcome probabilities of every measurement group for statevector psi (Qiskit order), using
    the plan's own basis-rotation circuits applied to psi (no re-simulation of the state circuit)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    sv = Statevector(psi)
    return [sv.evolve(qc.remove_final_measurements(inplace=False)).probabilities()
            for qc in plan.circuits(QuantumCircuit(plan.num_qubits))]


def _resources(record, circuit, psi, optimization_level: int, ionq_native: bool, target_stderr_ha: float):
    qc = build_circuit(record, circuit.operators, circuit.angles)
    res = {"qubits": record.num_qubits, "ops": len(circuit.operators), "params": len(circuit.angles),
           **resource_stats(qc, optimization_level)}
    res["depth"] = res["compiled_depth"]
    if ionq_native:
        try:
            nat = ionq_native_stats(qc)
            res["ionq_native_2q"], res["ionq_native"] = nat["native_2q_count"], nat
        except Exception as exc:  # qiskit-ionq missing or its offline transpile failed
            res["ionq_native_2q"], res["ionq_native_reason"] = None, f"{type(exc).__name__}: {exc}"
    else:
        res["ionq_native_2q"], res["ionq_native_reason"] = None, "not requested (ionq_native=False)"
    plan = MeasurementPlan.from_record(record)
    res["measurement_groups"] = plan.num_groups
    res["ionq_packed_qubits"] = plan.num_groups * record.num_qubits
    res["fits_ionq_packed"] = res["ionq_packed_qubits"] <= IONQ_MAX_QUBITS
    probs = group_probabilities(plan, psi)
    res["shots_for_target"] = {"target_stderr_ha": target_stderr_ha, "allocation": "uniform",
                               "total_shots": shots_for_stderr(plan, probs, target_stderr_ha),
                               "note": "ideal (noise-free) sampling estimate from exact group probabilities"}
    return res, plan


# ---------------------------------------------------------------------------------------- cost

def _cost(gen_log) -> dict:
    if gen_log is None:
        return {"source": "no GenerationLog given", "quantum_evals_total": 0}
    g = gen_log if isinstance(gen_log, dict) else vars(gen_log)
    keys = ("method", "model_id", "forward_passes", "candidates", "model_time_s", "quantum_evals_to_propose",
            "refine_energy_evals", "adapt_pool_gradient_evals")
    out = {k: g.get(k) for k in keys}
    out["quantum_evals_total"] = sum(int(g.get(k) or 0) for k in
                                     ("quantum_evals_to_propose", "refine_energy_evals", "adapt_pool_gradient_evals"))
    out["source"] = "GenerationLog"
    return out


# ------------------------------------------------------------------------------------- grading

def _check_structure(record: HamiltonianRecord, circuit: GeneratedCircuit) -> dict:
    pool_size = len(get_pool(record.system))
    ops = list(circuit.operators)
    angles = np.asarray(circuit.angles, float) if len(circuit.angles) else np.zeros(0)
    return {
        "same_geometry": circuit.geometry_id == record.geometry_id and circuit.system == record.system,
        "generator_valid_flag": bool(circuit.valid),
        "qubits": int(get_pool(record.system).generators[0].num_qubits) == record.num_qubits if pool_size else True,
        "operator_ids": all(isinstance(k, (int, np.integer)) and 0 <= k < pool_size for k in ops),
        "params_match_ops": len(ops) == len(angles),
        "finite_params": bool(np.all(np.isfinite(angles))),
    }


def grade(record: HamiltonianRecord, circuit: GeneratedCircuit, *, gen_log=None, target_mha: float = 1.6,
          shots: int | None = None, noise: str | None = None, seed=0, evaluator: Evaluator | None = None,
          optimization_level: int = 1, ionq_native: bool = True, executions: list[dict] | None = None,
          weights: dict | None = None, target_stderr_mha: float | None = None) -> GradeReport:
    """Grade one circuit for one Hamiltonian.

    noise: None (exact only) or "local" (adds finite-shot and GenQAI-noisy local estimates, `shots`
        total, default 10000). executions: precomputed ExecutionRecord-like dicts (e.g. IonQ via lane D),
        reported verbatim with their error vs the reference. evaluator: reuse an Evaluator (its counters
        then include this grading's evaluations). target_stderr_mha: stderr for the shots estimate
        (default = target_mha).
    """
    checks: dict = _check_structure(record, circuit)
    ref, kind, limitation = reference_energy(record)
    energy = {"energy": None, "reference": ref, "reference_kind": kind, "error_mha": None,
              "target_mha": target_mha, "thresholds": {}, "meets_target": False, "correlation_recovered": None,
              "hf_energy": record.hf_energy, "limitations": limitation, "caveat": active_space_caveat(record)}
    cost = _cost(gen_log)
    if not all(checks.values()):
        return GradeReport("invalid", checks, energy, {"qubits": record.num_qubits, "ops": len(circuit.operators)},
                           cost)

    ev = evaluator or Evaluator(record)
    ops, angles = list(circuit.operators), np.asarray(circuit.angles, float)
    psi = ev.state(ops, angles)
    norm = float(np.linalg.norm(psi))
    checks["norm"] = abs(norm - 1.0) <= NORM_TOL
    sym = symmetry_checks(record, psi)
    checks["particle_number"], checks["sz"] = sym["particle_number"], sym["sz"]
    checks["symmetry_values"] = (f"<N>={sym['particle_number_value']:.10f} (ref {sym['particle_number_ref']}), "
                                 f"<Sz>={sym['sz_value']:.10f} (ref {sym['sz_ref']:g})")
    e = ev.energy(ops, angles)
    energy["energy"] = e
    if ref is not None:
        err = 1000.0 * (e - ref)
        energy["error_mha"] = err
        energy["thresholds"] = {f"{t:g}": abs(err) <= t for t in THRESHOLDS_MHA}
        energy["meets_target"] = abs(err) <= target_mha
    if kind == "FCI":
        energy["correlation_recovered"] = ((record.hf_energy - e) / (record.hf_energy - ref)
                                           if abs(record.hf_energy - ref) > 1e-12 else None)
        checks["variational_bound"] = True if e >= ref - VARIATIONAL_TOL_HA else BELOW_EXACT
    else:
        checks["variational_bound"] = "not checked: no exact reference"

    resources, plan = _resources(record, circuit, psi, optimization_level, ionq_native,
                                 (target_stderr_mha or target_mha) / 1000.0)
    resources["executions"] = len(executions or [])

    hard_fail = not all(v is True for k, v in checks.items() if k not in ("symmetry_values", "variational_bound"))
    if hard_fail or checks["variational_bound"] == BELOW_EXACT:
        status = "invalid"
    elif kind != "FCI":
        status = "unverified_no_reference"
    else:
        status = "valid_meets_target" if energy["meets_target"] else "valid_misses_target"

    noise_block = _noise(record, ops, angles, e, ref, noise, shots, seed, plan, executions)
    rep = GradeReport(status, checks, energy, resources, cost, noise_block)
    if status == "valid_meets_target":
        w = dict(weights or DEFAULT_WEIGHTS)
        comps = {"error_over_target": abs(energy["error_mha"]) / target_mha,
                 "cx_per_100": resources["cx_qiskit"] / 100.0,
                 "log10_quantum_evals": math.log10(1 + cost["quantum_evals_total"])}
        rep.weighted_score = float(sum(w[k] * comps[k] for k in w))
        rep.weights, rep.score_components = w, comps
    return rep


def _noise(record, ops, angles, e_exact, ref, noise, shots, seed, plan, executions):
    if noise is None and not executions:
        return None

    def err(x):
        return None if ref is None else 1000.0 * (x - ref)

    out = {"ideal_exact": {"energy": e_exact, "stderr": 0.0, "error_mha": err(e_exact),
                           "source": "exact statevector (live computation)"}}
    if noise == "local":
        n = shots or 10_000
        for backend, label in (("local_shots", "finite-shot ideal sampling (local, Qiskit StatevectorSampler)"),
                               ("local_noisy", "GenQAI trapped-ion depolarising approximation (local Aer) — "
                                               "NOT IonQ's noise model")):
            est = measure_energy(record, ops, angles, backend, shots=n, seed=seed, plan=plan)
            out[backend] = {"energy": est.energy, "stderr": est.stderr, "error_mha": err(est.energy),
                            "total_shots": est.total_shots, "source": label, "meta": est.meta}
    elif noise is not None:
        raise ValueError("noise must be None or 'local' (IonQ noise: pass lane D execution records via executions=)")
    for i, x in enumerate(executions or []):
        out[f"execution_{i}:{x.get('backend', '?')}"] = {
            "energy": x.get("energy"), "stderr": x.get("stderr"), "error_mha": err(x["energy"]) if _finite(x.get("energy")) else None,
            "noise_model": x.get("noise_model"), "shots": x.get("shots"), "job_id": x.get("job_id"),
            "source": "precomputed execution record (not recomputed here)"}
    return out


# -------------------------------------------------------------------------------------- pareto

def objective_value(report: GradeReport, name: str) -> float:
    """Lower is better. error_mha -> |error|; quantum_evals -> total from the GenerationLog; anything
    else is looked up in resources then cost."""
    if name == "error_mha":
        return abs(report.energy["error_mha"])
    if name == "quantum_evals":
        return float(report.cost.get("quantum_evals_total", 0))
    for block in (report.resources, report.cost):
        if name in block and block[name] is not None:
            return float(block[name])
    raise KeyError(f"objective {name!r} not in the report")


def pareto_front(reports: dict[str, GradeReport],
                 objectives=("error_mha", "cx_qiskit", "quantum_evals")) -> list[str]:
    """Non-dominated ids among Stage-1 eligible (status == valid_meets_target) reports only."""
    elig = {k: np.array([objective_value(r, o) for o in objectives])
            for k, r in reports.items() if r.status == "valid_meets_target"}
    return [k for k, v in elig.items()
            if not any(np.all(w <= v) and np.any(w < v) for j, w in elig.items() if j != k)]
