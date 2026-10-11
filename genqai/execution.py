"""Run one circuit's energy measurement on one backend and keep full provenance (ExecutionRecord).

execute() never submits anything wider than the backend, never touches paid backends, and submits
at most ONE job per call (Qollab's IonQ gateway: one single-circuit job per run). IonQ backends are
measured packed by default: every QWC group is a copy of the state on its own qubit register in one
circuit (measure.MeasurementPlan.packed_circuit), width = groups x qubits. If that does not fit, the
call is refused with an explanation -- it does not silently split the groups over several jobs; the
record's notes list the `group_batch` plan that would fit, for the user to decide.

Shots: `shots` is the number of shots of the submitted job. Every group gets `shots` samples either
way; circuit executions are `shots` packed and `num_groups * shots` unpacked.

Statuses: "ok", "dry_run", "refused_width", "refused_paid", "unavailable", "failed".
"""
from __future__ import annotations

import json
import math
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

from genqai.backends import BackendInfo, get_backend_info, injected_info, make_backend
from genqai.circuits import build_circuit
from genqai.measure import (
    MeasurementPlan,
    _backend_name,
    compile_state,
    exact_probabilities,
    packed_width,
    result_counts,
)
from genqai.schema import HamiltonianRecord

STATUSES = ("ok", "dry_run", "refused_width", "refused_paid", "unavailable", "failed")


@dataclass
class ExecutionRecord:
    execution_id: str
    backend: str  # BackendInfo.name ("ionq_simulator_forte-1", "injected", ...)
    noise_model: str | None
    shots: int  # shots of the submitted job (0 for exact statevector)
    packed: bool
    num_groups: int
    width: int  # qubits of the widest submitted circuit
    energy: float | None  # None unless status == "ok"
    stderr: float | None
    error_mha: float | None  # vs FCI, when the record has a finite FCI reference
    job_id: str | None
    compiled_qasm: str | None  # OpenQASM 2 of the transpiled circuit (see compiled_qasm_kind)
    group_details: list[dict]
    runtime_s: float
    software_versions: dict
    status: str
    error: str | None
    # provenance beyond the contract (all optional)
    compiled_qasm_kind: str = ""  # "submitted_circuit" (single circuit) | "state_preparation" (per-group QASM in group_details)
    backend_detail: str = ""  # concrete Qiskit backend name, e.g. "ionq_simulator", "aer_simulator"
    max_qubits: int | None = None
    executions: int = 0  # circuit executions (packed: shots; unpacked: groups x shots)
    seed: int | None = None
    system: str = ""
    geometry_id: str = ""
    problem_key: str = ""
    reference_energy: float | None = None
    reference_kind: str | None = None
    planned: dict = field(default_factory=dict)  # dry-run / refusal plan: circuits, widths, shots, group_batch
    notes: list[str] = field(default_factory=list)
    created_at: str = ""
    circuits: list = field(default_factory=list, repr=False, compare=False)  # transpiled circuits, not serialized

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("circuits")
        return d

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionRecord":
        known = {f.name for f in fields(cls)} - {"circuits"}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "ExecutionRecord":
        return cls.from_dict(json.loads(text))


def _versions() -> dict:
    try:
        from genqai.schema import software_versions

        v = software_versions()
    except ImportError:  # pyscf / qiskit-nature absent (e.g. a light runtime)
        import qiskit

        v = {"python": platform.python_version(), "numpy": np.__version__, "qiskit": qiskit.__version__}
    for mod in ("qiskit_aer", "qiskit_ionq"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return v


def _state(record: HamiltonianRecord, circuit) -> QuantumCircuit:
    """GeneratedCircuit / (ops, angles) -> HF + ansatz; a QuantumCircuit is taken as the state itself."""
    if isinstance(circuit, QuantumCircuit):
        if circuit.num_qubits != record.num_qubits:
            raise ValueError(f"circuit has {circuit.num_qubits} qubits, Hamiltonian {record.num_qubits}")
        return circuit.remove_final_measurements(inplace=False)
    ops, angles = (circuit.operators, circuit.angles) if hasattr(circuit, "operators") else circuit
    return build_circuit(record, ops, angles)


def _qasm(qc: QuantumCircuit) -> str | None:
    try:
        return qasm2.dumps(qc)
    except Exception:  # gates without an OpenQASM 2 definition
        return None


def group_batches(plan: MeasurementPlan, max_qubits: int) -> list[list[int]]:
    """Consecutive group-index batches whose packed width fits max_qubits ([] if one copy doesn't fit)."""
    k = max_qubits // plan.num_qubits
    return [list(range(i, min(i + k, plan.num_groups))) for i in range(0, plan.num_groups, k)] if k else []


def _width_refusal(plan: MeasurementPlan, info: BackendInfo, packed: bool, width: int) -> tuple[str, list[str], dict]:
    n, g, cap = plan.num_qubits, plan.num_groups, info.max_qubits
    if packed:
        local = info.kind in ("exact", "local")
        msg = (f"{g} measurement groups x {n} qubits = {width} qubits in one packed circuit, but {info.name} "
               f"allows {cap}. One job per run (Qollab's IonQ gateway takes a single circuit), so the groups "
               f"cannot be split over several jobs here. Suggestion: "
               + (f"packed=False, which measures each group on its own {n}-qubit circuit." if local else
                  f"run on a local simulator (local_shots, local_noisy or statevector), which measures each "
                  f"group on {n} qubits."))
    else:
        msg = (f"the {n}-qubit state is wider than {info.name}'s {cap} qubits. Suggestion: a smaller active space, "
               f"or a backend with more qubits.")
    notes, planned = [], {"width": width, "max_qubits": cap}
    batches = group_batches(plan, cap) if packed else []
    if len(batches) > 1:
        planned["group_batch"] = batches
        notes.append(f"group_batch (NOT run): {len(batches)} separate packed jobs of <= {len(batches[0])} groups "
                     f"(<= {len(batches[0]) * n} qubits) would fit; that needs {len(batches)} runs and is never "
                     f"done silently.")
    return msg, notes, planned


def execute(record: HamiltonianRecord, circuit, backend: str | object, *, shots: int, seed: int | None = None,
            packed: bool | None = None, dry_run: bool = False) -> ExecutionRecord:
    """Measure <H> of `circuit` (GeneratedCircuit, (ops, angles) or a state QuantumCircuit) on `backend`.

    backend: a name from backends.list_backends() or a Qiskit backend object (recorded as "injected").
    packed: None -> the backend's default (IonQ: packed). statevector + packed reconstructs the
    packed circuit's exact probabilities (a check of the packing itself).
    dry_run: build and transpile, fill width/shots/QASM/planned, submit nothing.
    """
    t0 = time.perf_counter()
    info = get_backend_info(backend) if isinstance(backend, str) else injected_info(backend)
    obj = None if isinstance(backend, str) else backend
    plan = MeasurementPlan.from_record(record)
    exact = info.kind == "exact"
    packed = info.packed_default if packed is None else bool(packed)
    width = packed_width(plan) if packed else plan.num_qubits
    ref_kind = getattr(record, "reference_method", "FCI")
    ref = record.fci_energy if ref_kind == "FCI" and record.fci_energy is not None and math.isfinite(
        record.fci_energy) else None
    rec = ExecutionRecord(
        execution_id=f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}", backend=info.name,
        noise_model=info.noise_model, shots=0 if exact else int(shots), packed=packed, num_groups=plan.num_groups,
        width=width, energy=None, stderr=None, error_mha=None, job_id=None, compiled_qasm=None, group_details=[],
        runtime_s=0.0, software_versions=_versions(), status="", error=None, max_qubits=info.max_qubits, seed=seed,
        system=record.system, geometry_id=record.geometry_id, problem_key=getattr(record, "problem_key", ""),
        reference_energy=ref, reference_kind=ref_kind if ref is not None else None,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if obj is not None:
        rec.backend_detail = _backend_name(obj)

    def done(status: str, error: str | None = None) -> ExecutionRecord:
        rec.status, rec.error, rec.runtime_s = status, error, time.perf_counter() - t0
        return rec

    if info.paid:
        return done("refused_paid", f"{info.name} is a paid backend (hardware): GenQAI never submits to it")
    if not exact and shots < 2:
        raise ValueError("shots must be >= 2 (a stderr needs two samples)")
    if info.max_qubits is not None and width > info.max_qubits:
        msg, rec.notes, rec.planned = _width_refusal(plan, info, packed, width)
        return done("refused_width", msg)
    if not info.available and not dry_run:
        return done("unavailable", info.reason)

    try:
        state = compile_state(_state(record, circuit))
        circuits = plan.measurement_circuits(state, packed)
        if exact:
            tqc, rec.backend_detail = circuits, "statevector"
        else:
            obj = obj or make_backend(info, seed=seed, shots=shots, offline=dry_run and not info.available)
            rec.backend_detail = _backend_name(obj)
            tqc = transpile(circuits, backend=obj, optimization_level=1, seed_transpiler=0)
        rec.circuits = tqc
        if len(tqc) == 1:
            rec.compiled_qasm, rec.compiled_qasm_kind = _qasm(tqc[0]), "submitted_circuit"
        else:
            sp = state if exact else transpile(state, backend=obj, optimization_level=1, seed_transpiler=0)
            rec.compiled_qasm, rec.compiled_qasm_kind = _qasm(sp), "state_preparation"
        if rec.compiled_qasm is None:
            rec.notes.append("compiled circuit is not representable in OpenQASM 2")
        rec.executions = 0 if exact else shots * len(tqc)
        rec.planned = {"circuits": len(tqc), "width": width, "shots": rec.shots, "executions": rec.executions,
                       "qubits_per_circuit": [len({q for i in qc.data for q in i.qubits}) for qc in tqc],
                       "register_qubits": [qc.num_qubits for qc in tqc],  # after layout on the device (IonQ: 29)
                       "depths": [qc.depth() for qc in tqc],
                       "ops": [dict(qc.count_ops()) for qc in tqc]}
        if dry_run:
            if not info.available:
                rec.notes.append(f"dry run only: {info.reason}")
            return done("dry_run")

        if exact:
            dists = [{int(i): float(p[i]) for i in np.flatnonzero(p)} for p in exact_probabilities(tqc)]
            est = plan.estimate(plan.distributions(dists, packed), exact=True)
        else:
            if seed is not None and info.kind != "local":
                rec.notes.append("seed applies to local simulators only")
            job = obj.run(tqc[0] if packed else tqc, shots=shots)
            try:
                rec.job_id = str(job.job_id())
            except Exception:
                pass
            est = plan.estimate(plan.distributions(result_counts(job.result(), len(tqc)), packed))
    except Exception as exc:
        return done("failed", f"{type(exc).__name__}: {exc}")

    if info.kind == "ionq_sim" and not info.noise_model:
        rec.notes.append("IonQ ideal simulator: server returns probabilities; qiskit-ionq samples `shots` "
                         "counts from them client-side")
    if info.noise_model == "genqai_trapped_ion_depolarizing":
        rec.notes.append("local_noisy is GenQAI's depolarizing approximation, NOT IonQ's noise model")
    if packed:
        rec.notes.append("packed: group copies are independent (product state), stderr = sqrt(sum_g Var(E_g))")
    for g, (basis, grp) in enumerate(zip(plan.bases, plan.groups)):
        rec.group_details.append({
            "group": g, "basis": basis, "num_terms": len(grp),
            "coefficient_norm": float(np.abs(np.real(grp.coeffs)).sum()),
            "shots": est.shots_per_group[g], "mean": est.group_means[g], "stderr": est.group_stderrs[g],
            **({"compiled_qasm": _qasm(tqc[g])} if len(tqc) > 1 else {})})
    rec.energy, rec.stderr = est.energy, est.stderr
    rec.error_mha = 1000 * (est.energy - ref) if ref is not None else None
    return done("ok")
