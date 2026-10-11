"""Measured (shot-based) energies: qubit-wise-commuting grouping, local and IonQ backends.

Estimator
---------
H = c_I * I + sum_g sum_{j in g} h_j P_j, grouped into qubit-wise-commuting (QWC) groups g.
Each group is measured in one product basis (X -> H, Y -> Sdg then H, Z/I -> nothing), so every
P_j in g is diagonal there: P_j(b) = (-1)^{parity(b & support_j)} for outcome bitstring b.

Per shot b of group g the estimator is f_g(b) = sum_{j in g} h_j P_j(b). E_g = mean of f_g over
the group's N_g shots; it is unbiased for sum_j h_j <P_j>. Because all P_j in a group are read from
the SAME shot, their covariances are included automatically by taking the sample variance of f_g
(not summing per-term variances):
    Var(E_g) = s_g^2 / N_g,  s_g^2 = N_g/(N_g-1) * (mean(f_g^2) - mean(f_g)^2)
Groups use independent shots, so Var(E) = sum_g Var(E_g) and
    E = sum_g E_g + c_I + energy_shift,   stderr = sqrt(sum_g Var(E_g)).
With exact probabilities (no sampling) E_g = sum_b p(b) f_g(b) and stderr = 0; the same
probabilities give the *predicted* stderr for a shot allocation via Var_p(f_g) / N_g.

Packed single-circuit scheme (MeasurementPlan.packed_circuit): copy g of the state sits on qubits
[g*n, (g+1)*n) in group g's basis. The copies form a product state, so in every shot the G
sub-bitstrings are independent samples, one per group. One run of S shots therefore gives EACH group
S samples (circuit executions = S, group samples = G*S), and Var(E) = sum_g Var(E_g) holds exactly as
for separate circuits (ideal case; device crosstalk between copies would add correlations we ignore).
Its width is packed_width(plan) = G*n qubits.

Outcome keys
------------
Distributions may be keyed by int (bit q = qubit q), binary strings (qubit 0 rightmost, "0b" prefix
allowed), hex strings ("0x..") or space-separated classical registers ("c1 c0", Qiskit's format:
the last register is leftmost, so removing the spaces gives the right integer). A key wider than the
measured register raises (a register-order bug, not data).

Backends
--------
exact        Evaluator (sparse statevector) expectation. No circuits, no shots.
exact_probs  Statevector probabilities of every measurement circuit, reconstructed through the
             grouping estimator (checks basis rotations/qubit order; stderr is 0).
local_shots  Qiskit StatevectorSampler, ideal finite-shot sampling.
local_noisy  Aer + trapped_ion_noise_model(): OUR depolarizing approximation, NOT IonQ's model.
ionq_ideal   qiskit-ionq "simulator", noise_model="ideal". IonQ's ideal simulator returns exact
             probabilities (no sampling), so stderr = 0 -- but the API rounds them to 3 decimals,
             which leaves a small deterministic bias (meta["probability_rounding_bound_ha"]).
ionq_noisy   qiskit-ionq "simulator" with an IonQ noise model ("aria-1", "forte-1", ...): real
             finite-shot sampling with IonQ's own noise model. All groups go in ONE job.
IonQ backends need IONQ_API_KEY in the environment or a .env file (pip: qiskit-ionq).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

from genqai.circuits import BASIS, build_circuit
from genqai.schema import HamiltonianRecord

BACKENDS = ("exact", "exact_probs", "local_shots", "local_noisy", "ionq_ideal", "ionq_noisy")
SHOT_BACKENDS = ("local_shots", "local_noisy", "ionq_noisy")
IONQ_IDEAL_SAMPLER_SHOTS = 1_000_000  # client-side only; see submit_ionq
IONQ_PROB_RESOLUTION = 1e-3  # IonQ's ideal simulator returns probabilities rounded to 3 decimals

# GenQAI's own trapped-ion-like approximation (rough Aria/Forte-class public fidelities).
# NOT IonQ's noise model: 1q depolarizing on sx/x (rz is virtual, error-free), 2q depolarizing
# on every CX (~one MS/ZZ per CX on all-to-all hardware), symmetric readout flip.
TRAPPED_ION_NOISE = {"p1": 5e-4, "p2": 6e-3, "readout": 5e-3}


@dataclass
class MeasurementPlan:
    """QWC groups of the non-identity Pauli terms of one Hamiltonian."""

    num_qubits: int
    groups: list[SparsePauliOp]
    bases: list[str]  # per group, Qiskit-order label of X/Y/Z/I (qubit 0 rightmost)
    identity_coeff: float
    energy_shift: float

    @classmethod
    def from_record(cls, record: HamiltonianRecord) -> "MeasurementPlan":
        op = record.sparse_pauli_op().simplify(atol=0)
        is_id = ~(op.paulis.x | op.paulis.z).any(axis=1)
        identity = float(np.real(op.coeffs[is_id].sum()))
        rest = op[~is_id]
        groups = rest.group_commuting(qubit_wise=True) if len(rest) else []
        bases = []
        for g in groups:
            x, z = g.paulis.x.any(axis=0), g.paulis.z.any(axis=0)  # index = qubit
            chars = ["Y" if xi and zi else "X" if xi else "Z" if zi else "I" for xi, zi in zip(x, z)]
            bases.append("".join(reversed(chars)))
        return cls(record.num_qubits, groups, bases, identity, record.energy_shift)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def measurement_circuits(self, state: QuantumCircuit, packed: bool) -> list[QuantumCircuit]:
        """What one job submits: [packed_circuit] or one circuit per group."""
        return [self.packed_circuit(state)] if packed else self.circuits(state)

    def distributions(self, counts: list[dict], packed: bool) -> list[dict]:
        """Per-group distributions from a job's per-circuit counts (see measurement_circuits)."""
        if packed:
            if len(counts) != 1:
                raise ValueError(f"packed job must return 1 count dict, got {len(counts)}")
            return self.unpack_counts(counts[0])
        return counts

    def circuits(self, state: QuantumCircuit) -> list[QuantumCircuit]:
        """One measurement circuit per group: state + basis rotation + measure_all."""
        out = []
        for i, basis in enumerate(self.bases):
            qc = state.copy(name=f"group{i}_{basis}")
            for q, c in enumerate(reversed(basis)):
                if c == "X":
                    qc.h(q)
                elif c == "Y":
                    qc.sdg(q)
                    qc.h(q)
            qc.measure_all()
            out.append(qc)
        return out

    def packed_circuit(self, state: QuantumCircuit) -> QuantumCircuit:
        """ALL groups in ONE circuit: copy g of the state on qubits [g*n, (g+1)*n), rotated into group
        g's basis. Copies are a product state, so every shot samples each group independently.
        For backends that take one single-circuit job per run (Qollab's IonQ gateway)."""
        n = self.num_qubits
        packed = QuantumCircuit(n * self.num_groups, name="packed_groups")
        for g, qc in enumerate(self.circuits(state)):
            packed.compose(qc.remove_final_measurements(inplace=False), qubits=range(g * n, (g + 1) * n),
                           inplace=True)
        packed.measure_all()
        return packed

    def unpack_counts(self, counts: dict) -> list[dict]:
        """Per-group marginal counts from a packed_circuit result."""
        n, mask = self.num_qubits, (1 << self.num_qubits) - 1
        out = [dict() for _ in range(self.num_groups)]
        for key, c in counts.items():
            b = _outcome_int(key, packed_width(self))
            for g in range(self.num_groups):
                sub = (b >> (g * n)) & mask
                out[g][sub] = out[g].get(sub, 0) + c
        return out

    def _group_values(self, g: int, outcomes: np.ndarray) -> np.ndarray:
        """f_g(b) for integer outcomes b (bit q of b = qubit q)."""
        grp = self.groups[g]
        support = (grp.paulis.x | grp.paulis.z).astype(np.int64)  # terms x qubits
        bits = (outcomes[:, None] >> np.arange(self.num_qubits)) & 1  # outcomes x qubits
        signs = 1 - 2 * ((bits @ support.T) % 2)
        return signs @ np.real(grp.coeffs)

    def estimate(self, dists: list[dict], exact: bool = False) -> "EnergyEstimate":
        """Energy from per-group outcome distributions.

        dists[g]: {bitstring|int: count} (exact=False) or {bitstring|int: probability} (exact=True).
        """
        if len(dists) != self.num_groups:
            raise ValueError(f"expected {self.num_groups} distributions, got {len(dists)}")
        means, variances, shots = [], [], []
        for g, d in enumerate(dists):
            keys = np.array([_outcome_int(k, self.num_qubits) for k in d], dtype=np.int64)
            w = np.array(list(d.values()), dtype=float)
            f = self._group_values(g, keys)
            n = w.sum()
            p = w / n
            m = float(p @ f)
            means.append(m)
            if exact:
                variances.append(0.0)
                shots.append(0)
            else:
                n = int(round(n))
                s2 = (float(p @ f**2) - m**2) * n / (n - 1) if n > 1 else float("inf")
                variances.append(max(s2, 0.0) / n)
                shots.append(n)
        energy = float(sum(means)) + self.identity_coeff + self.energy_shift
        return EnergyEstimate(
            energy=energy,
            stderr=float(np.sqrt(sum(variances))),
            num_groups=self.num_groups,
            total_shots=int(sum(shots)),
            shots_per_group=shots,
            group_means=means,
            group_stderrs=[float(np.sqrt(v)) for v in variances],
        )

    def predicted_stderr(self, probs: list[np.ndarray], shots: list[int]) -> float:
        """Ideal stderr for a shot allocation, from exact per-group probability vectors."""
        var = 0.0
        outcomes = np.arange(2**self.num_qubits, dtype=np.int64)
        for g, (p, n) in enumerate(zip(probs, shots)):
            f = self._group_values(g, outcomes)
            var += (p @ f**2 - (p @ f) ** 2) / n
        return float(np.sqrt(max(var, 0.0)))

    def allocate_shots(self, total_shots: int, method: str = "uniform") -> list[int]:
        """Split a total shot budget over groups: 'uniform' or 'weighted' (proportional to the
        group's coefficient L1 norm sum_j |h_j|). Every group gets >= 1 shot."""
        k = self.num_groups
        if method == "uniform":
            w = np.ones(k)
        elif method == "weighted":
            w = np.array([np.abs(g.coeffs).sum() for g in self.groups])
        else:
            raise ValueError(f"unknown allocation {method!r}")
        alloc = np.maximum(1, np.floor(total_shots * w / w.sum())).astype(int)
        # hand leftover shots to the largest fractional remainders
        rem = total_shots - alloc.sum()
        if rem > 0:
            frac = total_shots * w / w.sum() - np.floor(total_shots * w / w.sum())
            alloc[np.argsort(-frac)[:rem]] += 1
        return [int(a) for a in alloc]


@dataclass
class EnergyEstimate:
    energy: float  # total energy, Hartree
    stderr: float
    num_groups: int
    total_shots: int
    shots_per_group: list[int]
    group_means: list[float]
    group_stderrs: list[float]
    backend: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def packed_width(plan: MeasurementPlan) -> int:
    """Qubits of plan.packed_circuit: one copy of the n-qubit state per measurement group."""
    return plan.num_qubits * plan.num_groups


def _outcome_int(key, num_bits: int | None = None) -> int:
    """Outcome key -> int with bit q = qubit q (formats: see module docstring)."""
    if isinstance(key, (int, np.integer)):
        b = int(key)
    else:
        k = str(key).replace(" ", "").replace("_", "").lower()
        b = int(k, 16) if k.startswith("0x") else int(k[2:] if k.startswith("0b") else k, 2)
    if b < 0 or (num_bits is not None and b >> num_bits):
        raise ValueError(f"outcome {key!r} does not fit {num_bits} measured qubits (register order/width mismatch)")
    return b


def compile_state(state: QuantumCircuit, optimization_level: int = 1) -> QuantumCircuit:
    """Decompose PauliEvolutionGates into the shared CX basis (all-to-all, like trapped ions)."""
    return transpile(state, basis_gates=BASIS, optimization_level=optimization_level, seed_transpiler=0)


def exact_probabilities(circuits: list[QuantumCircuit]) -> list[np.ndarray]:
    """Noise-free outcome probabilities (index bit q = qubit q) of measurement circuits."""
    return [Statevector(qc.remove_final_measurements(inplace=False)).probabilities() for qc in circuits]


def trapped_ion_noise_model(p1: float = None, p2: float = None, readout: float = None):
    """GenQAI's simple depolarizing approximation of a trapped-ion device (NOT IonQ's model)."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    p1 = TRAPPED_ION_NOISE["p1"] if p1 is None else p1
    p2 = TRAPPED_ION_NOISE["p2"] if p2 is None else p2
    ro = TRAPPED_ION_NOISE["readout"] if readout is None else readout
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def _sample(sampler, circuits, shots) -> list[dict]:
    res = sampler.run([(qc, None, n) for qc, n in zip(circuits, shots)]).result()
    return [r.data.meas.get_counts() for r in res]


# ---------------------------------------------------------------- IonQ

def _load_dotenv() -> None:
    """Load IONQ_API_KEY from the nearest .env (cwd upward, then this package upward) without
    overriding variables already set. The key is never printed."""
    from pathlib import Path

    from dotenv import find_dotenv, load_dotenv

    path = find_dotenv(usecwd=True)
    if not path:
        for d in Path(__file__).resolve().parents:
            if (d / ".env").is_file():
                path = str(d / ".env")
                break
    if path:
        load_dotenv(path, override=False)


def ionq_available() -> bool:
    _load_dotenv()
    return any(os.getenv(k) for k in ("IONQ_API_KEY", "QISKIT_IONQ_API_TOKEN", "IONQ_API_TOKEN"))


def ionq_backend(name: str = "simulator", gateset: str = "qis", token: str | None = None, **options):
    """qiskit-ionq backend. Token defaults to IONQ_API_KEY from the environment."""
    from qiskit_ionq import IonQProvider

    if token is None:
        _load_dotenv()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # IonQTranspileLevelWarning on construction
        return IonQProvider(token=token).get_backend(name, gateset=gateset, **options)


def submit_ionq(circuits: list[QuantumCircuit], shots: int, noise_model: str = "ideal", backend=None):
    """Submit ALL measurement circuits as ONE IonQ simulator job (backend.run(list)).

    noise_model is set as a BACKEND option, not only a run option: qiskit-ionq decides from
    backend.options.noise_model whether to re-sample the returned histogram client-side, and a
    run-only option would make it re-sample noisy results (double sampling).
    For "ideal", `shots` only sizes that client-side re-sampler (the server returns probabilities);
    qiskit-ionq keeps only probability bins that were hit, so it must be large (IONQ_IDEAL_SAMPLER_SHOTS).
    """
    backend = configure_ionq(backend or ionq_backend(), noise_model, shots)
    tqc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=0)
    return backend.run(tqc)


def configure_ionq(backend, noise_model: str, shots: int):
    """Set noise model and shots as BACKEND options (see submit_ionq for why). Returns backend."""
    backend.set_options(noise_model=noise_model, shots=shots)
    return backend


def collect_ionq(result, num_circuits: int, exact: bool) -> list[dict]:
    """Per-circuit distributions from an IonQ result: probabilities if exact, else counts."""
    if not exact:
        return result_counts(result, num_circuits)
    return [dict(result.get_probabilities(i)) for i in range(num_circuits)]


def result_counts(result, num_circuits: int) -> list[dict]:
    """Per-circuit counts of a multi-circuit job, whatever the provider.

    Qiskit BasicSimulator / Aer / qiskit-ionq answer result.get_counts(i); some providers only
    return a list from result.get_counts() (or a single dict for a one-circuit job).
    """
    try:
        return [dict(result.get_counts(i)) for i in range(num_circuits)]
    except Exception as first:
        allc = result.get_counts()
        allc = allc if isinstance(allc, list) else [allc]
        if len(allc) != num_circuits:
            raise ValueError(f"result has {len(allc)} count dicts for {num_circuits} circuits "
                             f"(get_counts(i) failed: {type(first).__name__}: {first})") from first
        return [dict(c) for c in allc]


def _backend_name(backend) -> str:
    name = getattr(backend, "name", type(backend).__name__)
    return str(name() if callable(name) else name)  # BackendV1: method, BackendV2: property


def is_ionq(backend) -> bool:
    """True for qiskit-ionq backends (and anything named like one, e.g. Qollab's injected IonQ)."""
    return "ionq" in (_backend_name(backend) + type(backend).__module__).lower()


def _backend_qubits(backend) -> int | None:
    n = getattr(backend, "num_qubits", None)
    if n is None and hasattr(backend, "configuration"):
        n = getattr(backend.configuration(), "n_qubits", None)
    return n


def measure_on_backend(record: HamiltonianRecord, ops, angles, backend, shots_per_group: int,
                       plan: MeasurementPlan | None = None, packed: bool | None = None) -> EnergyEstimate:
    """Measured energy on a ready-made Qiskit backend object (e.g. the one Qollab injects), in ONE job.

    packed=False: one backend.run(list_of_group_circuits). packed=True: one circuit holding a copy of
    the state per group (MeasurementPlan.packed_circuit) -- Qollab's IonQ gateway accepts only one
    single-circuit job per run. Default: packed for IonQ backends. Backend options are left untouched.
    Every group gets shots_per_group samples either way; meta["executions"] counts circuit executions
    (shots_per_group packed, num_groups * shots_per_group otherwise).
    """
    plan = plan or MeasurementPlan.from_record(record)
    name = _backend_name(backend)
    if packed is None:
        packed = is_ionq(backend)
    width, cap = packed_width(plan) if packed else plan.num_qubits, _backend_qubits(backend)
    if cap is not None and width > cap:
        raise ValueError(f"{plan.num_groups} measurement groups x {plan.num_qubits} qubits = {width} qubits in one "
                         f"circuit, but {name} has {cap}. Use h2 on IonQ, or a local simulator for this system."
                         if packed else f"{width}-qubit circuit, but {name} has {cap} qubits.")
    circuits = plan.measurement_circuits(compile_state(build_circuit(record, ops, angles)), packed)
    tqc = transpile(circuits, backend=backend, optimization_level=1, seed_transpiler=0)
    job = backend.run(tqc[0] if packed else tqc, shots=shots_per_group)
    est = plan.estimate(plan.distributions(result_counts(job.result(), len(circuits)), packed))
    est.backend = name
    est.meta.update(packed=packed, width=width, executions=shots_per_group * len(circuits))
    try:
        est.meta["job_id"] = job.job_id()
    except Exception:
        pass
    est.meta["label"] = est.backend
    return est


def ionq_native_stats(qc: QuantumCircuit, device: str = "aria-1") -> dict:
    """Local qiskit-ionq transpilation to IonQ native gates (no API call).

    Aria-family -> GPI/GPI2 + MS, Forte-family -> GPI/GPI2 + ZZ. This is qiskit-ionq's
    client-side native compilation; IonQ's server compiler may produce different counts.
    """
    b = ionq_backend(f"qpu.{device}", gateset="native", token="offline-native-transpile")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = transpile(qc, backend=b, optimization_level=1, seed_transpiler=0)
    ops = t.count_ops()
    two_q = sum(int(ops.get(g, 0)) for g in ("ms", "zz"))
    return {
        "device": device,
        "native_2q_gate": "zz" if "zz" in ops else "ms",
        "native_2q_count": two_q,
        "native_1q_count": int(ops.get("gpi", 0) + ops.get("gpi2", 0)),
        "native_depth": int(t.depth()),
    }


# ---------------------------------------------------------------- driver

def measure_energy(
    record: HamiltonianRecord,
    ops,
    angles,
    backend: str = "local_shots",
    shots: int = 10_000,
    allocation: str = "uniform",
    seed: int | None = None,
    noise_model: str = "aria-1",
    plan: MeasurementPlan | None = None,
) -> EnergyEstimate:
    """Energy of HF + prod exp(-i angle G_op) on one backend.

    shots is the TOTAL budget over all groups (ignored by exact/exact_probs/ionq_ideal).
    noise_model is the IonQ simulator noise model for ionq_noisy only.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == "exact":
        from genqai.sim import Evaluator

        e = Evaluator(record).energy(ops, angles)
        return EnergyEstimate(e, 0.0, 0, 0, [], [], [], backend=backend)

    plan = plan or MeasurementPlan.from_record(record)
    circuits = plan.circuits(compile_state(build_circuit(record, ops, angles)))
    alloc = plan.allocate_shots(shots, allocation)
    meta: dict = {"allocation": allocation}

    if backend == "exact_probs":
        probs = exact_probabilities(circuits)
        est = plan.estimate([dict(enumerate(p)) for p in probs], exact=True)
        meta["predicted_stderr_at_shots"] = {"shots": int(sum(alloc)), "stderr": plan.predicted_stderr(probs, alloc)}
    elif backend == "local_shots":
        from qiskit.primitives import StatevectorSampler

        est = plan.estimate(_sample(StatevectorSampler(seed=seed), circuits, alloc))
    elif backend == "local_noisy":
        from qiskit_aer.primitives import SamplerV2

        sampler = SamplerV2(seed=seed, options={"backend_options": {"noise_model": trapped_ion_noise_model()}})
        est = plan.estimate(_sample(sampler, circuits, alloc))
        meta["noise_model"] = {"name": "genqai_trapped_ion_depolarizing (OUR approximation, not IonQ)", **TRAPPED_ION_NOISE}
    else:  # IonQ simulator
        if not ionq_available():
            raise RuntimeError("IonQ backends need IONQ_API_KEY in the environment")
        exact = backend == "ionq_ideal"
        if not exact and allocation != "uniform":
            raise ValueError("ionq_noisy submits one job with one shot count: use allocation='uniform'")
        per_group = max(1, shots // plan.num_groups)
        nm = "ideal" if exact else noise_model
        job = submit_ionq(circuits, IONQ_IDEAL_SAMPLER_SHOTS if exact else per_group, nm)
        est = plan.estimate(collect_ionq(job.result(), plan.num_groups, exact), exact=exact)
        meta.update(ionq_job_id=job.job_id(), ionq_noise_model=nm)
        if exact:
            # worst-case energy error from IonQ's 1e-3 probability rounding: sum_g max|f_g| * 2^n * res/2
            bound = sum(np.abs(np.real(g.coeffs)).sum() for g in plan.groups) * 2**plan.num_qubits * IONQ_PROB_RESOLUTION / 2
            meta.update(note="IonQ ideal sim: exact probabilities rounded to 1e-3, stderr=0",
                        probability_rounding_bound_ha=float(bound))
    est.backend = backend
    est.meta = meta
    return est


def default_backends() -> list[str]:
    """Local backends always; IonQ simulator backends only when an API key is present."""
    return ["exact", "exact_probs", "local_shots", "local_noisy"] + (
        ["ionq_ideal", "ionq_noisy"] if ionq_available() else [])


def run_backends(
    record: HamiltonianRecord,
    ops,
    angles,
    backends=None,
    shots: int = 10_000,
    allocation: str = "uniform",
    ionq_noise_models=("aria-1",),
    seed: int | None = 0,
    plan: MeasurementPlan | None = None,
) -> list[EnergyEstimate]:
    """measure_energy on every backend (ionq_noisy once per IonQ noise model).

    Each estimate gets meta["label"] ("ionq_noisy/aria-1", ...). A failing backend is reported and
    skipped so it never loses the others' results. IonQ backends always use uniform allocation.
    """
    plan = plan or MeasurementPlan.from_record(record)
    runs = []
    for b in backends or default_backends():
        for nm in ionq_noise_models if b == "ionq_noisy" else [None]:
            label = f"{b}/{nm}" if nm else b
            try:
                e = measure_energy(record, ops, angles, b, shots=shots,
                                   allocation="uniform" if b.startswith("ionq") else allocation,
                                   seed=seed, noise_model=nm or "ideal", plan=plan)
            except Exception as exc:
                print(f"[{label}] failed: {type(exc).__name__}: {exc}")
                continue
            e.meta["label"] = label
            runs.append(e)
    return runs


def runs_table(record: HamiltonianRecord, runs: list[EnergyEstimate]) -> str:
    """Plain-text table of measured energies vs FCI (IonQ job IDs included)."""
    lines = [f"{'backend':<20}{'energy (Ha)':>14}{'stderr':>11}{'err mHa':>10}{'groups':>8}{'shots':>9}"]
    for e in runs:
        lines.append(
            f"{e.meta['label']:<20}{e.energy:>14.6f}{e.stderr:>11.6f}{1000 * (e.energy - record.fci_energy):>10.3f}"
            f"{e.num_groups:>8}{e.total_shots:>9}" + (f"  job={e.meta['ionq_job_id']}" if "ionq_job_id" in e.meta else "")
            + (f"  (1e-3 prob rounding: |bias| <= {1000 * e.meta['probability_rounding_bound_ha']:.1f} mHa)"
               if "probability_rounding_bound_ha" in e.meta else ""))
    lines.append("local_noisy = GenQAI depolarizing approximation (NOT IonQ's model); "
                 "ionq_noisy = IonQ's simulator noise model")
    return "\n".join(lines)
