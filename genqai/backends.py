"""Execution backends the workbench can offer, with availability, width limits and cost flags.

| name                     | kind       | what it is                                                   |
| statevector              | exact      | exact statevector probabilities, no shots                    |
| local_shots              | local      | Aer ideal sampler (BasicSimulator if Aer is missing)         |
| local_noisy              | local      | Aer + measure.trapped_ion_noise_model(): GenQAI's OWN        |
|                          |            | depolarizing approximation, NOT IonQ's noise model           |
| ionq_simulator           | ionq_sim   | IonQ cloud simulator, ideal (needs IONQ_API_KEY, qiskit-ionq)|
| ionq_simulator_forte-1   | ionq_sim   | IonQ cloud simulator with IonQ's Forte 1 noise model         |
| ionq_simulator_aria-1    | ionq_sim   | IonQ cloud simulator with IonQ's Aria 1 noise model          |
| ionq_qpu_forte-1/aria-1  | ionq_qpu   | hardware: paid, listed for completeness, NEVER available     |
| injected                 | injected   | a backend object handed in (e.g. Qollab's global `backend`)  |

Paid backends are never available here and genqai.execution refuses them.
"""
from __future__ import annotations

from dataclasses import dataclass

from genqai.measure import TRAPPED_ION_NOISE, _backend_name, _backend_qubits, ionq_available, is_ionq

IONQ_SIM_MAX_QUBITS = 29  # IonQ cloud simulator (qiskit-ionq ionq_simulator.num_qubits)
# GenQAI's own cap for local statevector-based simulation: 2^24 amplitudes (~256 MB complex128).
LOCAL_MAX_QUBITS = 24
IONQ_SIM_NOISE_MODELS = ("forte-1", "aria-1")
IONQ_QPUS = ("forte-1", "aria-1")


@dataclass
class BackendInfo:
    name: str
    kind: str  # "exact" | "local" | "ionq_sim" | "ionq_qpu" | "injected"
    available: bool
    reason: str
    max_qubits: int | None
    needs_key: bool
    paid: bool  # paid backends are never listed as available
    noise_model: str | None = None  # None = ideal
    packed_default: bool = False  # one single-circuit job holding every measurement group
    description: str = ""


def _ionq_status() -> tuple[bool, str]:
    try:
        import qiskit_ionq  # noqa: F401
    except ImportError:
        return False, "qiskit-ionq is not installed"
    if not ionq_available():
        return False, "no IONQ_API_KEY in the environment or .env"
    return True, "IonQ API key found (simulator only)"


def _aer_status() -> tuple[bool, str]:
    try:
        import qiskit_aer  # noqa: F401
    except ImportError:
        return False, "qiskit-aer is not installed"
    return True, "qiskit-aer installed"


def injected_info(backend) -> BackendInfo:
    """BackendInfo for a backend object handed in. IonQ hardware ("qpu" in the name) counts as paid."""
    name = _backend_name(backend)
    ionq = is_ionq(backend)
    paid = ionq and "qpu" in name.lower()
    nm = getattr(getattr(backend, "options", None), "noise_model", None)
    return BackendInfo(
        name="injected", kind="injected", available=not paid,
        reason=f"injected backend {name}" + (" is IonQ hardware (paid): refused" if paid else ""),
        max_qubits=_backend_qubits(backend), needs_key=False, paid=paid,
        noise_model=None if nm in (None, "ideal") else str(nm), packed_default=ionq,
        description=f"backend object {name} ({type(backend).__module__})")


def list_backends(injected=None) -> list[BackendInfo]:
    """Every backend with availability in THIS environment. `injected`: optional backend object."""
    aer_ok, aer_reason = _aer_status()
    ionq_ok, ionq_reason = _ionq_status()
    out = [
        BackendInfo("statevector", "exact", True, "always available", LOCAL_MAX_QUBITS, False, False,
                    description="exact statevector probabilities through the grouping estimator; no shots"),
        BackendInfo("local_shots", "local", True, aer_reason if aer_ok else "Aer missing: BasicSimulator fallback",
                    LOCAL_MAX_QUBITS, False, False, description="ideal finite-shot sampling (Aer, seeded)"),
        BackendInfo("local_noisy", "local", aer_ok, aer_reason, LOCAL_MAX_QUBITS, False, False,
                    noise_model="genqai_trapped_ion_depolarizing",
                    description="GenQAI approximation, NOT IonQ's model: depolarizing p1={p1}, p2={p2}, "
                                "readout={readout}".format(**TRAPPED_ION_NOISE)),
        BackendInfo("ionq_simulator", "ionq_sim", ionq_ok, ionq_reason, IONQ_SIM_MAX_QUBITS, True, False,
                    packed_default=True, description="IonQ cloud simulator, ideal; free"),
    ]
    out += [BackendInfo(f"ionq_simulator_{nm}", "ionq_sim", ionq_ok, ionq_reason, IONQ_SIM_MAX_QUBITS, True, False,
                        noise_model=nm, packed_default=True,
                        description=f"IonQ cloud simulator with IonQ's {nm} noise model; free")
            for nm in IONQ_SIM_NOISE_MODELS]
    out += [BackendInfo(f"ionq_qpu_{d}", "ionq_qpu", False, "IonQ hardware costs credits: never used by GenQAI",
                        None, True, True, packed_default=True, description=f"IonQ {d} QPU (paid)")
            for d in IONQ_QPUS]
    out.append(injected_info(injected) if injected is not None else BackendInfo(
        "injected", "injected", False, "no backend object handed in", None, False, False))
    return out


def get_backend_info(name: str, injected=None) -> BackendInfo:
    for b in list_backends(injected):
        if b.name == name:
            return b
    raise ValueError(f"unknown backend {name!r}; known: {[b.name for b in list_backends()]}")


def make_backend(info: BackendInfo, seed: int | None = None, shots: int | None = None, offline: bool = False):
    """A runnable Qiskit backend for a named (non-exact, non-injected) backend.

    offline=True builds the IonQ backend with a placeholder token (local transpilation for dry runs
    only; nothing can be submitted with it).
    """
    if info.paid:
        raise PermissionError(f"{info.name} is a paid backend: refused")
    if info.kind == "local":
        try:
            from qiskit_aer import AerSimulator
        except ImportError:
            if info.noise_model:
                raise
            from qiskit.providers.basic_provider import BasicSimulator

            return BasicSimulator()
        from genqai.measure import trapped_ion_noise_model

        kw = {"noise_model": trapped_ion_noise_model()} if info.noise_model else {}
        return AerSimulator(seed_simulator=seed, **kw)
    if info.kind == "ionq_sim":
        from genqai.measure import configure_ionq, ionq_backend

        b = ionq_backend("simulator", token="offline-dry-run" if offline else None)
        return configure_ionq(b, info.noise_model or "ideal", shots or 1024)
    raise ValueError(f"{info.name} ({info.kind}) has no constructible backend")
