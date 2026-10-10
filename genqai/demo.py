"""Judge-facing demo: one unseen geometry, end to end (notebooks/genqai_demo.ipynb, scripts/demo.py).

    record, how = get_record("lih4", (2.47,))          # live PySCF, or stored Hamiltonian fallback
    gens = generate_all(record)                          # {checkpoint name: (GeneratedCircuit, info)}
    evals = compare(record, gens, live_adapt=True)       # generator vs HF / CCSD / NN / fresh ADAPT
    print(comparison_table(record, evals))

Everything is computed live. Two degraded modes, both labelled in the output:
* no PySCF: off-grid geometries snap to the nearest stored geometry in data/<system>/hamiltonians.jsonl;
* no torch: generation is replaced by a REPLAY of results/<system>/gen_oneshot.jsonl (grid geometries only).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from genqai.adapt import run_adapt
from genqai.baselines import run_method
from genqai.chemistry import SYSTEMS, build_record, geometry_id
from genqai.data import load_hamiltonians, load_teacher
from genqai.evaluate import score_circuit
from genqai.schema import EvalRecord, GeneratedCircuit, HamiltonianRecord, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
CKPT_ORDER = ("sft", "grpo", "distill")  # known stages first; any other checkpoints/<system>/*.pt after
CHEM_ACC_MHA = 1.6

FIGURE_CAPTIONS = {
    "fig1_pes": "Potential-energy curves (top) and error vs FCI (bottom) of the generated circuits against HF/CCSD/ADAPT.",
    "fig2_test_errors": "Per-method |E - E_FCI| on the held-out test geometries (interp, interval, extrap).",
    "fig3_accuracy_cost": "Median test error vs circuit size (CX) and vs quantum evaluations per geometry: one-shot = 0 evals.",
    "fig4_training": "SFT training curves: validation one-shot error vs epoch.",
    "fig5_breakeven": "Amortisation: cumulative quantum evals / wall-clock vs new geometries; dots = break-even M* vs ADAPT.",
    "fig6_ionq": "Measured energies on local and IonQ cloud simulators (ideal and noisy) vs FCI.",
}


def _grid_match(system: str, params: tuple[float, ...]) -> HamiltonianRecord | None:
    gid = geometry_id(SYSTEMS[system], params)
    return next((r for r in load_hamiltonians(system) if r.geometry_id == gid), None)


def get_record(system: str, params) -> tuple[HamiltonianRecord, str]:
    """Hamiltonian + HF/CCSD/FCI for one geometry -> (record, how it was obtained).

    Stored geometries load from data/<system>/hamiltonians.jsonl (keep their split). Anything else is
    built live with PySCF (split "unseen"); without PySCF it snaps to the nearest stored geometry.
    """
    params = tuple(float(p) for p in params)
    if len(params) != len(SYSTEMS[system].geometry_params):
        raise ValueError(f"{system} takes parameters {SYSTEMS[system].geometry_params}")
    stored = _grid_match(system, params)
    if stored is not None:
        return stored, f"stored dataset geometry (split={stored.split})"
    try:
        t0 = time.perf_counter()
        rec = build_record(SYSTEMS[system], params)
        rec.split = "unseen"
        return rec, f"built live with PySCF in {time.perf_counter() - t0:.1f}s (not in any training data)"
    except ImportError:
        recs = load_hamiltonians(system)
        x = np.array(params)
        near = min(recs, key=lambda r: np.linalg.norm(np.array(list(r.geometry_params.values())) - x))
        return near, (f"PySCF NOT AVAILABLE: snapped to nearest stored geometry {near.geometry_params} "
                      f"(split={near.split})")


def discover_checkpoints(system: str) -> dict[str, Path]:
    """{stage name: path} for every checkpoints/<system>/*.pt (sft, grpo, distill, ... whichever exist)."""
    paths = {p.stem: p for p in sorted((ROOT / "checkpoints" / system).glob("*.pt"))}
    return {k: paths[k] for k in sorted(paths, key=lambda k: (CKPT_ORDER.index(k) if k in CKPT_ORDER else 99, k))}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def generate_oneshot(record: HamiltonianRecord, ckpt: Path) -> tuple[GeneratedCircuit, dict]:
    """Greedy one-shot generation: 1 model call, 0 energy evaluations. Falls back to a replay."""
    if torch_available():
        from genqai.model import GQEPolicy

        policy = GQEPolicy.load(ckpt)
        t0 = time.perf_counter()
        ops, angles, logp, valid = policy.sample(policy.condition([record]), greedy=True)[0]
        dt = time.perf_counter() - t0
        circ = GeneratedCircuit(record.geometry_id, record.system, ops, angles, valid,
                                source=f"{ckpt.stem}_oneshot", meta={"logprob": logp})
        return circ, {"mode": "live", "runtime_s": dt, "checkpoint": str(ckpt.relative_to(ROOT))}
    # replay files are what scripts/generate.py writes (--tag _<stage> for non-SFT checkpoints)
    replay = ROOT / "results" / record.system / ("gen_oneshot.jsonl" if ckpt.stem == "sft" else f"gen_oneshot_{ckpt.stem}.jsonl")
    bank = {c.geometry_id: c for c in load_jsonl(replay, GeneratedCircuit)} if replay.exists() else {}
    if record.geometry_id not in bank:
        raise RuntimeError(f"torch is not installed and {replay.name} has no circuit for {record.geometry_id}: "
                           "pick a stored grid geometry or install torch")
    circ = bank[record.geometry_id]
    circ.source = f"{ckpt.stem}_oneshot (REPLAY)"
    return circ, {"mode": "REPLAY of precomputed circuit, not live generation", "runtime_s": 0.0,
                  "checkpoint": str(replay.relative_to(ROOT))}


def generate_all(record: HamiltonianRecord) -> dict[str, tuple[GeneratedCircuit, dict]]:
    ckpts = discover_checkpoints(record.system)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints/{record.system}/*.pt")
    out = {}
    for name, p in ckpts.items():
        try:
            out[name] = generate_oneshot(record, p)
        except RuntimeError as exc:  # replay mode only: this geometry/stage has no precomputed circuit
            print(f"[{name}] skipped: {exc}")
    if not out:
        raise RuntimeError(f"no circuit could be generated or replayed for {record.geometry_id}")
    return out


def compare(record: HamiltonianRecord, gens: dict[str, tuple[GeneratedCircuit, dict]],
            live_adapt: bool = True) -> list[EvalRecord]:
    """Score generators and baselines on one geometry with the shared harness (exact statevector).

    Baselines: HF, CCSD, FCI (from the record), nearest-neighbour reuse/refine (TRAINING teacher
    labels only, genqai.baselines), and optionally a fresh ADAPT-VQE run with the teacher's settings.
    """
    evals = [run_method(m, record, [], {}) for m in ("hf", "ccsd", "fci")]
    for circ, info in gens.values():
        evals.append(score_circuit(record, circ, circ.source, model_forward_passes=1, runtime_s=info["runtime_s"]))
    train = load_hamiltonians(record.system, "train")
    teacher = load_teacher(record.system)
    bank = {r.geometry_id: teacher[r.geometry_id] for r in train}
    for m in ("nn_reuse", "nn_refine"):
        evals.append(run_method(m, record, train, bank))
    if live_adapt:
        settings = next(iter(bank.values())).meta
        own = run_adapt(record, max_ops=settings["max_ops"], grad_tol=settings["grad_tol"])
        evals.append(run_method("adapt", record, train, bank, own_teacher=own))
    return evals


def comparison_table(record: HamiltonianRecord, evals: list[EvalRecord]) -> str:
    head = (f"{'method':<26}{'energy (Ha)':>13}{'err mHa':>10}{'chem acc':>9}{'#ops':>5}{'CX':>5}{'depth':>6}"
            f"{'model calls':>12}{'energy evals':>13}{'pool grads':>11}{'time s':>8}")
    lines = [head, "-" * len(head)]
    for e in evals:
        ok = "yes" if abs(e.error_mha) <= CHEM_ACC_MHA else "no"
        cx = "-" if e.cnot_count is None else str(e.cnot_count)
        dp = "-" if e.depth is None else str(e.depth)
        lines.append(f"{e.method:<26}{e.energy:>13.6f}{e.error_mha:>10.3f}{ok:>9}{e.num_operators:>5}{cx:>5}{dp:>6}"
                     f"{e.model_forward_passes:>12}{e.energy_evals:>13}{e.pool_gradient_evals:>11}{e.runtime_s:>8.2f}")
    lines.append(f"FCI = {record.fci_energy:.6f} Ha; chemical accuracy = |err| <= {CHEM_ACC_MHA} mHa; "
                 "CX = Qiskit CX-basis count (all-to-all). Scoring itself is free (exact statevector).")
    return "\n".join(lines)


def describe_circuit(circ: GeneratedCircuit, max_lines: int = 40) -> str:
    from genqai.pool import get_pool

    pool = get_pool(circ.system)
    lines = [f"{t:>3}  op {k:>3}  {pool.describe(k):<28} theta = {a:+.5f} rad"
             for t, (k, a) in enumerate(zip(circ.operators, circ.angles))]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(circ.operators) - max_lines} more)"]
    return "\n".join(lines) + ("" if circ.valid else "\n(INVALID: no end-of-sequence token)")

