"""Lane C: hardware-aware circuit optimisation with before/after grading (docs/workbench_contracts.md).

    optimize(record, circuit, objective, ...) -> OptimizationResult

Strategies (all on the SAME Hamiltonian, energies from ONE genqai.sim.Evaluator whose counters give
the evaluation cost):
  simplify   exact, 0 evaluations: merge consecutive identical operators
             (exp(-i a G) exp(-i b G) = exp(-i (a+b) G)) and drop exactly-zero angles, to a fixed point.
  refine     BFGS over all angles (Evaluator.optimize), warm-started from the current angles.
  prune      greedy: screen every single-operator removal (1 energy evaluation each), re-refine the
             `beam` most promising, accept the acceptable one whose removal + re-refinement costs least
             energy (selection="energy") or least energy per compiled CX saved ("energy_per_cx"), repeat. Acceptable = |E - E_FCI| <= target (default; needs an exact
             reference), or, with reference_free_tol_mha, E - E_start <= tol (no reference used).
  compile    Qiskit transpile at optimization level 3 for compiled counts (reported alongside the
             level-1 counts every other number in this repo uses).
Objectives: accuracy (simplify + refine), min_cost (simplify only, 0 evaluations), min_2q / min_depth
(simplify + refine + prune, then the accepted circuit with the fewest CX / lowest compiled depth),
noise (the accepted circuit with the smallest local-noisy energy error; GenQAI's depolarising model,
NOT IonQ's), pareto (the accepted circuits' error/CX/depth frontier).
Pruning and refinement spend energy evaluations, so their results belong to a different cost class
than one-shot generation (0 evaluations to propose).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from genqai.adapt import wrap_angle
from genqai.circuits import build_circuit, resource_stats
from genqai.grading import GradeReport, grade, pareto_front, reference_energy
from genqai.measure import measure_energy
from genqai.schema import GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator

OBJECTIVES = ("accuracy", "min_2q", "min_depth", "min_cost", "noise", "pareto")
PRUNING = ("min_2q", "min_depth", "noise", "pareto")
SELECTIONS = ("energy", "energy_per_cx")


@dataclass
class OptimizationResult:
    objective: str
    original: GeneratedCircuit
    optimized: GeneratedCircuit
    before: GradeReport
    after: GradeReport
    steps: list[dict]
    evals_used: int  # Evaluator energy evaluations (incl. energy+gradient calls) spent optimising
    gradient_evals: int
    same_hamiltonian_check: dict
    compiled_level3: dict  # resource_stats at optimization level 3, original and optimized
    candidates: dict[str, GeneratedCircuit] = field(default_factory=dict)  # every accepted pruning step
    frontier: list[str] | None = None  # objective "pareto": non-dominated candidate ids


def record_hash(record: HamiltonianRecord) -> str:
    """sha256 of everything that defines the qubit problem and its references."""
    key = {k: getattr(record, k) for k in ("geometry_id", "system", "num_qubits", "pauli_terms", "energy_shift",
                                            "hf_state", "hf_energy", "ccsd_energy", "fci_energy")}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def simplify(ops, angles) -> tuple[list[int], list[float], dict]:
    """Exact simplification to a fixed point: merge consecutive identical operators, drop zero angles."""
    ops, angles = list(ops), [float(a) for a in angles]
    merged = dropped = 0
    changed = True
    while changed:
        changed = False
        o2, a2 = [], []
        for k, t in zip(ops, angles):
            if o2 and o2[-1] == k:
                a2[-1] += t
                merged += 1
                changed = True
            else:
                o2.append(k)
                a2.append(t)
        keep = [i for i, t in enumerate(a2) if t != 0.0]
        if len(keep) < len(o2):
            dropped += len(o2) - len(keep)
            changed = True
        ops, angles = [o2[i] for i in keep], [a2[i] for i in keep]
    return ops, angles, {"merged": merged, "dropped_zero": dropped}


def _cx(record, ops, angles, level: int = 1) -> dict:
    return resource_stats(build_circuit(record, ops, angles), level)


def optimize(record: HamiltonianRecord, circuit: GeneratedCircuit, objective: str, *, target_mha: float = 1.6,
             max_evals: int = 500, seed=0, reference_free_tol_mha: float | None = None, beam: int = 3,
             selection: str = "energy", noise_shots: int = 10_000, gen_log=None,
             grade_kw: dict | None = None) -> OptimizationResult:
    """Optimise one circuit for one objective; always returns original + optimized + both grades.

    selection: which acceptable removal pruning takes: "energy" (lowest energy after removal +
        re-refinement) or "energy_per_cx" (lowest energy increase per compiled CX saved).

    max_evals: soft cap on the optimiser's energy evaluations (pruning stops once it is exceeded; the
        refinement in progress is completed). reference_free_tol_mha: prune without the exact reference,
        accepting removals while the energy stays within tol of the refined starting energy.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    gkw = {"target_mha": target_mha, **(grade_kw or {})}
    h0 = record_hash(record)
    before = grade(record, circuit, gen_log=gen_log, seed=seed, **gkw)
    ev = Evaluator(record)
    steps: list[dict] = []
    ops, angles = list(circuit.operators), [float(a) for a in circuit.angles]
    candidates: dict[str, GeneratedCircuit] = {}

    def make(o, a, tag):
        return GeneratedCircuit(record.geometry_id, record.system, list(o), [wrap_angle(t) for t in a], True,
                                source=f"{circuit.source}+opt:{objective}", meta={"step": tag})

    if before.status != "invalid":
        ops, angles, info = simplify(ops, angles)
        e = ev.energy(ops, angles)
        steps.append({"action": "simplify", **info, "n_ops": len(ops), "energy": e, "evals": ev.energy_evals})
        if objective != "min_cost":
            x, e_ref, _ = ev.optimize(ops, angles)
            if e_ref <= e:
                angles, e = [float(t) for t in x], e_ref
            steps.append({"action": "refine", "n_ops": len(ops), "energy": e, "evals": ev.energy_evals})
        candidates["start"] = make(ops, angles, "start")
        if objective in PRUNING:
            ops, angles = _prune(record, ev, ops, angles, e, target_mha, max_evals, reference_free_tol_mha, beam,
                                 selection, steps, candidates, make)
    evals_used, grad_evals = ev.energy_evals, ev.gradient_evals

    frontier = None
    if objective in PRUNING and before.status != "invalid":
        pick, frontier = _select(record, objective, candidates, noise_shots, seed, gkw, steps)
        optimized = candidates[pick]
    else:
        optimized = make(ops, angles, "final") if before.status != "invalid" else circuit
    optimized.meta.update(objective=objective, evals_used=evals_used, target_mha=target_mha,
                          reference_free_tol_mha=reference_free_tol_mha)
    after_log = dict(vars(gen_log) if gen_log is not None and not isinstance(gen_log, dict) else (gen_log or {}))
    after_log["refine_energy_evals"] = int(after_log.get("refine_energy_evals") or 0) + evals_used
    after = grade(record, optimized, gen_log=after_log, seed=seed, **gkw)

    # same-Hamiltonian check: record untouched, both energies recomputed with the optimiser's Evaluator
    e_orig = ev.energy(circuit.operators, circuit.angles) if before.status != "invalid" else None
    e_opt = ev.energy(optimized.operators, optimized.angles) if before.status != "invalid" else None
    same = {"record_hash_before": h0, "record_hash_after": record_hash(record), "evaluator": "one shared Evaluator",
            "energy_original": e_orig, "energy_optimized": e_opt}
    same["ok"] = h0 == same["record_hash_after"] and (e_orig is None or (
        abs(e_orig - before.energy["energy"]) < 1e-10 and abs(e_opt - after.energy["energy"]) < 1e-10))
    l3 = ({"original": _cx(record, circuit.operators, circuit.angles, 3),
           "optimized": _cx(record, optimized.operators, optimized.angles, 3)} if before.status != "invalid" else {})
    return OptimizationResult(objective, circuit, optimized, before, after, steps, evals_used, grad_evals, same, l3,
                              candidates, frontier)


def _prune(record, ev, ops, angles, e_start, target_mha, max_evals, tol, beam, selection, steps, candidates, make):
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    ref, kind, _ = reference_energy(record)
    if tol is None and kind != "FCI":
        raise ValueError("pruning to a target needs an exact (FCI) reference; pass reference_free_tol_mha instead")

    def acceptable(e):
        return 1000 * (e - e_start) <= tol if tol is not None else abs(1000 * (e - ref)) <= target_mha

    if not acceptable(e_start):
        steps.append({"action": "prune", "stopped": "starting circuit not acceptable", "evals": ev.energy_evals})
        return ops, angles
    e_cur = e_start
    while ops and ev.energy_evals < max_evals:
        cx_cur = _cx(record, ops, angles)["cx_qiskit"] if selection == "energy_per_cx" else 0

        def cost(e, o, a):  # energy (selection "energy") or energy increase per compiled CX saved
            if selection == "energy":
                return e
            return (e - e_cur) / max(cx_cur - _cx(record, o, a)["cx_qiskit"], 1)

        trials = []
        for i in range(len(ops)):
            o, a, _ = simplify(ops[:i] + ops[i + 1:], angles[:i] + angles[i + 1:])
            e0 = ev.energy(o, a)
            trials.append((cost(e0, o, a), e0, i, o, a))
        best = None
        for _, e0, i, o, a in sorted(trials, key=lambda t: t[0])[:beam]:
            if o:
                x, e_r, _ = ev.optimize(o, a)
                if e_r <= e0:
                    e0, a = e_r, [float(t) for t in x]
            c = cost(e0, o, a)
            if acceptable(e0) and (best is None or c < best[0]):
                best = (c, e0, i, o, a)
        if best is None:
            steps.append({"action": "prune", "stopped": "no acceptable removal among the beam",
                          "evals": ev.energy_evals})
            break
        _, e_cur, i, o, a = best
        steps.append({"action": "prune", "removed_position": i, "removed_op": ops[i], "n_ops": len(o),
                      "energy": e_cur, "error_mha": None if ref is None else 1000 * (e_cur - ref),
                      "evals": ev.energy_evals})
        ops, angles = o, a
        tag = f"prune{len(candidates)}"
        candidates[tag] = make(ops, angles, tag)
    else:
        if ops:
            steps.append({"action": "prune", "stopped": f"max_evals={max_evals} reached", "evals": ev.energy_evals})
    return ops, angles


def _select(record, objective, candidates, noise_shots, seed, gkw, steps):
    """Pick the final circuit among the accepted candidates (every one already passed the acceptance rule)."""
    if objective in ("min_2q", "min_depth"):
        key = "cx_qiskit" if objective == "min_2q" else "compiled_depth"
        lvl = gkw.get("optimization_level", 1)
        scored = {k: (_cx(record, c.operators, c.angles, lvl)[key], len(c.operators)) for k, c in candidates.items()}
        return min(scored, key=scored.get), None
    if objective == "noise":
        ref, _, _ = reference_energy(record)
        errs = {}
        for k, c in candidates.items():
            est = measure_energy(record, c.operators, c.angles, "local_noisy", shots=noise_shots, seed=seed)
            errs[k] = abs(est.energy - ref)
            steps.append({"action": "noisy_eval", "candidate": k, "energy": est.energy, "stderr": est.stderr,
                          "noise_model": "GenQAI trapped-ion depolarising approximation (NOT IonQ)",
                          "shots": est.total_shots})
        return min(errs, key=errs.get), None
    reports = {k: grade(record, c, **{**gkw, "ionq_native": False}) for k, c in candidates.items()}
    front = pareto_front(reports, ("error_mha", "cx_qiskit", "compiled_depth"))
    return min(front, key=lambda k: reports[k].resources["cx_qiskit"]) if front else "start", front
