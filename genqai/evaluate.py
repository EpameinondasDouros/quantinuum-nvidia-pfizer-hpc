"""Workstream D: the single shared evaluation harness. Every method is scored through here.

    score_circuit(record, circuit, method, ...) -> EvalRecord
    summarize(evals) -> {method: {split: stats}}

Scoring energy evaluations are NOT charged to the method; only the costs the method reports
(passed as keyword arguments) plus, with refine=True, the angle re-optimization are counted.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from genqai.circuits import build_circuit, gate_stats
from genqai.schema import EvalRecord, GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator

COST_FIELDS = ("pool_gradient_evals", "energy_evals", "optimizer_iterations", "model_forward_passes", "runtime_s")
THRESHOLDS_MHA = (1.6, 5.0, 10.0)


def score_circuit(
    record: HamiltonianRecord,
    circuit: GeneratedCircuit,
    method: str,
    seed: int = 0,
    refine: bool = False,
    converged: bool | None = None,
    extra: dict | None = None,
    **cost,
) -> EvalRecord:
    """Exact energy, error vs FCI, compiled CX count/depth, and cost counters for one circuit.

    cost: any of COST_FIELDS, the method's own generation cost (e.g. model_forward_passes=12).
    refine: re-optimize all angles with BFGS starting from circuit.angles; the refinement's
        energy evaluations, iterations and wall time are ADDED to the reported cost and
        `converged` becomes the optimizer's success flag.
    An invalid circuit (circuit.valid False) gets energy/error NaN and converged False.
    """
    unknown = set(cost) - set(COST_FIELDS)
    if unknown:
        raise TypeError(f"unknown cost fields {sorted(unknown)}; expected {COST_FIELDS}")
    if circuit.geometry_id != record.geometry_id:
        raise ValueError(f"circuit {circuit.geometry_id} does not belong to record {record.geometry_id}")
    counters = {k: cost.get(k, 0) for k in COST_FIELDS}
    extra = dict(extra or {})
    ops, angles = list(circuit.operators), np.asarray(circuit.angles, float)
    common = dict(method=method, system=record.system, geometry_id=record.geometry_id, split=record.split, seed=seed)
    if not circuit.valid:
        return EvalRecord(**common, energy=math.nan, error_mha=math.nan, num_operators=len(ops),
                          converged=False, extra={**extra, "invalid": True}, **counters)
    ev = Evaluator(record)
    if refine:
        t0 = time.perf_counter()
        angles, _, res = ev.optimize(ops, angles)
        counters["energy_evals"] += ev.energy_evals
        counters["optimizer_iterations"] += res.nit if res is not None else 0
        counters["runtime_s"] += time.perf_counter() - t0
        converged = bool(res.success) if res is not None else True
        extra["refined_angles"] = [float(a) for a in angles]
    energy = ev.energy(ops, angles)
    stats = gate_stats(build_circuit(record, ops, angles))
    return EvalRecord(
        **common,
        energy=energy,
        error_mha=1000.0 * (energy - record.fci_energy),
        num_operators=len(ops),
        cnot_count=stats["cnot_count"],
        depth=stats["depth"],
        converged=converged,
        extra=extra,
        **counters,
    )


def _stats(evals: list[EvalRecord]) -> dict:
    err = np.array([abs(e.error_mha) for e in evals], float)
    ok = ~np.isnan(err)
    v = err[ok]
    cx = [e.cnot_count for e in evals if e.cnot_count is not None]
    out = {
        "n": len(evals),
        "valid_frac": float(ok.mean()),
        "median_abs_err_mha": float(np.median(v)) if v.size else math.nan,
        "mean_abs_err_mha": float(v.mean()) if v.size else math.nan,
        "max_abs_err_mha": float(v.max()) if v.size else math.nan,
        # invalid circuits count as failures in the success fractions
        **{f"frac_under_{t:g}mha": float(np.mean(np.where(ok, err, np.inf) <= t)) for t in THRESHOLDS_MHA},
        "mean_num_operators": float(np.mean([e.num_operators for e in evals])),
        "mean_cnot_count": float(np.mean(cx)) if cx else math.nan,
        "mean_depth": float(np.mean([e.depth for e in evals if e.depth is not None])) if cx else math.nan,
    }
    for k in COST_FIELDS:
        out[f"mean_{k}"] = float(np.mean([getattr(e, k) for e in evals]))
    conv = [e.converged for e in evals if e.converged is not None]
    out["converged_frac"] = float(np.mean(conv)) if conv else math.nan
    return out


def summarize(evals: list[EvalRecord]) -> dict[str, dict[str, dict]]:
    """{method: {split: stats}} with extra pseudo-splits "test" (all test_*) and "all"."""
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in evals:
        g = groups[e.method]
        g[e.split].append(e)
        g["all"].append(e)
        if e.split.startswith("test"):
            g["test"].append(e)
    return {m: {s: _stats(v) for s, v in g.items()} for m, g in groups.items()}
