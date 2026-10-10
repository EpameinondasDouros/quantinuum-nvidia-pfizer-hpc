"""Workstream B baselines, all scored through genqai.evaluate.score_circuit.

Methods (results/<system>/<method>.jsonl):
  hf          HF reference state, 0 operators
  ccsd, fci   classical references read from the HamiltonianRecord (no circuit)
  uccsd_vqe   every pool operator once, in pool order, BFGS from zero angles
  adapt       fresh ADAPT (the teacher run in data/<system>/teacher.jsonl; costs from its meta)
  adapt_warm  ADAPT continued from the nearest training geometry's teacher circuit
  nn_reuse    nearest training geometry's teacher circuit copied verbatim, no optimization
  nn_refine   nn_reuse + BFGS angle re-optimization

Leakage rule: nearest-neighbour methods only ever see TRAINING-split teacher labels. Training
geometries use leave-one-out (their own label is excluded), so their numbers are honest too.
"""
from __future__ import annotations

import math
import time

import numpy as np

from genqai.adapt import run_adapt
from genqai.data import load_hamiltonians, load_teacher
from genqai.evaluate import score_circuit
from genqai.pool import get_pool
from genqai.schema import EvalRecord, GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator

METHODS = ("hf", "ccsd", "fci", "uccsd_vqe", "adapt", "adapt_warm", "nn_reuse", "nn_refine")
NN_METHODS = ("adapt_warm", "nn_reuse", "nn_refine")


def nearest_train(record: HamiltonianRecord, train_records: list[HamiltonianRecord]) -> HamiltonianRecord:
    """Closest training geometry (Euclidean in geometry params), excluding the record itself."""
    if any(r.split != "train" for r in train_records):
        raise ValueError("nearest_train may only search training-split geometries")
    keys = list(record.geometry_params)
    x = np.array([record.geometry_params[k] for k in keys])
    cands = [r for r in train_records if r.geometry_id != record.geometry_id]
    d = [np.linalg.norm(np.array([r.geometry_params[k] for k in keys]) - x) for r in cands]
    return cands[int(np.argmin(d))]


def _circuit(record: HamiltonianRecord, ops, angles, source: str, **meta) -> GeneratedCircuit:
    return GeneratedCircuit(record.geometry_id, record.system, list(ops), [float(a) for a in angles],
                            source=source, meta=meta)


def _classical(record: HamiltonianRecord, method: str) -> EvalRecord:
    e = getattr(record, f"{method}_energy")
    return EvalRecord(method=method, system=record.system, geometry_id=record.geometry_id, split=record.split,
                      seed=0, energy=e, error_mha=1000.0 * (e - record.fci_energy), num_operators=0,
                      converged=None if math.isnan(e) else True)


def run_method(
    method: str,
    record: HamiltonianRecord,
    train_records: list[HamiltonianRecord],
    train_bank: dict[str, GeneratedCircuit],
    own_teacher: GeneratedCircuit | None = None,
) -> EvalRecord:
    """One baseline on one geometry.

    train_bank: teacher circuits of training geometries ONLY (used by NN_METHODS).
    own_teacher: this geometry's fresh ADAPT run (used only by method "adapt").
    """
    if method in ("ccsd", "fci"):
        return _classical(record, method)
    if method == "hf":
        return score_circuit(record, _circuit(record, [], [], "hf"), "hf", converged=True)
    if method == "uccsd_vqe":
        t0 = time.perf_counter()
        ev = Evaluator(record)
        ops = list(range(len(get_pool(record.system))))
        angles, _, res = ev.optimize(ops)
        return score_circuit(record, _circuit(record, ops, angles, "uccsd_vqe"), method,
                             energy_evals=ev.energy_evals, optimizer_iterations=res.nit,
                             runtime_s=time.perf_counter() - t0, converged=bool(res.success))
    if method == "adapt":
        m = own_teacher.meta
        return score_circuit(record, own_teacher, method, converged=m["converged"],
                             pool_gradient_evals=m["pool_gradient_evals"], energy_evals=m["energy_evals"],
                             optimizer_iterations=m["optimizer_iterations"], runtime_s=m["runtime_s"],
                             extra={"final_grad_norm": m["final_grad_norm"]})
    if method not in NN_METHODS:
        raise ValueError(f"unknown method {method}")

    nb = nearest_train(record, train_records)
    src = train_bank[nb.geometry_id]
    info = {"neighbor_geometry_id": nb.geometry_id, "neighbor_params": nb.geometry_params}
    if method == "nn_reuse":
        return score_circuit(record, _circuit(record, src.operators, src.angles, method), method, extra=info)
    if method == "nn_refine":
        return score_circuit(record, _circuit(record, src.operators, src.angles, method), method,
                             refine=True, extra=info)
    # adapt_warm: same ADAPT settings as the teacher, continued from the neighbour's circuit
    c = run_adapt(record, max_ops=src.meta["max_ops"], grad_tol=src.meta["grad_tol"],
                  init_ops=src.operators, init_angles=src.angles)
    m = c.meta
    return score_circuit(record, c, method, converged=m["converged"],
                         pool_gradient_evals=m["pool_gradient_evals"], energy_evals=m["energy_evals"],
                         optimizer_iterations=m["optimizer_iterations"], runtime_s=m["runtime_s"],
                         extra={**info, "num_init_ops": len(src.operators),
                                "num_added_ops": len(c.operators) - len(src.operators)})


def run_baselines(system: str, methods=METHODS) -> dict[str, list[EvalRecord]]:
    records = load_hamiltonians(system)
    teacher = load_teacher(system)
    train = [r for r in records if r.split == "train"]
    train_bank = {r.geometry_id: teacher[r.geometry_id] for r in train}
    return {
        m: [run_method(m, r, train, train_bank, own_teacher=teacher[r.geometry_id] if m == "adapt" else None)
            for r in records]
        for m in methods
    }
