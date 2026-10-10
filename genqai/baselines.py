"""Workstream B baselines, all scored through genqai.evaluate.score_circuit.

Methods (results/<system>/<method>.jsonl; test_fresh rows in results_fresh/<system>/<method>.jsonl):
  hf               HF reference state, 0 operators
  ccsd, fci        classical references read from the HamiltonianRecord (no circuit)
  uccsd_vqe        every pool operator once, in pool order, BFGS from zero angles
  adapt            fresh ADAPT (the teacher run in data/<system>/teacher.jsonl; costs from its meta)
  adapt_warm       ADAPT continued from the nearest training geometry's teacher circuit
  nn_reuse         NAIVE reuse: nearest training geometry's raw teacher circuit copied verbatim,
                   no optimization and no orbital-sign correction
  nn_refine        nn_reuse + BFGS angle re-optimization
  nn_reuse_gauge   nearest training geometry's raw teacher circuit, angles mapped through the
                   orbital-phase gauge (genqai.features op_signs) into the target's sign convention
  nn_refine_gauge  nn_reuse_gauge + BFGS
  nn_reuse_canon   as nn_reuse_gauge but copying the nearest training CANONICAL label
                   (teacher_canonical.jsonl): the closest non-ML analogue of the transformer
  nn_refine_canon  nn_reuse_canon + BFGS
  angle_regression canonical sequence, angles by kernel ridge on gauge-fixed Hamiltonian features
                   (genqai.regression; hyperparameters on VAL), 0 energy evals
  angle_interp     1-D systems: canonical angles linearly interpolated in the bond length

Leakage rule: every learned baseline only ever sees TRAINING-split labels (TrainSet), VAL Hamiltonians
for angle_regression hyperparameters. Training geometries use leave-one-out for the NN methods
(their own label is excluded), so their numbers are honest too; the angle fits are in-sample there.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from genqai.adapt import run_adapt
from genqai.data import FRESH_SPLIT, load_hamiltonians, load_teacher, train_subset
from genqai.evaluate import score_circuit
from genqai.features import Featurizer, fit_featurizer
from genqai.pool import get_pool
from genqai.regression import AngleInterpolator, AngleRegressor
from genqai.schema import EvalRecord, GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator

METHODS = ("hf", "ccsd", "fci", "uccsd_vqe", "adapt", "adapt_warm", "nn_reuse", "nn_refine",
           "nn_reuse_gauge", "nn_refine_gauge", "nn_reuse_canon", "nn_refine_canon", "angle_regression",
           "angle_interp")
NN_METHODS = ("adapt_warm", "nn_reuse", "nn_refine", "nn_reuse_gauge", "nn_refine_gauge", "nn_reuse_canon",
              "nn_refine_canon")
ANGLE_METHODS = ("angle_regression", "angle_interp")


@dataclass
class TrainSet:
    """Everything a learned baseline may read: TRAIN geometries and their labels, VAL Hamiltonians."""

    records: list[HamiltonianRecord]
    bank: dict[str, GeneratedCircuit]  # raw ADAPT teacher circuits, train geometries only
    canon_bank: dict[str, GeneratedCircuit]  # canonical labels, train geometries only
    featurizer: Featurizer
    val_records: list[HamiltonianRecord]

    @cached_property
    def regressor(self) -> AngleRegressor:
        return AngleRegressor(self.featurizer).fit(self.records, self.canon_bank, self.val_records)

    @cached_property
    def interpolator(self) -> AngleInterpolator:
        return AngleInterpolator(self.featurizer).fit(self.records, self.canon_bank)


def load_train_set(system: str, n: int | None = None, subset_seed: int = 0, canon: str = "teacher_canonical",
                   featurizer: Featurizer | None = None) -> TrainSet:
    """Default: all TRAIN geometries. Learning curve / ablations: n random TRAIN geometries
    (genqai.data.train_subset), another label file as the canonical bank, another featurizer."""
    train = train_subset(load_hamiltonians(system, "train"), n, subset_seed)
    ids = [r.geometry_id for r in train]
    teacher, cbank = load_teacher(system), load_teacher(system, canon)
    return TrainSet(train, {g: teacher[g] for g in ids}, {g: cbank[g] for g in ids},
                    featurizer or fit_featurizer(system), load_hamiltonians(system, "val"))


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


def gauge_map(featurizer: Featurizer, record: HamiltonianRecord, source: HamiltonianRecord, circuit) -> list[float]:
    """circuit's angles (in source's orbital-sign convention) mapped into record's convention."""
    s_t, s_src = featurizer.op_signs([record, source])
    return [float(s_t[o] * s_src[o] * t) for o, t in zip(circuit.operators, circuit.angles)]


def run_method(
    method: str,
    record: HamiltonianRecord,
    train: TrainSet,
    own_teacher: GeneratedCircuit | None = None,
) -> EvalRecord:
    """One baseline on one geometry.

    train: the only labels NN_METHODS / ANGLE_METHODS may read.
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
    if method in ANGLE_METHODS:
        t0 = time.perf_counter()
        c = (train.regressor if method == "angle_regression" else train.interpolator).predict(record)
        return score_circuit(record, c, method, runtime_s=time.perf_counter() - t0, extra=c.meta)
    if method not in NN_METHODS:
        raise ValueError(f"unknown method {method}")

    nb = nearest_train(record, train.records)
    src = (train.canon_bank if method.endswith("_canon") else train.bank)[nb.geometry_id]
    info = {"neighbor_geometry_id": nb.geometry_id, "neighbor_params": nb.geometry_params}
    if method.startswith("nn_"):
        naive = method in ("nn_reuse", "nn_refine")
        angles = src.angles if naive else gauge_map(train.featurizer, record, nb, src)
        return score_circuit(record, _circuit(record, src.operators, angles, method), method,
                             refine=method.startswith("nn_refine"), extra=info)
    # adapt_warm: same ADAPT settings as the teacher, continued from the neighbour's circuit
    c = run_adapt(record, max_ops=src.meta["max_ops"], grad_tol=src.meta["grad_tol"],
                  init_ops=src.operators, init_angles=src.angles)
    m = c.meta
    return score_circuit(record, c, method, converged=m["converged"],
                         pool_gradient_evals=m["pool_gradient_evals"], energy_evals=m["energy_evals"],
                         optimizer_iterations=m["optimizer_iterations"], runtime_s=m["runtime_s"],
                         extra={**info, "num_init_ops": len(src.operators),
                                "num_added_ops": len(c.operators) - len(src.operators)})


def applicable(method: str, system: str) -> bool:
    return method != "angle_interp" or len(load_hamiltonians(system, "train")[0].geometry_params) == 1


def run_baselines(system: str, methods=METHODS, fresh: bool = False) -> dict[str, list[EvalRecord]]:
    """Every method on every grid geometry, or (fresh=True) on the test_fresh geometries, whose own
    ADAPT runs (data/<system>/teacher_fresh.jsonl) are read only by method "adapt"."""
    records = load_hamiltonians(system, FRESH_SPLIT if fresh else None)
    own = load_teacher(system, "teacher_fresh" if fresh else "teacher")
    train = load_train_set(system)
    return {
        m: [run_method(m, r, train, own_teacher=own[r.geometry_id] if m == "adapt" else None) for r in records]
        for m in methods if applicable(m, system)
    }
