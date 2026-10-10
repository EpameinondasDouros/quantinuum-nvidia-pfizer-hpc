"""Workstream B: ADAPT-VQE teacher with full per-step logging.

Fermionic singles+doubles pool (genqai.pool). At each step: pool gradients at theta=0, append
argmax |g|, re-optimize ALL angles (BFGS, warm-started from previous angles + 0).
Convergence (gradient-norm criterion) is recorded separately from the iteration cap: a run that
hits max_ops is labelled converged=False regardless of its energy error.
"""
from __future__ import annotations

import time

import numpy as np

from genqai.schema import GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator


def wrap_angle(t: float) -> float:
    return float((t + np.pi) % (2 * np.pi) - np.pi)


def run_adapt(
    record: HamiltonianRecord,
    max_ops: int = 16,
    grad_tol: float = 1e-3,
    energy_tol_mha: float | None = None,
    init_ops: list[int] | None = None,
    init_angles: list[float] | None = None,
) -> GeneratedCircuit:
    """Fresh ADAPT when init_ops is None; otherwise continues from a warm-start ansatz."""
    t0 = time.perf_counter()
    ev = Evaluator(record)
    ops = list(init_ops or [])
    angles = np.array(init_angles if init_angles is not None else [0.0] * len(ops), float)
    steps, nit = [], 0
    if ops:
        angles, energy, res = ev.optimize(ops, angles)
        nit += res.nit if res is not None else 0
    else:
        energy = ev.energy(ops, angles)
    converged = False
    while True:
        g = ev.pool_gradients(ops, angles)
        gnorm = float(np.linalg.norm(g))
        if gnorm < grad_tol or (
            energy_tol_mha is not None and 1000 * (energy - record.fci_energy) < energy_tol_mha
        ):
            converged = True
            break
        if len(ops) >= max_ops:
            break
        k = int(np.argmax(np.abs(g)))
        ops.append(k)
        angles, energy, res = ev.optimize(ops, np.append(angles, 0.0))
        nit += res.nit
        steps.append(
            {
                "op": k,
                "grad": float(g[k]),
                "grad_norm": gnorm,
                "energy": energy,
                "error_mha": 1000 * (energy - record.fci_energy),
                "angles": [float(a) for a in angles],
                "cum_pool_gradient_evals": ev.pool_gradient_evals,
                "cum_energy_evals": ev.energy_evals,
                "cum_optimizer_iterations": nit,
            }
        )
    return GeneratedCircuit(
        geometry_id=record.geometry_id,
        system=record.system,
        operators=ops,
        angles=[wrap_angle(a) for a in angles],
        valid=True,
        source="adapt" if init_ops is None else "adapt_warm",
        meta={
            "energy": energy,
            "error_mha": 1000 * (energy - record.fci_energy),
            "converged": converged,
            "final_grad_norm": gnorm,
            "pool_gradient_evals": ev.pool_gradient_evals,
            "energy_evals": ev.energy_evals,
            "optimizer_iterations": nit,
            "runtime_s": time.perf_counter() - t0,
            "grad_tol": grad_tol,
            "max_ops": max_ops,
            "steps": steps,
        },
    )
