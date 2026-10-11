"""Circuit generation behind a routing decision, with exact cost accounting (workbench lane B).

    decision = genqai.registry.route(config, record, report)
    circuit, log = generate(record, "auto", decision)            # or "krr", "transformer_sft+refine", ...

Methods: a registry model kind ("transformer_sft", "transformer_grpo", "transformer_budget" (opt
cnot_budget), "krr", "nn_reuse_gauge", "angle_interp"), optionally suffixed "+bestofk" (transformers;
opts k, temperature, seed) and/or "+refine"; "bestofk" alone = "transformer_sft+bestofk"; the
conventional solvers "adapt" and "uccsd_vqe"; "auto" = decision.recommended.

Accounting (GenerationLog). "Zero quantum evaluations" means zero to PROPOSE the circuit:
  forward_passes            sequences decoded (transformer), regressor/interpolator queries (krr,
                            angle_interp); 0 for nn_reuse_gauge (a table lookup) and the solvers.
                            details["decoder_steps"] counts the autoregressive steps behind them.
  candidates                circuits proposed before selection (K for best-of-K).
  quantum_evals_to_propose  0 for one-shot / KRR / NN reuse / interpolation; K for best-of-K (the K
                            candidate energies used for ranking); energy evals + pool-gradient evals
                            for ADAPT; energy evals for UCCSD-VQE. Never includes refinement.
  refine_energy_evals       BFGS energy(+gradient) evaluations of "+refine", counted separately.
  adapt_pool_gradient_evals individual pool-operator gradients of ADAPT (also inside the total above).
Scoring the final circuit (lane C) is never counted here.

Safety: a learned generator only runs when the routing decision lists its registry card as
compatible (paths A and B); on path B it runs but the log is marked experimental. Path C allows only
"adapt"/"uccsd_vqe" (and logs the circuits as teacher candidates, runs/teacher_candidates/); path D
runs nothing. Without torch, transformer_sft runs from its NumPy export (genqai.infer_numpy; same
circuits to float32 round-off); torch-only methods raise.
"""
from __future__ import annotations

import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from genqai.schema import GeneratedCircuit, HamiltonianRecord

ROOT = Path(__file__).resolve().parent.parent
TEACHER_CANDIDATES_DIR = ROOT / "runs" / "teacher_candidates"  # gitignored (runs/)

TRANSFORMER_STAGES = {"transformer_sft": "sft", "transformer_grpo": "grpo", "transformer_budget": "budget"}
LEARNED_KINDS = (*TRANSFORMER_STAGES, "krr", "nn_reuse_gauge", "angle_interp")
SOLVERS = ("adapt", "uccsd_vqe")
NUMPY_EXPORTS = {"transformer_sft": "sft"}  # qollab/checkpoints/<system>/<stage>.json
DEFAULT_ADAPT = {"max_ops": 16, "grad_tol": 1e-3}  # genqai.adapt defaults; presets reuse the teacher's settings


class GenerationError(ValueError):
    """User-facing refusal (incompatible model, missing dependency, no execution on path D)."""


@dataclass
class GenerationLog:
    method: str
    model_id: str | None
    forward_passes: int
    candidates: int
    model_time_s: float
    quantum_evals_to_propose: int  # 0 for genuine one-shot
    refine_energy_evals: int
    adapt_pool_gradient_evals: int
    validity: bool
    experimental: bool
    notes: list[str]
    details: dict = field(default_factory=dict)  # backend, decoder_steps, refine iterations, energies, ...


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def checkpoint_path(system: str, kind: str) -> Path | None:
    stage = TRANSFORMER_STAGES.get(kind)
    return ROOT / "checkpoints" / system / f"{stage}.pt" if stage else None


def numpy_export_path(system: str, kind: str) -> Path | None:
    stage = NUMPY_EXPORTS.get(kind)
    return ROOT / "qollab" / "checkpoints" / system / f"{stage}.json" if stage else None


def parse_method(method: str) -> tuple[str, str, bool]:
    """'transformer_grpo+bestofk+refine' -> ('transformer_grpo', 'bestofk', True)."""
    base, *suffixes = method.split("+")
    if base == "bestofk":
        base, suffixes = "transformer_sft", ["bestofk", *suffixes]
    if base not in LEARNED_KINDS + SOLVERS or set(suffixes) - {"bestofk", "refine"} or len(set(suffixes)) != len(suffixes):
        raise GenerationError(f"unknown method {method!r}; expected one of {LEARNED_KINDS + SOLVERS} "
                              "with optional '+bestofk' (transformers) / '+refine'")
    if "bestofk" in suffixes and base not in TRANSFORMER_STAGES:
        raise GenerationError(f"best-of-K sampling needs a transformer, not {base!r}")
    if base in SOLVERS and suffixes:
        raise GenerationError(f"{base} already optimises its angles; no suffixes allowed")
    return base, "bestofk" if "bestofk" in suffixes else "oneshot", "refine" in suffixes


# --------------------------------------------------------------------------- model loading (cached)

@lru_cache(maxsize=None)
def _torch_policy(path: str):
    from genqai.model import GQEPolicy

    return GQEPolicy.load(path)


@lru_cache(maxsize=None)
def _numpy_policy(path: str):
    from genqai.infer_numpy import NumpyPolicy

    return NumpyPolicy.load(path)


@lru_cache(maxsize=None)
def train_set(system: str):
    """TRAIN labels + VAL Hamiltonians of a preset system (genqai.baselines.load_train_set)."""
    from genqai.baselines import load_train_set

    return load_train_set(system)


def krr_config(system: str) -> dict:
    """Hyperparameters of the deterministic KRR fit (alpha/gamma/sequence rule chosen on VAL energies)."""
    reg = train_set(system).regressor
    return {"alpha": float(reg.alpha), "gamma": float(reg.gamma), "seq_rule": reg.rule}


# --------------------------------------------------------------------------- transformer sampling

def sample_policy(policy, record: HamiltonianRecord, mode: str = "oneshot", k: int = 16,
                  temperature: float = 1.0, cnot_budget: int | None = None):
    """Greedy (oneshot) or greedy + (k-1) samples ranked by exact energy (bestofk) from a GQEPolicy.

    -> (ops, angles, logprob, valid, cost) with cost = {forward_passes, candidates, ranking_energy_evals,
    decoder_steps}. Shared by scripts/generate.py and generate().
    """
    from genqai.sim import Evaluator

    cond = policy.condition([record], cnot_budget=cnot_budget)
    cands = policy.sample(cond, greedy=True)
    if mode == "bestofk":
        cands += policy.sample(cond, n=k - 1, temperature=temperature)
    elif mode != "oneshot":
        raise ValueError(mode)
    cost = {"forward_passes": len(cands), "candidates": len(cands), "ranking_energy_evals": 0,
            "decoder_steps": sum(min(len(o) + 1, policy.cfg.max_len) for o, _, _, _ in cands)}
    best = 0
    if mode == "bestofk":
        ev = Evaluator(record)
        energies = [ev.energy(o, a) if v else np.inf for o, a, _, v in cands]
        cost["ranking_energy_evals"] = ev.energy_evals
        best = int(np.argmin(energies))
    ops, angles, logp, valid = cands[best]
    return ops, angles, logp, valid, cost


# --------------------------------------------------------------------------- proposal (no routing)

def propose(record: HamiltonianRecord, kind: str, system: str | None = None, *, sampler: str = "oneshot",
            k: int = 16, temperature: float = 1.0, seed: int = 0, cnot_budget: int | None = None,
            backend: str = "auto") -> tuple[GeneratedCircuit, dict]:
    """One circuit from the `kind` model trained on preset `system` (default record.system), unrouted.

    record.geometry_params must carry the family parameters for nn_reuse_gauge / angle_interp.
    backend: "auto" | "torch" | "numpy" (transformers). -> (circuit, cost dict).
    """
    system = system or record.system
    t0 = time.perf_counter()
    cost = {"forward_passes": 1, "candidates": 1, "ranking_energy_evals": 0}
    meta: dict = {}
    if kind in TRANSFORMER_STAGES:
        use_numpy = backend == "numpy" or (backend == "auto" and not torch_available())
        if use_numpy:
            export = numpy_export_path(system, kind)
            if sampler != "oneshot" or cnot_budget is not None or export is None or not export.exists():
                raise GenerationError(f"{kind}{'+' + sampler if sampler != 'oneshot' else ''} needs torch "
                                      "(only greedy, unconditioned transformer_sft has a NumPy export)")
            ops, angles, valid = _numpy_policy(str(export)).generate(record)
            cost["decoder_steps"] = min(len(ops) + 1, _numpy_policy(str(export)).cfg["max_len"])
            meta["backend"] = f"numpy ({export.relative_to(ROOT)})"
        else:
            if not torch_available():
                raise GenerationError(f"{kind} needs torch, which is not installed")
            import torch

            torch.manual_seed(seed)
            ckpt = checkpoint_path(system, kind)
            ops, angles, logp, valid, cost = sample_policy(_torch_policy(str(ckpt)), record, sampler, k,
                                                           temperature, cnot_budget)
            meta.update(backend=f"torch ({ckpt.relative_to(ROOT)})", logprob=logp)
            if sampler == "bestofk":
                meta.update(k=k, temperature=temperature, seed=seed)
            if cnot_budget is not None:
                meta["cnot_budget"] = cnot_budget
        circ = GeneratedCircuit(record.geometry_id, record.system, ops, angles, valid, meta=meta)
    elif kind == "krr":
        try:
            circ = train_set(system).regressor.predict(record)
        except ImportError as exc:
            raise GenerationError(f"krr needs scikit-learn ({exc})") from exc
    elif kind == "angle_interp":
        circ = train_set(system).interpolator.predict(record)
    elif kind == "nn_reuse_gauge":
        from genqai.baselines import gauge_map, nearest_train

        ts = train_set(system)
        nb = nearest_train(record, ts.records)
        src = ts.bank[nb.geometry_id]
        circ = GeneratedCircuit(record.geometry_id, record.system, list(src.operators),
                                gauge_map(ts.featurizer, record, nb, src),
                                meta={"neighbor_geometry_id": nb.geometry_id, "neighbor_params": nb.geometry_params})
        cost["forward_passes"] = 0
    else:
        raise GenerationError(f"{kind!r} is not a learned generator")
    cost["model_time_s"] = time.perf_counter() - t0
    cost.setdefault("decoder_steps", 0)
    circ.source = kind if sampler == "oneshot" else f"{kind}+{sampler}"
    return circ, cost


# --------------------------------------------------------------------------- conventional solvers

def _solve(record: HamiltonianRecord, method: str, system: str | None, opts: dict) -> tuple[GeneratedCircuit, dict]:
    from genqai.pool import get_pool
    from genqai.sim import Evaluator

    t0 = time.perf_counter()
    if method == "adapt":
        from genqai.adapt import run_adapt

        settings = dict(DEFAULT_ADAPT)
        if system is not None:  # same settings as that system's teacher labels
            m = next(iter(train_set(system).bank.values())).meta
            settings = {"max_ops": m["max_ops"], "grad_tol": m["grad_tol"]}
        settings.update({k: opts[k] for k in ("max_ops", "grad_tol") if k in opts})
        circ = run_adapt(record, **settings)
        m = circ.meta
        return circ, {"energy_evals": m["energy_evals"], "pool_gradient_evals": m["pool_gradient_evals"],
                      "optimizer_iterations": m["optimizer_iterations"], "converged": m["converged"],
                      "energy": m["energy"], "settings": settings, "time_s": time.perf_counter() - t0}
    ev = Evaluator(record)  # uccsd_vqe: every pool operator once, in pool order, BFGS from zero
    ops = list(range(len(get_pool(record.system))))
    angles, energy, res = ev.optimize(ops)
    circ = GeneratedCircuit(record.geometry_id, record.system, ops, [float(a) for a in angles], source="uccsd_vqe",
                            meta={"energy": energy})
    return circ, {"energy_evals": ev.energy_evals, "pool_gradient_evals": 0, "optimizer_iterations": res.nit,
                  "converged": bool(res.success), "energy": energy, "time_s": time.perf_counter() - t0}


def _log_teacher_candidate(record: HamiltonianRecord, circ: GeneratedCircuit, method: str, info: dict) -> Path:
    key = getattr(record, "problem_key", "") or record.geometry_id
    path = TEACHER_CANDIDATES_DIR / f"{key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"problem_key": key, "method": method, "record_system": record.system, "geometry_id": record.geometry_id,
           "num_qubits": record.num_qubits, "fci_energy": record.fci_energy,
           "circuit": asdict(circ), "cost": {k: v for k, v in info.items() if k != "settings"}}
    with path.open("a") as f:
        f.write(json.dumps(row, default=float) + "\n")
    return path


def _valid(record: HamiltonianRecord, circ: GeneratedCircuit) -> bool:
    from genqai.pool import get_pool

    n = len(get_pool(record.system))
    return (bool(circ.valid) and len(circ.operators) == len(circ.angles)
            and all(0 <= int(o) < n for o in circ.operators) and all(math.isfinite(a) for a in circ.angles))


# --------------------------------------------------------------------------- public entry point

def generate(record: HamiltonianRecord, method: str, decision, **opts) -> tuple[GeneratedCircuit, GenerationLog]:
    """One circuit for `record` by `method`, allowed by `decision` (genqai.registry.RoutingDecision).

    opts: k (16), temperature (1.0), seed (0) for best-of-K; cnot_budget for transformer_budget
    (default: the registry card's VAL-chosen operating point); backend "auto"|"torch"|"numpy";
    max_ops / grad_tol for adapt.
    """
    if decision.path == "D":
        raise GenerationError("nothing is executed for this problem (path D): " + "; ".join(decision.reasons))
    if method == "auto":
        if not decision.recommended:
            raise GenerationError("no recommended method for this problem")
        method = decision.recommended
    base, sampler, refine = parse_method(method)
    notes: list[str] = []
    details: dict = {}
    experimental = decision.path == "B"
    model_id = None

    if base in SOLVERS:
        circ, info = _solve(record, base, decision.system if decision.path in ("A", "B") else None, opts)
        circ.source = base
        details.update(info)
        log = GenerationLog(base, None, 0, 1, 0.0, info["energy_evals"] + info["pool_gradient_evals"], 0,
                            info["pool_gradient_evals"] if base == "adapt" else 0, _valid(record, circ),
                            experimental, notes, details)
        notes.append(f"{base}: {info['energy_evals']} energy evals + {info['pool_gradient_evals']} pool-gradient "
                     "evals to propose the circuit")
        if decision.path == "C":
            p = _log_teacher_candidate(record, circ, base, info)
            notes.append(f"logged as a potential teacher example: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
        return circ, log

    model_id = f"{decision.system}/{base}"
    if decision.path == "C" or decision.system is None or model_id not in decision.compatible:
        raise GenerationError(f"no compatible trained generator for this problem ({method!r} requested); "
                              f"available: {', '.join(decision.available_methods) or 'none'}")
    view = record
    if decision.params is not None:  # family parameters as matched (e.g. r of an H2 typed in any frame)
        from genqai.registry import card_for

        names = card_for(model_id).family["params"]
        view = replace(record, geometry_params=dict(zip(names, decision.params)))
    budget = opts.get("cnot_budget")
    if base == "transformer_budget" and budget is None:
        from genqai.registry import card_for

        budget = card_for(model_id).config["default_cnot_budget"]
        notes.append(f"cnot_budget = {budget} (VAL-chosen operating point from the registry)")
    circ, cost = propose(view, base, decision.system, sampler=sampler, k=opts.get("k", 16),
                         temperature=opts.get("temperature", 1.0), seed=opts.get("seed", 0),
                         cnot_budget=budget, backend=opts.get("backend", "auto"))
    circ.meta.pop("val_mean_abs_err_mha", None)  # a VAL statistic, not a property of this circuit
    details.update(decoder_steps=cost["decoder_steps"], backend=circ.meta.get("backend", "numpy/sklearn"))
    if sampler == "bestofk":
        notes.append(f"best-of-{cost['candidates']}: {cost['ranking_energy_evals']} exact energy evaluations "
                     "used to rank the candidates (counted in quantum_evals_to_propose)")
    refine_evals = 0
    if refine and circ.valid:
        from genqai.sim import Evaluator

        t0 = time.perf_counter()
        ev = Evaluator(record)
        angles, energy, res = ev.optimize(circ.operators, circ.angles)
        refine_evals = ev.energy_evals
        circ.meta["unrefined_angles"] = list(circ.angles)
        circ.angles = [float(a) for a in angles]
        circ.source += "+refine"
        details.update(refine_iterations=res.nit if res is not None else 0, refine_time_s=time.perf_counter() - t0,
                       refined_energy=energy)
    if experimental:
        notes.append("EXPERIMENTAL: outside the validated domain of this generator ("
                     + "; ".join(decision.reasons) + "); one-shot circuits are unreliable here, "
                     "prefer '+refine' or 'adapt'")
    log = GenerationLog(circ.source, model_id, cost["forward_passes"], cost["candidates"], cost["model_time_s"],
                        cost["ranking_energy_evals"], refine_evals, 0, _valid(record, circ), experimental, notes, details)
    return circ, log
