"""Generate circuits with a trained policy and score them with the shared harness (Workstream C).

Modes (cost counts only what the METHOD needs; final scoring is free, see genqai.evaluate):
  oneshot - greedy generation: 1 model generation, 0 energy evals (no VQE at inference)
  refine  - oneshot, then BFGS over all angles from the generated angles (counted by the harness)
  bestofk - greedy + (K-1) samples, each scored exactly (K energy evals), keep the lowest

    python scripts/generate.py --system lih4 --modes oneshot refine bestofk
Writes, for ALL geometries: results/<system>/gen_<mode><tag>.jsonl (GeneratedCircuit) and
results/<system>/sft_<mode><tag>.jsonl (EvalRecord, method "sft_<mode>").
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from genqai.data import load_hamiltonians, load_teacher
from genqai.evaluate import score_circuit, summarize
from genqai.model import GQEPolicy
from genqai.schema import GeneratedCircuit, dump_jsonl
from genqai.sim import Evaluator

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test_interp", "test_interval", "test_extrap", "test"]


def generate(policy: GQEPolicy, record, mode: str, k: int = 16, temperature: float = 1.0):
    """-> (GeneratedCircuit, EvalRecord) for one Hamiltonian."""
    t0 = time.perf_counter()
    cond = policy.condition([record])
    ops, angles, logp, valid = policy.sample(cond, greedy=True)[0]
    cost = {"model_forward_passes": 1}
    if mode == "bestofk":
        ev = Evaluator(record)
        cands = [(ops, angles, logp, valid)] + policy.sample(cond, n=k - 1, temperature=temperature)
        energies = [ev.energy(o, a) if v else np.inf for o, a, _, v in cands]
        ops, angles, logp, valid = cands[int(np.argmin(energies))]
        cost = {"model_forward_passes": k, "energy_evals": k}
    elif mode not in ("oneshot", "refine"):
        raise ValueError(mode)
    cost["runtime_s"] = time.perf_counter() - t0
    method = f"sft_{mode}"
    extra = {"logprob": logp, **({"k": k, "temperature": temperature} if mode == "bestofk" else {})}
    circ = GeneratedCircuit(record.geometry_id, record.system, ops, angles, valid, source=method, meta=extra)
    rec = score_circuit(record, circ, method, refine=mode == "refine", extra=extra, **cost)
    if mode == "refine" and valid:
        circ.angles = rec.extra["refined_angles"]
    circ.meta.update(energy=rec.energy, error_mha=rec.error_mha, split=record.split,
                     **{f: getattr(rec, f) for f in ("model_forward_passes", "energy_evals", "runtime_s")})
    return circ, rec


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--modes", nargs="+", default=["oneshot", "refine", "bestofk"])
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    policy = GQEPolicy.load(a.ckpt or ROOT / "checkpoints" / a.system / "sft.pt")
    teacher = load_teacher(a.system)
    records = load_hamiltonians(a.system)
    outdir = ROOT / "results" / a.system
    for mode in a.modes:
        torch.manual_seed(a.seed)
        circuits, evals = zip(*(generate(policy, r, mode, a.k, a.temperature) for r in records))
        dump_jsonl(circuits, outdir / f"gen_{mode}{a.tag}.jsonl")
        dump_jsonl(evals, outdir / f"sft_{mode}{a.tag}.jsonl")
        stats = summarize(list(evals))[f"sft_{mode}"]
        print(f"\n== {a.system} {mode}{a.tag}")
        print(f"{'split':14s} {'n':>3s} {'med|err|':>9s} {'max|err|':>9s} {'<1.6':>6s} {'<5':>6s} {'<10':>6s} "
              f"{'valid':>6s} {'#ops':>5s} {'match':>6s} {'evals':>7s}")
        for split in SPLITS:
            s = stats[split]
            ids = {e.geometry_id for e in evals if e.split == split or (split == "test" and e.split.startswith("test"))}
            match = np.mean([c.operators == teacher[c.geometry_id].operators for c in circuits if c.geometry_id in ids])
            print(f"{split:14s} {s['n']:3d} {s['median_abs_err_mha']:9.3f} {s['max_abs_err_mha']:9.3f} "
                  f"{100 * s['frac_under_1.6mha']:6.1f} {100 * s['frac_under_5mha']:6.1f} {100 * s['frac_under_10mha']:6.1f} "
                  f"{s['valid_frac']:6.2f} {s['mean_num_operators']:5.1f} {match:6.2f} {s['mean_energy_evals']:7.1f}")


if __name__ == "__main__":
    main()
