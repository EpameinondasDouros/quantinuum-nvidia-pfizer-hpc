"""Generate circuits with a trained policy and score them with the shared harness (Workstream C).

Modes (cost counts only what the METHOD needs; final scoring is free, see genqai.evaluate):
  oneshot - greedy generation: 1 model generation, 0 energy evals (no VQE at inference)
  refine  - oneshot, then BFGS over all angles from the generated angles (counted by the harness)
  bestofk - greedy + (K-1) samples, each scored exactly (K energy evals), keep the lowest

    python scripts/generate.py --system lih4 --modes oneshot refine bestofk
Writes, for ALL geometries: results/<system>/gen_<mode><tag>.jsonl (GeneratedCircuit) and
results/<system>/sft_<mode><tag>.jsonl (EvalRecord, method "sft_<mode>").
--fresh: the same for the untouched test_fresh geometries, into results_fresh/<system>/.
With --family grpo (etc.) and --ckpt: gen_grpo_<mode><tag>.jsonl and grpo_<mode><tag>.jsonl
(method "grpo_<mode>"). --cnot-budget conditions every generation on that CX budget.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from genqai.data import FRESH_SPLIT, load_hamiltonians, load_teacher, results_dir
from genqai.evaluate import score_circuit, summarize
from genqai.generate import sample_policy
from genqai.model import GQEPolicy
from genqai.schema import GeneratedCircuit, dump_jsonl

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test_interp", "test_interval", "test_extrap", "test"]
FRESH_SPLITS = [FRESH_SPLIT]


def generate(policy: GQEPolicy, record, mode: str, k: int = 16, temperature: float = 1.0, family: str = "sft",
             cnot_budget: int | None = None):
    """-> (GeneratedCircuit, EvalRecord) for one Hamiltonian."""
    if mode not in ("oneshot", "refine", "bestofk"):
        raise ValueError(mode)
    t0 = time.perf_counter()
    ops, angles, logp, valid, c = sample_policy(policy, record, "bestofk" if mode == "bestofk" else "oneshot", k,
                                                temperature, cnot_budget)
    cost = {"model_forward_passes": c["forward_passes"], "runtime_s": time.perf_counter() - t0}
    if mode == "bestofk":
        cost["energy_evals"] = c["ranking_energy_evals"]
    method = f"{family}_{mode}"
    extra = {"logprob": logp, **({"k": k, "temperature": temperature} if mode == "bestofk" else {}),
             **({"cnot_budget": cnot_budget} if cnot_budget is not None else {})}
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
    p.add_argument("--family", default="sft", help="method prefix: <family>_<mode>")
    p.add_argument("--cnot-budget", type=int, default=None)
    p.add_argument("--fresh", action="store_true", help="score the test_fresh geometries (results_fresh/)")
    a = p.parse_args(argv)

    policy = GQEPolicy.load(a.ckpt or ROOT / "checkpoints" / a.system / "sft.pt")
    teacher = load_teacher(a.system, "teacher_fresh" if a.fresh else "teacher")
    records = load_hamiltonians(a.system, FRESH_SPLIT if a.fresh else None)
    outdir = results_dir(a.fresh) / a.system
    for mode in a.modes:
        torch.manual_seed(a.seed)
        circuits, evals = zip(*(generate(policy, r, mode, a.k, a.temperature, a.family, a.cnot_budget) for r in records))
        gen_prefix = "gen_" if a.family == "sft" else f"gen_{a.family}_"
        dump_jsonl(circuits, outdir / f"{gen_prefix}{mode}{a.tag}.jsonl")
        dump_jsonl(evals, outdir / f"{a.family}_{mode}{a.tag}.jsonl")
        stats = summarize(list(evals))[f"{a.family}_{mode}"]
        print(f"\n== {a.system} {mode}{a.tag}")
        print(f"{'split':14s} {'n':>3s} {'med|err|':>9s} {'max|err|':>9s} {'<1.6':>6s} {'<5':>6s} {'<10':>6s} "
              f"{'valid':>6s} {'#ops':>5s} {'match':>6s} {'evals':>7s}")
        for split in FRESH_SPLITS if a.fresh else SPLITS:
            s = stats[split]
            ids = {e.geometry_id for e in evals if e.split == split or (split == "test" and e.split.startswith("test"))}
            match = np.mean([c.operators == teacher[c.geometry_id].operators for c in circuits if c.geometry_id in ids])
            print(f"{split:14s} {s['n']:3d} {s['median_abs_err_mha']:9.3f} {s['max_abs_err_mha']:9.3f} "
                  f"{100 * s['frac_under_1.6mha']:6.1f} {100 * s['frac_under_5mha']:6.1f} {100 * s['frac_under_10mha']:6.1f} "
                  f"{s['valid_frac']:6.2f} {s['mean_num_operators']:5.1f} {match:6.2f} {s['mean_energy_evals']:7.1f}")


if __name__ == "__main__":
    main()
