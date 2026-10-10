"""Generate circuits with a trained policy and score them exactly (Workstream C).

Modes (energy_evals counts only evaluations the METHOD needs; final scoring is free):
  oneshot - greedy generation, 1 model generation, 0 energy evals (no VQE at inference)
  refine  - oneshot, then BFGS over all angles starting from the generated angles
  bestofk - greedy + (K-1) samples, each scored exactly (K energy evals), keep the lowest

    python scripts/generate.py --system lih4 --modes oneshot refine bestofk
Writes results/<system>/gen_<mode><tag>.jsonl and results/<system>/gen_summary<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from genqai.adapt import wrap_angle
from genqai.data import load_hamiltonians, load_teacher
from genqai.model import GQEPolicy
from genqai.schema import GeneratedCircuit, dump_jsonl
from genqai.sim import Evaluator

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "val", "test_interp", "test_interval", "test_extrap"]


def generate(policy: GQEPolicy, record, mode: str, k: int = 16, temperature: float = 1.0) -> GeneratedCircuit:
    t0 = time.perf_counter()
    cond = policy.condition([record])
    ops, angles, logp, valid = policy.sample(cond, greedy=True)[0]
    passes, evals = 1, 0
    if mode == "refine":
        ev = Evaluator(record)
        x, _, _ = ev.optimize(ops, angles)
        angles, evals = [wrap_angle(t) for t in x], ev.energy_evals
    elif mode == "bestofk":
        ev = Evaluator(record)
        cands = [(ops, angles, logp, valid)] + policy.sample(cond, n=k - 1, temperature=temperature)
        energies = [ev.energy(o, a) if v else np.inf for o, a, _, v in cands]
        ops, angles, logp, valid = cands[int(np.argmin(energies))]
        passes, evals = k, ev.energy_evals
    elif mode != "oneshot":
        raise ValueError(mode)
    runtime = time.perf_counter() - t0
    energy = Evaluator(record).energy(ops, angles)  # scoring only, not charged to the method
    return GeneratedCircuit(
        geometry_id=record.geometry_id, system=record.system, operators=ops, angles=angles, valid=valid,
        source=f"sft_{mode}",
        meta={
            "energy": energy, "error_mha": 1000 * (energy - record.fci_energy), "split": record.split,
            "model_forward_passes": passes, "energy_evals": evals, "runtime_s": runtime, "logprob": logp,
            **({"k": k, "temperature": temperature} if mode == "bestofk" else {}),
        },
    )


def summarize(circuits, teacher) -> dict:
    out = {}
    for split in SPLITS + ["test"]:
        cs = [c for c in circuits if c.meta["split"] == split or (split == "test" and c.meta["split"].startswith("test"))]
        if not cs:
            continue
        e = np.abs([c.meta["error_mha"] for c in cs])
        out[split] = {
            "n": len(cs),
            "median_abs_mha": float(np.median(e)), "max_abs_mha": float(e.max()),
            **{f"pct_under_{t}": float(100 * np.mean(e < t)) for t in (1.6, 5, 10)},
            "valid_rate": float(np.mean([c.valid for c in cs])),
            "mean_ops": float(np.mean([len(c.operators) for c in cs])),
            "teacher_exact_match": float(np.mean([c.operators == teacher[c.geometry_id].operators for c in cs])),
            "mean_energy_evals": float(np.mean([c.meta["energy_evals"] for c in cs])),
        }
    return out


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
    summary_path = outdir / f"gen_summary{a.tag}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for mode in a.modes:
        torch.manual_seed(a.seed)
        circuits = [generate(policy, r, mode, a.k, a.temperature) for r in records]
        dump_jsonl(circuits, outdir / f"gen_{mode}{a.tag}.jsonl")
        summary[mode] = summarize(circuits, teacher)
        print(f"\n== {a.system} {mode}{a.tag}")
        print(f"{'split':14s} {'n':>3s} {'med|err|':>9s} {'max|err|':>9s} {'<1.6':>6s} {'<5':>6s} {'<10':>6s} "
              f"{'valid':>6s} {'#ops':>5s} {'match':>6s} {'evals':>7s}")
        for split, s in summary[mode].items():
            print(f"{split:14s} {s['n']:3d} {s['median_abs_mha']:9.3f} {s['max_abs_mha']:9.3f} "
                  f"{s['pct_under_1.6']:6.1f} {s['pct_under_5']:6.1f} {s['pct_under_10']:6.1f} {s['valid_rate']:6.2f} "
                  f"{s['mean_ops']:5.1f} {s['teacher_exact_match']:6.2f} {s['mean_energy_evals']:7.1f}")
    summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
