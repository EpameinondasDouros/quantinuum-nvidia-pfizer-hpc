"""Energy-vs-CX frontier of the budget-conditioned policy vs truncated ADAPT (Workstream D3).

For every VAL and TEST geometry:
  budget  - ONE model (checkpoints/<system>/budget.pt), greedy one-shot, conditioned on each CX budget
            level (--fractions x the mean CX of the TRAIN canonical labels); exact energy and exact
            compiled CX (gate_stats) of the generated circuit.
  adapt   - fresh ADAPT truncated after k = 1..n operators, with ADAPT's own re-optimised angles at
            step k (data/<system>/teacher.jsonl meta.steps); exact compiled CX of each prefix. Each
            point needs its own ADAPT run on that geometry (cumulative evaluations are recorded).
Rows go to results/<system>/budget_frontier.jsonl (keys: curve, level, geometry_id, split, cnot_count,
err_mha, n_ops, budget, energy_evals, pool_gradient_evals; deliberately not EvalRecords).

The shipped budget level is chosen on VAL only: the smallest level at which every VAL geometry is
within --target mHa. At that level, scripts/generate.generate scores ALL geometries with the shared
harness -> results/<system>/budget_oneshot.jsonl (method "budget_oneshot").

    python -m scripts.budget_frontier --system h4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from genqai.circuits import build_circuit, gate_stats
from genqai.data import load_hamiltonians, load_teacher
from genqai.model import GQEPolicy
from genqai.rl import cnot_estimate
from genqai.schema import dump_jsonl
from genqai.sim import Evaluator
from scripts.generate import generate

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--ckpt", default=None, help="default checkpoints/<system>/budget.pt")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    p.add_argument("--target", type=float, default=1.6, help="VAL accuracy target for the shipped level (mHa)")
    p.add_argument("--family", default="budget")
    a = p.parse_args(argv)

    torch.manual_seed(0)
    policy = GQEPolicy.load(a.ckpt or ROOT / "checkpoints" / a.system / f"{a.family}.pt")
    labels = load_teacher(a.system, "teacher_canonical")
    adapt = load_teacher(a.system)
    train = load_hamiltonians(a.system, "train")
    ref = float(np.mean([cnot_estimate(a.system, labels[r.geometry_id].operators) for r in train]))
    levels = [int(round(f * ref)) for f in a.fractions]
    recs = [r for r in load_hamiltonians(a.system) if r.split == "val" or r.split.startswith("test")]

    rows = []
    for r in recs:
        ev = Evaluator(r)
        common = {"geometry_id": r.geometry_id, "split": r.split}
        for lv in levels:
            ops, angles, _, valid = policy.sample(policy.condition([r], cnot_budget=lv), greedy=True)[0]
            err = 1000 * (ev.energy(ops, angles) - r.fci_energy) if valid else float("nan")
            rows.append({"curve": a.family, "level": lv, "budget": lv, **common, "n_ops": len(ops), "err_mha": err,
                         "cnot_count": gate_stats(build_circuit(r, ops, angles))["cnot_count"],
                         "energy_evals": 0, "pool_gradient_evals": 0})
        steps = adapt[r.geometry_id].meta["steps"]
        for k, st in enumerate(steps, 1):
            ops = [s["op"] for s in steps[:k]]
            rows.append({"curve": "adapt", "level": k, "budget": None, **common, "n_ops": k, "err_mha": st["error_mha"],
                         "cnot_count": gate_stats(build_circuit(r, ops, st["angles"]))["cnot_count"],
                         "energy_evals": st["cum_energy_evals"], "pool_gradient_evals": st["cum_pool_gradient_evals"]})
    dump_jsonl(rows, ROOT / "results" / a.system / f"{a.family}_frontier.jsonl")

    def table(curve, split_pred):
        out = {}
        for lv in sorted({x["level"] for x in rows if x["curve"] == curve}):
            sel = [x for x in rows if x["curve"] == curve and x["level"] == lv and split_pred(x["split"])]
            e = np.abs([x["err_mha"] for x in sel])
            out[lv] = (len(sel), float(np.mean([x["cnot_count"] for x in sel])), float(np.nanmedian(e)),
                       float(np.nanmax(e)), float(np.mean(e <= a.target)))
        return out

    val = table(a.family, lambda s: s == "val")
    ok = [lv for lv, v in val.items() if v[3] <= a.target]
    chosen = min(ok) if ok else max(levels)
    for name, curve in ((a.family, a.family), ("adapt (truncated, fresh per geometry)", "adapt")):
        print(f"\n== {a.system} {name}: TEST  (level = CX budget or #ADAPT steps)")
        print(f"{'level':>6s} {'n':>3s} {'meanCX':>7s} {'med|err|':>9s} {'max|err|':>9s} {'<1.6':>5s}")
        for lv, (n, cx, med, mx, fr) in table(curve, lambda s: s.startswith("test")).items():
            print(f"{lv:6d} {n:3d} {cx:7.0f} {med:9.3f} {mx:9.3f} {100 * fr:5.0f}" + ("  <- VAL-chosen" if curve == a.family and lv == chosen else ""))
    print(f"\nVAL-chosen budget {chosen} CX (VAL max |err| {val[chosen][3]:.3f} mHa)")

    evals = [generate(policy, r, "oneshot", family=a.family, cnot_budget=chosen)[1] for r in load_hamiltonians(a.system)]
    dump_jsonl(evals, ROOT / "results" / a.system / f"{a.family}_oneshot.jsonl")


if __name__ == "__main__":
    main()
