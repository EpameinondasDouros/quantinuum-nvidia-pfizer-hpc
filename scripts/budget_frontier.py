"""Energy-vs-CX frontier of the budget-conditioned policy vs fair comparators (Workstream D3).

Curves, each evaluated on every VAL, grid TEST and test_fresh geometry with exact energies and exact
compiled CX (gate_stats):
  budget          ONE model (checkpoints/<system>/budget.pt), greedy one-shot, conditioned on each CX
                  budget level (--fractions x mean CX of the TRAIN canonical labels). 0 energy evals.
  adapt           fresh ADAPT truncated after k = 1..n operators with ADAPT's own re-optimised angles at
                  step k (data/<system>/teacher[_fresh].jsonl meta.steps). Needs a fresh ADAPT run per
                  geometry (cumulative evaluations recorded).
  regress_adapt   non-ML: kernel-ridge angle regression (genqai.regression.AngleRegressor, VAL-tuned) fitted
                  on the TRAIN ADAPT circuits truncated to k operators (angles from step k). 0 energy evals.
  regress_pruned  same regressor fitted on the TRAIN pruned canonical labels with n operators
                  (data/<system>/teacher_budget.jsonl; n = full is the canonical label). 0 energy evals.
Rows -> results/<system>/budget_frontier.jsonl (keys: curve, level, geometry_id, split, cnot_count,
err_mha, n_ops, energy_evals, pool_gradient_evals; deliberately not EvalRecords).

Operating points chosen on VAL only: for each curve, the smallest level at which every VAL geometry is
within --target mHa. The budget model at its chosen level is scored on all grid geometries and on
test_fresh with the shared harness -> results[_fresh]/<system>/budget_oneshot.jsonl.

    python -m scripts.budget_frontier --system h4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from genqai.circuits import build_circuit, gate_stats
from genqai.data import DATA_DIR, FRESH_SPLIT, load_hamiltonians, load_teacher, results_dir
from genqai.features import fit_featurizer
from genqai.model import GQEPolicy
from genqai.regression import AngleRegressor
from genqai.rl import cnot_estimate
from genqai.schema import GeneratedCircuit, dump_jsonl, load_jsonl
from genqai.sim import Evaluator
from scripts.generate import generate

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {"VAL": lambda s: s == "val", "TEST (grid)": lambda s: s.startswith("test_") and s != FRESH_SPLIT,
          "TEST_FRESH": lambda s: s == FRESH_SPLIT}


def regression_banks(system, train):
    """{(curve, level): {gid: GeneratedCircuit}} TRAIN label banks for the non-ML comparators."""
    adapt = load_teacher(system)
    banks = {}
    for r in train:
        steps = adapt[r.geometry_id].meta["steps"]
        for k in range(1, len(steps) + 1):
            banks.setdefault(("regress_adapt", k), {})[r.geometry_id] = GeneratedCircuit(
                r.geometry_id, system, [s["op"] for s in steps[:k]], steps[k - 1]["angles"])
    canon = load_teacher(system, "teacher_canonical")
    for r in train:
        banks.setdefault(("regress_pruned", len(canon[r.geometry_id].operators)), {})[r.geometry_id] = canon[r.geometry_id]
    for c in load_jsonl(DATA_DIR / system / "teacher_budget.jsonl", GeneratedCircuit):
        banks.setdefault(("regress_pruned", c.meta["n_ops"]), {})[c.geometry_id] = c
    return banks


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--ckpt", default=None, help="default checkpoints/<system>/<family>.pt")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    p.add_argument("--target", type=float, default=1.6, help="VAL accuracy target for operating points (mHa)")
    p.add_argument("--family", default="budget")
    a = p.parse_args(argv)

    torch.manual_seed(0)
    policy = GQEPolicy.load(a.ckpt or ROOT / "checkpoints" / a.system / f"{a.family}.pt")
    train, val = load_hamiltonians(a.system, "train"), load_hamiltonians(a.system, "val")
    labels = load_teacher(a.system, "teacher_canonical")
    ref = float(np.mean([cnot_estimate(a.system, labels[r.geometry_id].operators) for r in train]))
    levels = [int(round(f * ref)) for f in a.fractions]
    grid = [r for r in load_hamiltonians(a.system) if r.split == "val" or r.split.startswith("test")]
    fresh = load_hamiltonians(a.system, FRESH_SPLIT)
    adapt = {**load_teacher(a.system), **load_teacher(a.system, "teacher_fresh")}
    recs = grid + fresh
    ev = {r.geometry_id: Evaluator(r) for r in recs}

    rows = []

    def add(curve, level, r, ops, angles, valid=True, **cost):
        e = ev[r.geometry_id].energy(ops, angles)
        rows.append({"curve": curve, "level": level, "geometry_id": r.geometry_id, "split": r.split, "n_ops": len(ops),
                     "err_mha": 1000 * (e - r.fci_energy) if valid else float("nan"),
                     "cnot_count": gate_stats(build_circuit(r, ops, angles))["cnot_count"],
                     "energy_evals": cost.get("energy_evals", 0), "pool_gradient_evals": cost.get("pool_gradient_evals", 0)})

    for r in recs:
        for lv in levels:
            ops, angles, _, valid = policy.sample(policy.condition([r], cnot_budget=lv), greedy=True)[0]
            add(a.family, lv, r, ops, angles, valid)
        steps = adapt[r.geometry_id].meta["steps"]
        for k, st in enumerate(steps, 1):
            add("adapt", k, r, [s["op"] for s in steps[:k]], st["angles"], energy_evals=st["cum_energy_evals"],
                pool_gradient_evals=st["cum_pool_gradient_evals"])
    feat = fit_featurizer(a.system)
    for (curve, level), bank in sorted(regression_banks(a.system, train).items()):
        tr = [r for r in train if r.geometry_id in bank]
        if len(tr) < 5:
            continue
        reg = AngleRegressor(feat).fit(tr, bank, val)
        for r in recs:
            c = reg.predict(r)
            add(curve, level, r, c.operators, c.angles)
    dump_jsonl(rows, ROOT / "results" / a.system / f"{a.family}_frontier.jsonl")

    def table(curve, pred):
        out = {}
        for lv in sorted({x["level"] for x in rows if x["curve"] == curve}):
            sel = [x for x in rows if x["curve"] == curve and x["level"] == lv and pred(x["split"])]
            if not sel:
                continue
            e = np.abs([x["err_mha"] for x in sel])
            out[lv] = (len(sel), float(np.mean([x["cnot_count"] for x in sel])), float(np.nanmedian(e)),
                       float(np.nanmax(e)), float(np.mean(e <= a.target)))
        return out

    chosen = {}
    for curve in dict.fromkeys(x["curve"] for x in rows):
        val_t = table(curve, GROUPS["VAL"])
        ok = [lv for lv, v in val_t.items() if v[3] <= a.target]
        chosen[curve] = min(ok, key=lambda lv: val_t[lv][1]) if ok else None
        for g in ("TEST (grid)", "TEST_FRESH"):
            print(f"\n== {a.system} {curve}: {g}  (level = CX budget / #ops / #ADAPT steps)")
            print(f"{'level':>6s} {'n':>3s} {'meanCX':>7s} {'med|err|':>9s} {'max|err|':>9s} {'<1.6':>5s}")
            for lv, (n, cx, med, mx, fr) in table(curve, GROUPS[g]).items():
                print(f"{lv:6d} {n:3d} {cx:7.0f} {med:9.3f} {mx:9.3f} {100 * fr:5.0f}"
                      + ("  <- VAL-chosen" if lv == chosen[curve] else ""))
    print(f"\nVAL-chosen operating points (smallest mean VAL CX with every VAL geometry <= {a.target} mHa): {chosen}")

    lv = chosen[a.family] or max(levels)
    for fresh_flag, rs in ((False, load_hamiltonians(a.system)), (True, fresh)):
        evals = [generate(policy, r, "oneshot", family=a.family, cnot_budget=lv)[1] for r in rs]
        dump_jsonl(evals, results_dir(fresh_flag) / a.system / f"{a.family}_oneshot.jsonl")


if __name__ == "__main__":
    main()
