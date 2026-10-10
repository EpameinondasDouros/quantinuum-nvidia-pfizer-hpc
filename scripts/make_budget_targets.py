"""Budget-conditioned SFT targets: nested pruned sub-circuits of the canonical label (D3).

For each TRAIN geometry, start from its data/<system>/teacher_canonical.jsonl circuit and repeatedly
drop the operator with the smallest energy increase per CX saved, re-optimising the remaining angles
each time (anchored BFGS, scripts/make_canonical_teacher.anchored_minimize, anchor = current angles).
Every circuit along the way is a target: a subsequence of the canonical order (so operator order stays
consistent across geometries) with exactly optimised angles. meta.cnot_count is its exact compiled CX
count; train_sft --budget-targets conditions on it. Pruning stops below --min-ops operators.

    python -m scripts.make_budget_targets --system h4   ->  data/<system>/teacher_budget.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from genqai.adapt import wrap_angle
from genqai.circuits import build_circuit, gate_stats
from genqai.data import DATA_DIR, load_hamiltonians, load_teacher
from genqai.rl import op_cnot_counts
from genqai.schema import GeneratedCircuit, dump_jsonl
from genqai.sim import Evaluator
from scripts.make_canonical_teacher import anchored_minimize


def prune_path(rec, label, mu: float, min_ops: int) -> list[GeneratedCircuit]:
    ev, table = Evaluator(rec), op_cnot_counts(rec.system)
    ops, angles = list(label.operators), list(label.angles)
    e = ev.energy(ops, angles)
    out = []
    while len(ops) > min_ops:
        best = None
        for i in range(len(ops)):
            o, x0 = ops[:i] + ops[i + 1:], angles[:i] + angles[i + 1:]
            x, e_new = anchored_minimize(ev, o, x0, mu)
            score = max(e_new - e, 1e-9) / table[ops[i]]
            if best is None or score < best[0]:
                best = (score, o, [wrap_angle(t) for t in x], e_new)
        _, ops, angles, e = best
        out.append(GeneratedCircuit(rec.geometry_id, rec.system, ops, angles, True, source="canonical_pruned", meta={
            "n_ops": len(ops), "energy": e, "error_mha": 1000 * (e - rec.fci_energy),
            "cnot_count": gate_stats(build_circuit(rec, ops, angles))["cnot_count"]}))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--mu", type=float, default=1e-4, help="anchor strength (Ha/rad^2)")
    p.add_argument("--min-ops", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="teacher_budget")
    a = p.parse_args(argv)

    labels = load_teacher(a.system, "teacher_canonical")
    train = load_hamiltonians(a.system, "train")
    with ProcessPoolExecutor(a.workers) as pool:
        paths = list(pool.map(prune_path, train, [labels[r.geometry_id] for r in train],
                              [a.mu] * len(train), [a.min_ops] * len(train)))
    out = [c for path in paths for c in path]
    dump_jsonl(out, DATA_DIR / a.system / f"{a.out}.jsonl")
    for n in sorted({c.meta["n_ops"] for c in out}, reverse=True):
        v = np.array([(c.meta["cnot_count"], c.meta["error_mha"]) for c in out if c.meta["n_ops"] == n])
        print(f"{n:2d} ops: CX mean {v[:, 0].mean():6.0f} [{v[:, 0].min():.0f}-{v[:, 0].max():.0f}]  err median "
              f"{np.median(v[:, 1]):8.3f}  max {v[:, 1].max():8.3f} mHa  <1.6: {np.mean(v[:, 1] <= 1.6):.2f}  "
              f"distinct seqs {len({tuple(c.operators) for c in out if c.meta['n_ops'] == n})}")


if __name__ == "__main__":
    main()
