"""Self-distillation labels from a GRPO replay DB (Workstream D2).

For every TRAIN geometry, replace the canonical teacher label by the best replay circuit only if it
is better: lower energy by more than --tol mHa, or energy within --tol mHa with fewer CX. Among
qualifying circuits the one with the fewest CX (then lowest energy) wins. With --polish, replay
circuits first get an anchored BFGS angle polish (exact, TRAIN geometries only). Non-train labels
are copied unchanged (train_sft uses VAL labels only for its VAL loss).

    python -m scripts.distill_targets --system h4 --replay grpo
    python -m scripts.train_sft --system h4 --init checkpoints/h4/sft.pt \
        --targets results/h4/distill_targets.jsonl --out checkpoints/h4/distill.pt \
        --log results/h4/distill_train_log.json --epochs 500 --lr 3e-4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from genqai.adapt import wrap_angle
from genqai.data import load_hamiltonians, load_teacher
from genqai.rl import cnot_estimate
from genqai.schema import GeneratedCircuit, dump_jsonl, load_jsonl
from genqai.sim import Evaluator
from scripts.make_canonical_teacher import anchored_minimize

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--replay", default="grpo", help="results/<system>/<replay>_replay.jsonl")
    p.add_argument("--tol", type=float, default=0.01, help="'equal energy' tolerance (mHa)")
    p.add_argument("--polish", action="store_true")
    p.add_argument("--mu", type=float, default=1e-4)
    p.add_argument("--out", default=None, help="default results/<system>/distill_targets.jsonl")
    a = p.parse_args(argv)

    labels = load_teacher(a.system, "teacher_canonical")
    recs = {r.geometry_id: r for r in load_hamiltonians(a.system, "train")}
    replay = [c for c in load_jsonl(ROOT / "results" / a.system / f"{a.replay}_replay.jsonl", GeneratedCircuit)
              if c.geometry_id in recs and c.source != "teacher_canonical"]
    ev = {g: Evaluator(r) for g, r in recs.items()}
    cands: dict[str, list] = {}
    for c in replay:
        angles, e = c.angles, c.meta["energy"]
        if a.polish:
            x, e = anchored_minimize(ev[c.geometry_id], c.operators, c.angles, a.mu)
            angles = [wrap_angle(t) for t in x]
        cands.setdefault(c.geometry_id, []).append((cnot_estimate(a.system, c.operators), e, c.operators, angles))

    out, n_better, n_fewer = [], 0, 0
    for g, lab in labels.items():
        if g not in recs:
            out.append(lab)
            continue
        e_t = ev[g].energy(lab.operators, lab.angles)
        cx_t = cnot_estimate(a.system, lab.operators)
        tol = a.tol / 1000
        ok = [c for c in cands.get(g, []) if c[1] < e_t - tol or (c[1] <= e_t + tol and c[0] < cx_t)]
        if not ok:
            out.append(lab)
            continue
        cx, e, ops, angles = min(ok, key=lambda c: (c[0], c[1]))
        n_better += e < e_t - tol
        n_fewer += cx < cx_t
        out.append(GeneratedCircuit(g, a.system, ops, angles, True, source=f"distill_{a.replay}", meta={
            "energy": e, "error_mha": 1000 * (e - recs[g].fci_energy), "teacher_energy": e_t,
            "cnot_estimate": cx, "teacher_cnot_estimate": cx_t, "polished": a.polish}))
    path = Path(a.out) if a.out else ROOT / "results" / a.system / "distill_targets.jsonl"
    dump_jsonl(out, path)
    print(f"{a.system}: {len(recs)} train labels, replaced {sum(c.source.startswith('distill') for c in out)} "
          f"({n_better} lower energy, {n_fewer} fewer CX) -> {path}")


if __name__ == "__main__":
    main()
