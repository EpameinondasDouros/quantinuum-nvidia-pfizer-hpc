"""Canonicalised, continuation-smoothed SFT targets: data/<system>/teacher_canonical.jsonl.

Raw ADAPT labels are noisy as a function of geometry: near-degenerate gradients permute the
operator order, symmetric solutions flip angle signs, and BFGS lands in different basins for
neighbouring geometries. A model cannot interpolate that. This script relabels every geometry
with a complete, exactly re-optimised circuit that is chemically accurate AND varies smoothly:

1. Canonical order: operators sorted by (occurrence index, global rank), where the global rank is
   the mean first-occurrence position of each operator across TRAIN teacher sequences.
2. Continuation over TRAIN geometries along a minimum spanning tree in standardised geometry
   space (Prim order, rooted at the smallest or, with --start max, largest parameters): each geometry is warm-started from
   its nearest already-labelled TRAIN neighbour. Candidates, all BFGS-optimised exactly:
     A. neighbour's canonical sequence, neighbour's angles
     B. own ADAPT operators (canonical order), neighbour's angles matched by (op, occurrence)
     C. own ADAPT operators (canonical order), own ADAPT angles permuted accordingly
   Among candidates under --tol mHa, the fewest operators wins, ties in order A, B, C
   (--prefer neighbour: the first of A, B, C under --tol wins, giving larger contiguous regimes).
   If none qualifies, the raw ADAPT circuit is kept unchanged.
   Each BFGS minimises E(x) + mu |x - x_init|^2 (tiny mu): the energy has near-flat valleys in
   which plain BFGS drifts (e.g. sign flips of mixed-spin doubles), and the anchor keeps the
   relabelled angles continuous along the path. The stored energy is the true E(x).
   Neighbour angles are first mapped through the orbital-phase gauge (genqai.features), since
   PySCF MO signs can flip between neighbouring geometries.
Non-train geometries get the same treatment warm-started from their nearest TRAIN label (used
only for val loss and diagnostics; training never sees them).

    python scripts/make_canonical_teacher.py --system h4
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from genqai.adapt import wrap_angle
from genqai.data import DATA_DIR, load_hamiltonians, load_teacher
from genqai.features import fit_featurizer
from genqai.schema import GeneratedCircuit, dump_jsonl
from genqai.sim import Evaluator


def occurrence_keys(ops):
    seen = defaultdict(int)
    keys = []
    for o in ops:
        keys.append((o, seen[o]))
        seen[o] += 1
    return keys


def canonical_order(ops, rank):
    keys = occurrence_keys(ops)
    return sorted(range(len(ops)), key=lambda i: (keys[i][1], rank.get(ops[i], 1e9), i))


def anchored_minimize(ev, ops, init, mu, anchor=None):
    """argmin E(x) + mu * |x - anchor|^2 (BFGS from init, exact gradients; anchor defaults to init).
    Where the energy has flat valleys (redundant parameters), the anchor picks the minimiser closest
    to it, which keeps labels continuous. Returns (x, true energy E(x))."""
    x0 = np.array(init, float)
    xa = x0 if anchor is None else np.asarray(anchor, float)

    def f(x):
        e, g = ev.energy_and_grad(ops, x)
        d = x - xa
        return e + mu * d @ d, g + 2 * mu * d

    res = minimize(f, x0, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    return res.x, ev.energy(ops, res.x)


def relabel(rec, teacher, neighbour, rank, tol, mu, flip=None, prefer="short"):
    ev = Evaluator(rec)
    perm = canonical_order(teacher.operators, rank)
    own_ops = [teacher.operators[i] for i in perm]
    cands = []
    if neighbour is not None:
        nb_angles = [flip[o] * t for o, t in zip(neighbour.operators, neighbour.angles)]  # -> own gauge
        nb = dict(zip(occurrence_keys(neighbour.operators), nb_angles))
        cands.append(("A", neighbour.operators, nb_angles))
        cands.append(("B", own_ops, [nb.get(k, 0.0) for k in occurrence_keys(own_ops)]))
    cands.append(("C", own_ops, [teacher.angles[i] for i in perm]))
    best = None
    for tag, ops, init in cands:
        x, e = anchored_minimize(ev, list(ops), init, mu)
        err = 1000 * (e - rec.fci_energy)
        if err < tol and (best is None or (prefer == "short" and len(ops) < len(best[1]))):
            best = (tag, list(ops), [wrap_angle(t) for t in x], e, err)
    if best is None:
        tag, ops, angles = "raw", teacher.operators, teacher.angles
        e = ev.energy(ops, angles)
        best = (tag, ops, angles, e, 1000 * (e - rec.fci_energy))
    tag, ops, angles, e, err = best
    return GeneratedCircuit(
        geometry_id=rec.geometry_id, system=rec.system, operators=ops, angles=angles, valid=True,
        source="adapt_canonical",
        meta={"energy": e, "error_mha": err, "candidate": tag, "teacher_ops": teacher.operators},
    )


def smooth(labels, train, S, dist, sweeps, mu, tol):
    """Pull each TRAIN label towards the mean of its same-sequence TRAIN neighbours (within 1.6x its
    nearest-neighbour distance) by minimising E + mu |x - mean|^2; a move is kept only if the true
    error stays below tol. Gives the smoothest chemically accurate label field we can cheaply find."""
    ids = list(train)
    near = {}
    for g in ids:
        d = {h: dist(g, h) for h in ids if h != g}
        dmin = min(d.values())
        near[g] = [h for h, v in d.items() if v <= 1.6 * dmin + 1e-9]

    def target(g):
        nbs = [h for h in near[g] if labels[h].operators == labels[g].operators]
        if not nbs:
            return None
        return np.mean([[S[g][o] * S[h][o] * t for o, t in zip(labels[h].operators, labels[h].angles)] for h in nbs], 0)

    def roughness():
        r = [np.linalg.norm(np.array(labels[g].angles) - t) for g in ids if (t := target(g)) is not None]
        return float(np.mean(r))

    if sweeps:
        print(f"label roughness before smoothing {roughness():.4f} rad")
    for _ in range(sweeps):
        for g in ids:
            t = target(g)
            if t is None:
                continue
            lab, rec = labels[g], train[g]
            x, e = anchored_minimize(Evaluator(rec), lab.operators, lab.angles, mu, anchor=t)
            err = 1000 * (e - rec.fci_energy)
            if err < tol:
                lab.angles = [wrap_angle(v) for v in x]
                lab.meta.update(energy=e, error_mha=err, smoothed=True)
        print(f"  sweep roughness {roughness():.4f} rad")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--tol", type=float, default=1.0, help="accept a relabel only below this error (mHa)")
    p.add_argument("--mu", type=float, default=1e-4, help="anchor strength (Ha/rad^2)")
    p.add_argument("--prefer", default="short", choices=["short", "neighbour"])
    p.add_argument("--start", default="min", choices=["min", "max"], help="continuation root: smallest or largest geometry")
    p.add_argument("--smooth-sweeps", type=int, default=0, help="Laplacian smoothing sweeps over TRAIN labels")
    p.add_argument("--smooth-mu", type=float, default=1e-3, help="smoothing pull strength (Ha/rad^2)")
    p.add_argument("--out", default="teacher_canonical", help="output name under data/<system>/")
    a = p.parse_args(argv)

    recs = load_hamiltonians(a.system)
    teacher = load_teacher(a.system)
    train = [r for r in recs if r.split == "train"]
    pos = defaultdict(list)
    for r in train:
        for i, (o, occ) in enumerate(occurrence_keys(teacher[r.geometry_id].operators)):
            if occ == 0:
                pos[o].append(i)
    rank = {o: float(np.mean(v)) for o, v in pos.items()}

    S = dict(zip([r.geometry_id for r in recs], fit_featurizer(a.system).op_signs(recs)))
    P = {r.geometry_id: np.array(list(r.geometry_params.values())) for r in recs}
    tp = np.stack([P[r.geometry_id] for r in train])
    sd = tp.std(0) + 1e-12
    dist = lambda g, h: float(np.linalg.norm((P[g] - P[h]) / sd))  # noqa: E731

    labels: dict[str, GeneratedCircuit] = {}
    todo = sorted(train, key=lambda r: tuple(P[r.geometry_id]), reverse=a.start == "max")
    first = todo.pop(0)
    labels[first.geometry_id] = relabel(first, teacher[first.geometry_id], None, rank, a.tol, a.mu, prefer=a.prefer)
    while todo:  # Prim: attach the unlabelled train geometry closest to the labelled set
        j, nb = min(
            ((j, g) for j, r in enumerate(todo) for g in labels),
            key=lambda jg: dist(todo[jg[0]].geometry_id, jg[1]),
        )
        r = todo.pop(j)
        labels[r.geometry_id] = relabel(r, teacher[r.geometry_id], labels[nb], rank, a.tol, a.mu, S[r.geometry_id] * S[nb], a.prefer)
    train_ids = list(labels)
    smooth(labels, {r.geometry_id: r for r in train}, S, dist, a.smooth_sweeps, a.smooth_mu, a.tol)
    for r in recs:
        if r.split != "train":
            nb = min(train_ids, key=lambda g: dist(r.geometry_id, g))
            labels[r.geometry_id] = relabel(r, teacher[r.geometry_id], labels[nb], rank, a.tol, a.mu, S[r.geometry_id] * S[nb], a.prefer)

    out = [labels[r.geometry_id] for r in recs]
    dump_jsonl(out, DATA_DIR / a.system / f"{a.out}.jsonl")
    tags = defaultdict(int)
    for c in out:
        tags[c.meta["candidate"]] += 1
    errs = [c.meta["error_mha"] for c in out]
    print(f"{a.system}: {len(out)} labels, candidates {dict(tags)}, distinct seqs "
          f"{len({tuple(c.operators) for c in out})} (raw {len({tuple(t.operators) for t in teacher.values()})}), "
          f"max err {max(errs):.3f} mHa")


if __name__ == "__main__":
    main()
