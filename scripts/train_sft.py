"""Supervised imitation of the ADAPT teacher (Workstream C).

Targets are complete teacher circuits (operators + angles) on TRAIN geometries: raw ADAPT, or
(default) the canonicalised ADAPT labels from scripts/make_canonical_teacher.py. Checkpoint
selection uses VAL one-shot accuracy: greedy generation -> exact energy -> mean |error| (mHa),
ties broken by VAL teacher-forced loss.

    python scripts/train_sft.py --system lih4

Fine-tuning variants (Workstream D): --init starts from a checkpoint (its featurizer and architecture);
--targets may be a .jsonl path (e.g. self-distillation labels from scripts/distill_targets.py);
--budget-targets adds extra, always budget-conditioned examples (e.g. canonical prefixes with
re-optimised angles from scripts/make_prefix_targets.py) and switches every budget vector to the
CX-only form [1, 0, cnot/500] used for budget-conditioned generation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from genqai.circuits import build_circuit, gate_stats
from genqai.data import DATA_DIR, load_hamiltonians, load_teacher
from genqai.features import budget_vector, fit_featurizer
from genqai.model import Cond, GQEPolicy, build_policy
from genqai.schema import GeneratedCircuit, load_jsonl
from genqai.sim import Evaluator

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--cond", default="hamiltonian", choices=["hamiltonian", "geometry"])
    p.add_argument("--no-gauge", action="store_true", help="disable orbital-phase gauge fixing")
    p.add_argument("--targets", default="teacher_canonical",
                   help="data/<system>/<targets>.jsonl: raw ADAPT 'teacher' or canonicalised labels "
                        "from scripts/make_canonical_teacher.py")
    p.add_argument("--init", default=None, help="fine-tune from this checkpoint instead of a fresh model")
    p.add_argument("--budget-targets", default=None,
                   help="data/<system>/<name>.jsonl of extra budget-conditioned targets (meta.cnot_count)")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    p.add_argument("--lambda-theta", type=float, default=3.0)
    p.add_argument("--coef-noise", type=float, default=0.0, help="Gaussian noise on standardised features")
    p.add_argument("--budget-prob", type=float, default=0.5, help="prob. of conditioning on the teacher's own budget")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n-mix", type=int, default=3)
    p.add_argument("--enc-blocks", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100, help="epochs")
    p.add_argument("--patience", type=int, default=0, help="evals without improvement before stopping (0=off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="checkpoint path (default checkpoints/<system>/sft.pt)")
    p.add_argument("--log", default=None, help="log path (default results/<system>/sft_train_log.json)")
    return p.parse_args(argv)


def oneshot_errors(policy, records) -> tuple[list[float], list[bool]]:
    """Greedy one-shot circuits -> exact energy error (mHa) per record."""
    gen = policy.sample(policy.condition(records), greedy=True)
    errs = [1000 * (Evaluator(r).energy(o, a) - r.fci_energy) for r, (o, a, _, _) in zip(records, gen)]
    return errs, [v for *_, v in gen]


def nll(policy, cond, ops, angles, lam):
    op_lp, op_m, ang_lp, ang_m = policy.token_log_probs(cond, ops, angles)
    ce = -op_lp.sum() / op_m.sum()
    ang = -ang_lp.sum() / ang_m.sum().clamp(min=1)
    return ce + lam * ang, ce, ang


def main(argv=None):
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)

    if a.targets.endswith(".jsonl"):
        teacher = {c.geometry_id: c for c in load_jsonl(a.targets, GeneratedCircuit)}
    else:
        teacher = load_teacher(a.system, a.targets)
    train = load_hamiltonians(a.system, "train")
    val = load_hamiltonians(a.system, "val")
    if a.init:
        policy = GQEPolicy.load(a.init).to(dev)
        feat = policy.featurizer
    else:
        feat = fit_featurizer(a.system, a.cond, gauge=not a.no_gauge)
        policy = build_policy(
            a.system, feat, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, d_ff=a.ffn,
            dropout=a.dropout, n_mix=a.n_mix, enc_blocks=a.enc_blocks,
        ).to(dev)

    # one example per train geometry (its target), plus optional extra budget-conditioned targets
    ex = list(range(len(train)))
    ops = [teacher[r.geometry_id].operators for r in train]
    angles = [teacher[r.geometry_id].angles for r in train]
    budgets = [budget_vector(None if a.budget_targets else len(o), gate_stats(build_circuit(r, o, t))["cnot_count"])
               for r, o, t in zip(train, ops, angles)]
    force = [False] * len(train)
    if a.budget_targets:
        row = {r.geometry_id: i for i, r in enumerate(train)}
        for c in load_jsonl(DATA_DIR / a.system / f"{a.budget_targets}.jsonl", GeneratedCircuit):
            if c.geometry_id in row:
                ex.append(row[c.geometry_id])
                ops.append(c.operators)
                angles.append(c.angles)
                budgets.append(budget_vector(None, c.meta["cnot_count"]))
                force.append(True)
    X = torch.from_numpy(feat(train)).to(dev)[ex]
    op_sign = torch.from_numpy(feat.op_signs(train)).to(dev)[ex]
    own_budget = torch.from_numpy(np.stack(budgets)).to(dev)
    force = torch.tensor(force, device=dev)
    val_cond = policy.condition(val)
    val_ops = [teacher[r.geometry_id].operators for r in val]
    val_angles = [teacher[r.geometry_id].angles for r in val]

    N = len(ex)
    steps_per_epoch = math.ceil(N / a.batch_size)
    total = a.epochs * steps_per_epoch
    warm = max(1, int(a.warmup * total))
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )

    log = {"args": vars(a), "n_train": len(train), "n_examples": N, "n_val": len(val), "evals": []}
    best, best_key, bad, t0 = None, None, 0, time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        policy.train()
        perm = rng.permutation(N)
        tot = []
        for i in range(0, N, a.batch_size):
            idx = perm[i : i + a.batch_size]
            x = X[idx]
            if a.coef_noise > 0:
                x = x + a.coef_noise * torch.randn_like(x)
            use = (torch.from_numpy(rng.random(len(idx)) < a.budget_prob).to(dev) | force[idx])[:, None]
            budget = torch.where(use, own_budget[idx], torch.zeros_like(own_budget[idx]))
            cond = Cond(x, budget, op_sign[idx])
            loss, ce, ang = nll(policy, cond, [ops[j] for j in idx], [angles[j] for j in idx], a.lambda_theta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            tot.append([loss.item(), ce.item(), ang.item()])
        if epoch % a.eval_every and epoch != a.epochs:
            continue
        policy.eval()
        with torch.no_grad():
            vl, vce, vang = (t.item() for t in nll(policy, val_cond, val_ops, val_angles, a.lambda_theta))
        errs, valid = oneshot_errors(policy, val)
        ab = np.abs(errs)
        rec = {
            "epoch": epoch, "lr": sched.get_last_lr()[0], "train": np.mean(tot, 0).tolist(),
            "val_loss": vl, "val_ce": vce, "val_angle_nll": vang,
            "val_mean_abs_mha": float(ab.mean()), "val_median_abs_mha": float(np.median(ab)),
            "val_max_abs_mha": float(ab.max()), "val_valid": float(np.mean(valid)),
            "elapsed_s": time.perf_counter() - t0,
        }
        log["evals"].append(rec)
        print(
            f"ep {epoch:5d} loss {rec['train'][0]:8.3f} ce {rec['train'][1]:.4f} | val loss {vl:8.3f} ce {vce:.3f} "
            f"| val |err| mean {rec['val_mean_abs_mha']:.3f} med {rec['val_median_abs_mha']:.3f} "
            f"max {rec['val_max_abs_mha']:.3f} mHa valid {rec['val_valid']:.2f} ({rec['elapsed_s']:.0f}s)",
            flush=True,
        )
        key = (round(rec["val_mean_abs_mha"], 3), vl)
        if best_key is None or key < best_key:
            best_key, best, bad = key, copy.deepcopy(policy.state_dict()), 0
            log["best"] = rec
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                print("early stop")
                break

    policy.load_state_dict(best)
    log["train_time_s"] = time.perf_counter() - t0
    out = Path(a.out) if a.out else ROOT / "checkpoints" / a.system / "sft.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.to("cpu").save(out, args=vars(a), best=log["best"])
    lp = Path(a.log) if a.log else ROOT / "results" / a.system / "sft_train_log.json"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(log, indent=1))
    print(f"best epoch {log['best']['epoch']} val mean |err| {log['best']['val_mean_abs_mha']:.3f} mHa -> {out}")


if __name__ == "__main__":
    main()
