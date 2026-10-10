"""GRPO fine-tuning from the SFT checkpoint (Workstream D1 / D3). See genqai/rl.py for the objective.

    python -m scripts.train_grpo --system h4                                  # D1: energy only
    python -m scripts.train_grpo --system h4 --family budget --init checkpoints/h4/budget_sft.pt \
        --budget 0.4 1.1                                                      # D3: CX-budget conditioned

Every iteration uses all TRAIN Hamiltonians (or --records-per-step of them). Checkpoint selection on
VAL only: mean greedy one-shot reward (budget mode: averaged over --val-fractions of each VAL
geometry's reference CX). Iteration 0 (the initial checkpoint) is a candidate, so a run that never
improves VAL returns the initial model. Test geometries are never touched.

Writes checkpoints/<system>/<family>.pt, results/<system>/<family>_train_log.json and
results/<system>/<family>_replay.jsonl (Pareto set of valid sampled circuits per TRAIN geometry).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from genqai.data import load_hamiltonians, load_teacher
from genqai.model import GQEPolicy
from genqai.rl import (GRPOConfig, GRPOTrainer, ReplayDB, RewardConfig, RewardModel, cnot_estimate, greedy_eval,
                       record_train_time)
from genqai.schema import dump_jsonl

ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True)
    p.add_argument("--family", default="grpo", help="output name: checkpoints/<system>/<family>.pt etc.")
    p.add_argument("--init", default=None, help="start checkpoint (default checkpoints/<system>/sft.pt)")
    p.add_argument("--ref", default=None, help="KL reference checkpoint (default: --init)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--beta-kl", type=float, default=0.04)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--records-per-step", type=int, default=0, help="0 = all train Hamiltonians")
    p.add_argument("--lam-cx", type=float, default=0.0, help="lambda: penalty on CX / reference CX")
    p.add_argument("--budget", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="budget mode: per-group CX budget ~ U(LO, HI) x reference CX")
    p.add_argument("--budget-penalty", type=float, default=1.0)
    p.add_argument("--val-fractions", type=float, nargs="+", default=[0.5, 0.7, 0.85, 1.0])
    p.add_argument("--invalid-penalty", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    init = Path(a.init) if a.init else ROOT / "checkpoints" / a.system / "sft.pt"
    policy = GQEPolicy.load(init)
    ref = GQEPolicy.load(a.ref or init)
    train = load_hamiltonians(a.system, "train")
    val = load_hamiltonians(a.system, "val")
    labels = load_teacher(a.system, "teacher_canonical")
    cx_ref = {r.geometry_id: float(cnot_estimate(a.system, labels[r.geometry_id].operators)) for r in train + val}
    reward = RewardModel(train + val, RewardConfig(lam_cx=a.lam_cx, budget_penalty=a.budget_penalty,
                                                   invalid_penalty=a.invalid_penalty), cx_ref)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)
    trainer = GRPOTrainer(policy, ref, reward, GRPOConfig(a.group, a.temperature, a.clip, a.beta_kl,
                                                          a.ppo_epochs, a.minibatch), opt, ReplayDB())
    for r in train:  # seed the replay DB with the teacher labels so "best" is never worse than them
        c = labels[r.geometry_id]
        trainer.replay.add(r.geometry_id, c.operators, c.angles, reward.energy(r.geometry_id, c.operators, c.angles),
                           cnot_estimate(a.system, c.operators), "teacher_canonical")

    if a.budget:
        val_recs = [r for f in a.val_fractions for r in val]
        val_budgets = [f * cx_ref[r.geometry_id] for f in a.val_fractions for r in val]
    else:
        val_recs, val_budgets = val, None

    def evaluate(it, stats, t0):
        v = greedy_eval(policy, reward, val_recs, val_budgets)
        v.pop("circuits")
        rec = {"epoch": it, "train": [stats.get("loss", float("nan")), stats.get("reward_mean", float("nan"))],
               **{f"val_{k}": x for k, x in v.items()}, **stats, "elapsed_s": time.perf_counter() - t0}
        print(f"it {it:4d} R {rec['train'][1]:+.4f} kl {stats.get('kl', 0):.4f} clip {stats.get('clip_frac', 0):.3f} "
              f"| smp med {stats.get('sample_median_abs_mha', float('nan')):.3f} mHa cx {stats.get('sample_mean_cx', float('nan')):.0f} "
              f"len {stats.get('sample_mean_len', float('nan')):.1f} | VAL score {v['score']:+.5f} |err| mean "
              f"{v['mean_abs_mha']:.3f} max {v['max_abs_mha']:.3f} cx {v['mean_cx']:.0f} "
              f"inbudget {v['within_budget']:.2f} ({rec['elapsed_s']:.0f}s)", flush=True)
        return rec

    log = {"args": vars(a), "n_train": len(train), "n_val": len(val), "evals": []}
    t0 = time.perf_counter()
    best_rec = evaluate(0, {}, t0)
    log["evals"].append(best_rec)
    best = copy.deepcopy(policy.state_dict())
    for it in range(1, a.iters + 1):
        recs = train if not a.records_per_step else list(rng.choice(train, a.records_per_step, replace=False))
        budgets = None if not a.budget else [rng.uniform(*a.budget) * cx_ref[r.geometry_id] for r in recs]
        stats = trainer.step(recs, budgets)
        if it % a.eval_every and it != a.iters:
            continue
        rec = evaluate(it, stats, t0)
        log["evals"].append(rec)
        if rec["val_score"] > best_rec["val_score"]:
            best_rec, best = rec, copy.deepcopy(policy.state_dict())
    log["best"] = best_rec
    record_train_time(log, time.perf_counter() - t0, ROOT / "results", a.system, init)
    policy.load_state_dict(best)
    out = ROOT / "checkpoints" / a.system / f"{a.family}.pt"
    policy.save(out, args=vars(a), best=best_rec, init=str(init))
    (ROOT / "results" / a.system / f"{a.family}_train_log.json").write_text(json.dumps(log, indent=1))
    dump_jsonl(trainer.replay.circuits(a.system), ROOT / "results" / a.system / f"{a.family}_replay.jsonl")
    print(f"best it {best_rec['epoch']} VAL score {best_rec['val_score']:+.5f} mean |err| "
          f"{best_rec['val_mean_abs_mha']:.3f} mHa cx {best_rec['val_mean_cx']:.0f} -> {out} "
          f"({log['own_train_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
