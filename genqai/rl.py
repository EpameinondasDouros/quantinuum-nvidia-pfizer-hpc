"""Workstream D: GRPO fine-tuning of the GQE policy against exact energies.

Reward for one sampled circuit c on Hamiltonian H (optionally under a CX budget B):

    R = -clip((E - E_FCI) / (E_HF - E_FCI + eps), 0, err_clip)
        - lam_cx * CX(c) / CX_ref(H)
        - budget_penalty * max(0, CX(c) - B) / B          (budget mode only)
        - invalid_penalty                                  (no EOS within max_len)

E is the exact statevector energy (genqai.sim.Evaluator, cached on the exact serialization of
the circuit). CX(c) is the ANALYTIC compiled CX count: the sum over operators of each pool
operator's own compiled CX count (``op_cnot_counts``, one Qiskit transpile per operator, cached).
Compiling a whole circuit can cancel a few CX between neighbouring excitations, so the analytic
count is an upper bound that matched full-circuit ``gate_stats`` within 0-4 CX (<0.5%) on every
teacher circuit we checked (h2/lih4/lih6 exact, h4 +2..+4 of ~900). Reported results always use
the exact ``gate_stats`` count via genqai.evaluate.score_circuit.

GRPO step: for each training Hamiltonian draw a group of G samples, normalise rewards within the
group (advantage = (R - mean) / (std + 1e-6)), then a few epochs of the PPO clipped objective at the
token level (token t = op_t plus its angle, or EOS), plus beta * KL(pi || pi_ref) to the frozen SFT
policy with the k3 estimator exp(r) - r - 1, r = log pi_ref - log pi. Dropout is off throughout (the
policy stays in eval mode) so sampling, old log-probs and updates see the same network.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from genqai.circuits import build_circuit, gate_stats
from genqai.data import load_hamiltonians
from genqai.features import budget_vector
from genqai.model import Cond, GQEPolicy
from genqai.pool import get_pool
from genqai.schema import GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator


def record_train_time(log: dict, own_s: float, results_dir, system: str, init_ckpt) -> None:
    """log["train_time_s"] = total wall time to produce a model: own_s + the init checkpoint's
    (results/<system>/<init stem>_train_log.json, itself cumulative), for break-even accounting."""
    p = Path(results_dir) / system / f"{Path(init_ckpt).stem}_train_log.json" if init_ckpt else None
    log["own_train_time_s"] = own_s
    log["init_train_time_s"] = json.loads(p.read_text()).get("train_time_s", 0.0) if p and p.exists() else 0.0
    log["train_time_s"] = own_s + log["init_train_time_s"]


# ------------------------------------------------------------------------------- CX accounting

@lru_cache(maxsize=None)
def op_cnot_counts(system: str) -> tuple[int, ...]:
    """Compiled CX count of each pool operator on its own (geometry independent)."""
    rec = load_hamiltonians(system)[0]
    return tuple(gate_stats(build_circuit(rec, [k], [0.1]))["cnot_count"] for k in range(len(get_pool(system))))


def cnot_estimate(system: str, ops) -> int:
    table = op_cnot_counts(system)
    return int(sum(table[k] for k in ops))


def budget_cond(policy: GQEPolicy, records: list[HamiltonianRecord], cnot_budgets=None) -> Cond:
    """Conditioning with a per-row CX budget (None entries / None list = unconstrained)."""
    cond = policy.condition(records)
    if cnot_budgets is not None:
        b = np.stack([budget_vector(None, None if v is None else int(v)) for v in cnot_budgets])
        cond.budget = torch.from_numpy(b).to(policy.device)
    return cond


# ------------------------------------------------------------------------------------- reward

@dataclass
class RewardConfig:
    lam_cx: float = 0.0
    budget_penalty: float = 1.0
    invalid_penalty: float = 1.0
    err_clip: float = 1.0
    eps: float = 1e-6


class RewardModel:
    """Exact-energy reward with a cache keyed on the exact circuit serialization."""

    def __init__(self, records: list[HamiltonianRecord], cfg: RewardConfig, cx_ref: dict[str, float]):
        self.records = {r.geometry_id: r for r in records}
        self.cfg, self.cx_ref = cfg, cx_ref
        self._ev: dict[str, Evaluator] = {}
        self.cache: dict[tuple, float] = {}
        self.hits = self.calls = 0

    def energy(self, gid: str, ops, angles) -> float:
        key = (gid, tuple(int(o) for o in ops), tuple(float(a) for a in angles))
        self.calls += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if gid not in self._ev:
            self._ev[gid] = Evaluator(self.records[gid])
        e = self.cache[key] = self._ev[gid].energy(list(key[1]), np.array(key[2]))
        return e

    def __call__(self, gid: str, ops, angles, valid: bool, budget: float | None = None) -> dict:
        r, c = self.records[gid], self.cfg
        e = self.energy(gid, ops, angles)
        cx = cnot_estimate(r.system, ops)
        err_n = min(max((e - r.fci_energy) / (r.hf_energy - r.fci_energy + c.eps), 0.0), c.err_clip)
        R = -err_n - c.lam_cx * cx / self.cx_ref[gid]
        if budget is not None:
            R -= c.budget_penalty * max(0.0, cx - budget) / budget
        if not valid:
            R -= c.invalid_penalty
        return {"reward": R, "energy": e, "error_mha": 1000 * (e - r.fci_energy), "cx": cx, "valid": valid}


# ------------------------------------------------------------------------------------- replay

@dataclass
class ReplayDB:
    """Per training Hamiltonian: the Pareto set of valid circuits over (energy, analytic CX)."""

    cap: int = 32
    front: dict[str, list[dict]] = field(default_factory=dict)

    def add(self, gid: str, ops, angles, energy: float, cx: int, source: str = "") -> None:
        cur = self.front.setdefault(gid, [])
        if any(p["energy"] <= energy and p["cx"] <= cx for p in cur):
            return
        cur[:] = [p for p in cur if not (energy <= p["energy"] and cx <= p["cx"])]
        cur.append({"energy": energy, "cx": cx, "ops": list(ops), "angles": [float(a) for a in angles], "source": source})
        cur.sort(key=lambda p: p["cx"])
        if len(cur) > self.cap:  # keep the extremes, thin the middle
            del cur[len(cur) // 2]

    def circuits(self, system: str) -> list[GeneratedCircuit]:
        return [GeneratedCircuit(gid, system, p["ops"], p["angles"], True, source=p["source"],
                                 meta={"energy": p["energy"], "cnot_estimate": p["cx"]})
                for gid, ps in self.front.items() for p in ps]


# --------------------------------------------------------------------------------------- GRPO

@dataclass
class GRPOConfig:
    group: int = 8
    temperature: float = 1.0
    clip: float = 0.2
    beta_kl: float = 0.04
    ppo_epochs: int = 2
    minibatch: int = 64
    grad_clip: float = 1.0


def token_logp(policy: GQEPolicy, cond: Cond, ops, angles):
    """Per-position log-prob (op or EOS + its angle) [B,P] and mask [B,P]."""
    op_lp, op_m, ang_lp, _ = policy.token_log_probs(cond, ops, angles)
    return op_lp + ang_lp, op_m.float()


def group_advantages(rewards: np.ndarray, group: int) -> np.ndarray:
    R = rewards.reshape(-1, group)
    return ((R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-6)).reshape(-1)


def _rows(cond: Cond, idx) -> Cond:
    return Cond(cond.x[idx], cond.budget[idx], cond.op_sign[idx])


class GRPOTrainer:
    def __init__(self, policy: GQEPolicy, ref: GQEPolicy, reward: RewardModel, cfg: GRPOConfig,
                 opt: torch.optim.Optimizer, replay: ReplayDB | None = None):
        self.policy, self.ref, self.reward, self.cfg, self.opt = policy.eval(), ref.eval(), reward, cfg, opt
        self.replay = replay if replay is not None else ReplayDB()
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def step(self, records: list[HamiltonianRecord], budgets: list[float | None] | None = None) -> dict:
        cfg, G = self.cfg, self.cfg.group
        rows = [r for r in records for _ in range(G)]
        row_budgets = None if budgets is None else [b for b in budgets for _ in range(G)]
        cond = budget_cond(self.policy, rows, row_budgets)
        samples = self.policy.sample(cond, temperature=cfg.temperature)
        ops = [s[0] for s in samples]
        angles = [s[1] for s in samples]
        info = [self.reward(r.geometry_id, o, a, s[3], None if row_budgets is None else row_budgets[i])
                for i, (r, o, a, s) in enumerate(zip(rows, ops, angles, samples))]
        for r, o, a, d in zip(rows, ops, angles, info):
            if d["valid"]:
                self.replay.add(r.geometry_id, o, a, d["energy"], d["cx"], "grpo")
        R = np.array([d["reward"] for d in info])
        adv = torch.from_numpy(group_advantages(R, G)).float().to(self.policy.device)

        with torch.no_grad():
            old, mask = token_logp(self.policy, cond, ops, angles)
            ref, _ = token_logp(self.ref, cond, ops, angles)
        n = len(rows)
        losses, kls, clipped = [], [], []
        active = (adv.abs() > 1e-8).nonzero().squeeze(-1).tolist()  # zero-advantage groups only add KL
        for _ in range(cfg.ppo_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, cfg.minibatch):
                idx = perm[i: i + cfg.minibatch]
                c = _rows(cond, torch.as_tensor(idx, device=self.policy.device))
                o_mb, a_mb = [ops[j] for j in idx], [angles[j] for j in idx]
                new, m = token_logp(self.policy, c, o_mb, a_mb)
                P = new.shape[1]
                lr = (new - old[idx, :P]).clamp(-20, 20)
                ratio = lr.exp()
                A = adv[idx].unsqueeze(1)
                surr = torch.minimum(ratio * A, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * A)
                d = ref[idx, :P] - new
                k3 = (d.clamp(max=20).exp() - d - 1)
                tok = (-surr + cfg.beta_kl * k3) * m
                loss = tok.sum() / m.sum()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
                self.opt.step()
                losses.append(loss.item())
                kls.append(float((k3.detach() * m).sum() / m.sum()))
                clipped.append(float((((ratio.detach() - 1).abs() > cfg.clip).float() * m).sum() / m.sum()))
        err = np.array([abs(d["error_mha"]) for d in info])
        return {
            "loss": float(np.mean(losses)), "kl": float(np.mean(kls)), "clip_frac": float(np.mean(clipped)),
            "reward_mean": float(R.mean()), "reward_std_in_group": float(R.reshape(-1, G).std(1).mean()),
            "sample_median_abs_mha": float(np.median(err)), "sample_valid": float(np.mean([d["valid"] for d in info])),
            "sample_mean_cx": float(np.mean([d["cx"] for d in info])),
            "sample_mean_len": float(np.mean([len(o) for o in ops])), "active_frac": len(active) / n,
            "cache_hit_rate": self.reward.hits / max(1, self.reward.calls),
        }


VAL_FRACTIONS = (0.5, 0.7, 0.85, 1.0)


def budget_val_set(records, cx_ref: dict[str, float], fractions=VAL_FRACTIONS):
    """(records, budgets): every record at each fraction of its reference CX (budget-mode validation)."""
    return ([r for _ in fractions for r in records],
            [f * cx_ref[r.geometry_id] for f in fractions for r in records])


def greedy_eval(policy: GQEPolicy, reward: RewardModel, records, budgets=None) -> dict:
    """Greedy one-shot circuits scored with the reward (analytic CX). For checkpoint selection."""
    gen = policy.sample(budget_cond(policy, records, budgets), greedy=True)
    info = [reward(r.geometry_id, o, a, v, None if budgets is None else budgets[i])
            for i, (r, (o, a, _, v)) in enumerate(zip(records, gen))]
    err = np.array([abs(d["error_mha"]) for d in info])
    return {
        "score": float(np.mean([d["reward"] for d in info])),
        "mean_abs_mha": float(err.mean()), "median_abs_mha": float(np.median(err)), "max_abs_mha": float(err.max()),
        "frac_under_1.6": float(np.mean(err <= 1.6)), "mean_cx": float(np.mean([d["cx"] for d in info])),
        "valid": float(np.mean([d["valid"] for d in info])),
        "within_budget": (float(np.mean([d["cx"] <= b for d, b in zip(info, budgets)]))
                          if budgets is not None else math.nan),
        "circuits": [(r.geometry_id, o, a) for r, (o, a, _, _) in zip(records, gen)],
    }
