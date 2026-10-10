import copy

import numpy as np
import torch

from genqai.circuits import build_circuit, gate_stats
from genqai.data import load_hamiltonians, load_teacher
from genqai.features import fit_featurizer
from genqai.model import build_policy
from genqai.rl import (GRPOConfig, GRPOTrainer, ReplayDB, RewardConfig, RewardModel, budget_cond, cnot_estimate,
                       greedy_eval, group_advantages)


def test_cnot_estimate_matches_compiled_counts():
    recs = load_hamiltonians("lih4")[:5]
    teacher = load_teacher("lih4")
    for r in recs:
        c = teacher[r.geometry_id]
        assert cnot_estimate("lih4", c.operators) == gate_stats(build_circuit(r, c.operators, c.angles))["cnot_count"]


def test_reward_and_cache():
    r = load_hamiltonians("lih4", "train")[0]
    c = load_teacher("lih4")[r.geometry_id]
    rm = RewardModel([r], RewardConfig(lam_cx=0.1), {r.geometry_id: 172.0})
    d = rm(r.geometry_id, c.operators, c.angles, True)
    assert abs(d["error_mha"]) < 0.01 and d["cx"] == 172
    assert np.isclose(d["reward"], -0.1, atol=1e-3)
    rm(r.geometry_id, c.operators, c.angles, True)
    assert rm.hits == 1
    hf = rm(r.geometry_id, [], [], True, budget=100.0)
    assert np.isclose(hf["reward"], -1.0, atol=1e-3)  # HF energy -> normalised error 1, no CX
    over = rm(r.geometry_id, c.operators, c.angles, False, budget=86.0)
    assert np.isclose(over["reward"], d["reward"] - 1.0 - 1.0, atol=1e-3)  # hinge (172-86)/86 + invalid


def test_group_advantages_and_replay():
    a = group_advantages(np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0]), 3)
    assert np.allclose(a[:3].mean(), 0) and a[2] > 0 and np.allclose(a[3:], 0)
    db = ReplayDB()
    db.add("g", [1], [0.1], -1.0, 10)
    db.add("g", [1, 2], [0.1, 0.2], -1.5, 20)
    db.add("g", [3], [0.3], -0.5, 15)  # dominated
    db.add("g", [4], [0.3], -1.2, 5)  # dominates the first
    assert sorted((p["energy"], p["cx"]) for p in db.front["g"]) == [(-1.5, 20), (-1.2, 5)]


def test_grpo_step_runs_and_respects_budget_rows():
    torch.manual_seed(0)
    recs = load_hamiltonians("lih4", "train")[:2]
    policy = build_policy("lih4", fit_featurizer("lih4"), d_model=32, n_layers=1, d_ff=64, max_len=12)
    cond = budget_cond(policy, recs, [None, 100])
    assert torch.all(cond.budget[0] == 0) and cond.budget[1, 0] == 1
    rm = RewardModel(recs, RewardConfig(), {r.geometry_id: 172.0 for r in recs})
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    tr = GRPOTrainer(policy, copy.deepcopy(policy), rm, GRPOConfig(group=4, minibatch=4), opt)
    stats = tr.step(recs, budgets=[80.0, 120.0])
    assert np.isfinite(stats["loss"]) and set(tr.replay.front) <= {r.geometry_id for r in recs}
    v = greedy_eval(policy, rm, recs, [80.0, 120.0])
    assert len(v["circuits"]) == 2 and 0 <= v["within_budget"] <= 1
