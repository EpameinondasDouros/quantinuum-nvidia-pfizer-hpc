import numpy as np
import pytest
import torch

from genqai.data import load_hamiltonians, load_teacher
from genqai.features import fit_featurizer
from genqai.model import GQEPolicy, build_policy


@pytest.fixture(scope="module", params=["hamiltonian", "geometry"])
def setup(request):
    torch.manual_seed(0)
    recs = load_hamiltonians("lih4", "train")[:4]
    policy = build_policy("lih4", fit_featurizer("lih4", request.param), d_model=32, n_layers=2, d_ff=64, max_len=12)
    return policy.eval(), recs


def test_featurizer_uses_train_stats_only():
    f = fit_featurizer("lih4")
    x = f(load_hamiltonians("lih4", "train"))
    assert f.dim == len(set(f.keys)) and f.keys == sorted(f.keys)
    assert np.allclose(x.mean(0), 0, atol=1e-4)


def test_shapes(setup):
    policy, recs = setup
    teacher = load_teacher("lih4")
    ops = [teacher[r.geometry_id].operators for r in recs]
    angles = [teacher[r.geometry_id].angles for r in recs]
    op_lp, op_m, ang_lp, ang_m = policy.token_log_probs(policy.condition(recs), ops, angles)
    P = max(map(len, ops)) + 1
    assert op_lp.shape == op_m.shape == ang_lp.shape == (len(recs), P)
    assert int(op_m.sum()) == sum(map(len, ops)) + len(ops)  # ops + EOS
    assert int(ang_m.sum()) == sum(map(len, ops))
    lp = policy.log_prob(policy.condition(recs), ops, angles)
    assert lp.shape == (len(recs),) and lp.requires_grad


def test_sampled_logprob_matches_log_prob(setup):
    policy, recs = setup
    cond = policy.condition(recs[:1])
    torch.manual_seed(1)
    samples = policy.sample(cond, n=8) + policy.sample(cond, greedy=True)
    lp = policy.log_prob(cond, [s[0] for s in samples], [s[1] for s in samples])
    assert np.allclose(lp.detach().numpy(), [s[2] for s in samples], rtol=1e-4, atol=1e-3)
    for ops, angles, _, valid in samples:
        assert all(0 <= o < policy.cfg.n_ops for o in ops)
        assert len(ops) == len(angles) <= policy.cfg.max_len
        assert all(-np.pi <= a <= np.pi for a in angles)
        assert valid == (len(ops) < policy.cfg.max_len)


def test_batched_greedy_matches_single(setup):
    policy, recs = setup
    batch = policy.sample(policy.condition(recs), greedy=True)
    single = [policy.sample(policy.condition([r]), greedy=True)[0] for r in recs]
    for b, s in zip(batch, single):
        assert b[0] == s[0] and np.allclose(b[1], s[1], atol=1e-5)


def test_checkpoint_roundtrip(setup, tmp_path):
    policy, recs = setup
    policy.save(tmp_path / "p.pt", note="x")
    loaded = GQEPolicy.load(tmp_path / "p.pt")
    assert loaded.featurizer == policy.featurizer and loaded.extra == {"note": "x"}
    a = policy.sample(policy.condition(recs), greedy=True)
    b = loaded.sample(loaded.condition(recs), greedy=True)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert np.allclose([x[2] for x in a], [x[2] for x in b], atol=1e-4)


def test_gauge_fixing_is_exact_symmetry():
    """E_{H_fixed}(ops, theta_model) == E_H(ops, op_sign * theta_model)."""
    import dataclasses

    from genqai.sim import Evaluator

    f = fit_featurizer("h4")
    recs = load_hamiltonians("h4")
    signs = f.op_signs(recs)
    flipped = [i for i, s in enumerate(signs) if (s < 0).any()]
    assert flipped, "expected some h4 geometries in a non-reference orbital gauge"
    rng = np.random.default_rng(0)
    for i in flipped[:2]:
        r = recs[i]
        terms = [{"pauli": k, "coefficient": float(c)} for k, c in zip(f.keys, f.raw(r))]
        fixed = dataclasses.replace(r, pauli_terms=terms)
        ops = list(rng.integers(0, len(signs[i]), 10))
        theta = rng.uniform(-1, 1, 10)
        e_fixed = Evaluator(fixed).energy(ops, theta)
        e_orig = Evaluator(r).energy(ops, signs[i][ops] * theta)
        assert abs(e_fixed - e_orig) < 1e-10
