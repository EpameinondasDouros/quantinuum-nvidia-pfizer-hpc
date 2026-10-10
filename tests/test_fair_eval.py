"""B4 fair-evaluation guards: fresh test set integrity and train-only access of the learned baselines."""
import numpy as np
import pytest

from genqai.baselines import gauge_map, load_train_set, nearest_train, run_method
from genqai.data import (FRESH_MIN_GRID_DIST, FRESH_SPLIT, fresh_geometries, fresh_regime, geometry_grid,
                         hamiltonians_path, load_hamiltonians, load_teacher)
from genqai.features import fit_featurizer
from genqai.regression import AngleInterpolator, AngleRegressor
from genqai.sim import Evaluator

SYSTEMS = ["h2", "lih4", "lih6", "h4"]


class _LoggingBank(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.accessed = set()

    def __getitem__(self, k):
        self.accessed.add(k)
        return super().__getitem__(k)


def _all_label_bank(system, variant):
    """A bank holding EVERY grid geometry's label (train, val and test), logging which ones are read."""
    return _LoggingBank(load_teacher(system, variant))


@pytest.mark.parametrize("system", SYSTEMS)
def test_fresh_geometries_are_new_and_fixed(system):
    grid = np.array(geometry_grid(system))
    geoms = fresh_geometries(system)
    assert geoms == fresh_geometries(system)  # deterministic (fixed seed)
    P = np.array([p for p, _ in geoms])
    assert min(np.linalg.norm(grid - p, axis=1).min() for p in P) >= FRESH_MIN_GRID_DIST
    assert len({p for p, _ in geoms}) == len(geoms)
    if not hamiltonians_path(system, fresh=True).exists():
        pytest.skip("fresh Hamiltonians not built")
    fresh = load_hamiltonians(system, FRESH_SPLIT)
    assert [tuple(h.geometry_params.values()) for h in fresh] == [p for p, _ in geoms]
    assert [fresh_regime(system, h.geometry_params) for h in fresh] == [r for _, r in geoms]
    assert {h.split for h in fresh} == {FRESH_SPLIT}
    assert not {h.geometry_id for h in load_hamiltonians(system)} & {h.geometry_id for h in fresh}
    # existing loaders are unchanged: the fresh set only comes back when asked for explicitly
    assert all(h.split != FRESH_SPLIT for h in load_hamiltonians(system) + load_hamiltonians(system, "test"))
    # the frozen featurizer's Pauli vocabulary covers every fresh term (no silently dropped features)
    keys = set(fit_featurizer(system).keys)
    assert all(t["pauli"] in keys for h in fresh for t in h.pauli_terms)


@pytest.mark.parametrize("system", ["lih4", "h4"])
def test_gauge_nn_reads_only_train_labels(system):
    train = load_train_set(system)
    train_ids = {r.geometry_id for r in train.records}
    train.bank, train.canon_bank = _all_label_bank(system, "teacher"), _all_label_bank(system, "teacher_canonical")
    recs = load_hamiltonians(system)[::4]
    if hamiltonians_path(system, fresh=True).exists():
        recs += load_hamiltonians(system, FRESH_SPLIT)[::4]
    for rec in recs:
        for m in ("nn_reuse_gauge", "nn_reuse_canon"):
            nb = run_method(m, rec, train).extra["neighbor_geometry_id"]
            assert nb in train_ids and nb != rec.geometry_id
    assert train.bank.accessed and train.bank.accessed <= train_ids
    assert train.canon_bank.accessed and train.canon_bank.accessed <= train_ids


def test_gauge_map_identity_and_h4_sign_flips_exist():
    train = load_train_set("h4")
    canon = load_teacher("h4", "teacher_canonical")
    r = train.records[0]
    assert gauge_map(train.featurizer, r, r, canon[r.geometry_id]) == pytest.approx(canon[r.geometry_id].angles)
    flips = sum(not np.array_equal(*train.featurizer.op_signs([rec, nearest_train(rec, train.records)]))
                for rec in train.records)
    assert flips > 0  # neighbouring h4 geometries do differ in orbital-sign convention


def test_angle_baselines_use_train_labels_and_val_hamiltonians_only():
    system = "lih4"
    train = load_train_set(system)
    train_ids = {r.geometry_id for r in train.records}
    bank = _all_label_bank(system, "teacher_canonical")
    reg = AngleRegressor(train.featurizer).fit(train.records, bank, train.val_records)
    assert bank.accessed == train_ids  # every train label used, no val/test label read
    test = load_hamiltonians(system, "test")
    for r in test[:3]:
        c = reg.predict(r)
        assert Evaluator(r).energy(c.operators, c.angles) > r.fci_energy - 1e-9
    assert bank.accessed == train_ids
    with pytest.raises(ValueError):
        AngleRegressor(train.featurizer).fit(train.records + test[:1], bank, train.val_records)
    with pytest.raises(ValueError):  # hyperparameters may only be selected on VAL
        AngleRegressor(train.featurizer).fit(train.records, bank, test[:2])
    interp_bank = _all_label_bank(system, "teacher_canonical")
    AngleInterpolator(train.featurizer).fit(train.records, interp_bank).predict(test[0])
    assert interp_bank.accessed == train_ids
    with pytest.raises(ValueError):
        AngleInterpolator(train.featurizer).fit(train.val_records, interp_bank)
