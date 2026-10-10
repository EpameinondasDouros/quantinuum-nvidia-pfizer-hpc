"""Non-ML-architecture angle baselines on the canonical labels (Workstream B4, "is the transformer needed?").

Both predict a complete circuit with ZERO energy evaluations at inference: a canonical operator
sequence from the TRAIN canonical labels (data/<system>/teacher_canonical.jsonl) and its angles as a
smooth function of the geometry. Angles are regressed in the orbital-phase-fixed frame of
genqai.features (theta_canonical = s_k * theta_original), exactly like the transformer.

  angle_regression  kernel ridge regression (RBF) on the same standardised, gauge-fixed Hamiltonian
                    coefficient features the transformer is conditioned on. One regressor per TRAIN
                    canonical sequence; the sequence of a new geometry is either the TRAIN majority
                    or that of its nearest TRAIN neighbour in feature space. alpha, gamma and the
                    sequence rule are chosen on VAL one-shot energy error only (VAL Hamiltonians and
                    FCI energies; VAL labels are never read).
  angle_interp      1-D systems only: piecewise-linear interpolation of each canonical angle in the
                    bond length over TRAIN geometries, linear extrapolation beyond the end segments.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
from scipy.interpolate import interp1d
from sklearn.kernel_ridge import KernelRidge

from genqai.adapt import wrap_angle
from genqai.features import Featurizer
from genqai.schema import GeneratedCircuit, HamiltonianRecord
from genqai.sim import Evaluator

ALPHAS = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
GAMMAS = tuple(np.logspace(-3, 1, 9))  # multiplied by 1 / feature dim
SEQ_RULES = ("majority", "nearest")


def _check_train(records: list[HamiltonianRecord]) -> None:
    if any(r.split != "train" for r in records):
        raise ValueError("angle baselines may only be fitted on training-split geometries")


def canonical_frame_angles(featurizer: Featurizer, records, bank) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """[(ops, gauge-fixed angles)] of the bank labels of `records` (theta_canonical = s * theta)."""
    signs = featurizer.op_signs(records)
    out = []
    for r, s in zip(records, signs):
        c = bank[r.geometry_id]
        out.append((tuple(c.operators), s[c.operators] * np.asarray(c.angles, float)))
    return out


def _to_circuit(record, featurizer, ops, canon_angles, source, **meta) -> GeneratedCircuit:
    s = featurizer.op_signs([record])[0]
    angles = [wrap_angle(a) for a in s[list(ops)] * canon_angles]
    return GeneratedCircuit(record.geometry_id, record.system, list(ops), angles, source=source, meta=meta)


class AngleRegressor:
    """KRR angle_regression. fit() reads TRAIN labels and VAL Hamiltonians only."""

    def __init__(self, featurizer: Featurizer):
        self.featurizer = featurizer

    def _fit_config(self, alpha: float, gamma: float):
        models = {}
        for seq, idx in self.seq_index.items():
            Y = np.stack([self.Y[i] for i in idx])
            mean = Y.mean(0)
            m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma / self.X.shape[1]).fit(self.X[idx], Y - mean)
            models[seq] = (m, mean)
        return models

    def _predict(self, models, rule, x: np.ndarray):
        if rule == "majority":
            seq = self.majority
        else:
            seq = self.seqs[int(np.argmin(np.linalg.norm(self.X - x, axis=1)))]
        m, mean = models[seq]
        return seq, m.predict(x[None])[0] + mean

    def fit(self, train_records, train_canon_bank, val_records) -> "AngleRegressor":
        _check_train(train_records)
        if any(r.split != "val" for r in val_records):
            raise ValueError("hyperparameters are selected on validation-split geometries only")
        labels = canonical_frame_angles(self.featurizer, train_records, train_canon_bank)
        self.X = self.featurizer(train_records).astype(float)
        self.seqs = [s for s, _ in labels]
        self.Y = [y for _, y in labels]
        self.seq_index = {s: [i for i, t in enumerate(self.seqs) if t == s] for s in set(self.seqs)}
        self.majority = Counter(self.seqs).most_common(1)[0][0]
        Xv = self.featurizer(val_records).astype(float)
        evs = [Evaluator(r) for r in val_records]
        self.val_scores = {}
        for alpha, gamma in itertools.product(ALPHAS, GAMMAS):
            models = self._fit_config(alpha, gamma)
            for rule in SEQ_RULES:
                errs = []
                for r, ev, x in zip(val_records, evs, Xv):
                    seq, y = self._predict(models, rule, x)
                    c = _to_circuit(r, self.featurizer, seq, y, "angle_regression")
                    errs.append(abs(1000 * (ev.energy(c.operators, c.angles) - r.fci_energy)))
                self.val_scores[(alpha, gamma, rule)] = float(np.mean(errs))
        self.alpha, self.gamma, self.rule = min(self.val_scores, key=self.val_scores.get)
        self.models = self._fit_config(self.alpha, self.gamma)
        return self

    def predict(self, record: HamiltonianRecord) -> GeneratedCircuit:
        seq, y = self._predict(self.models, self.rule, self.featurizer([record])[0].astype(float))
        return _to_circuit(record, self.featurizer, seq, y, "angle_regression",
                           alpha=float(self.alpha), gamma=float(self.gamma), seq_rule=self.rule,
                           val_mean_abs_err_mha=self.val_scores[(self.alpha, self.gamma, self.rule)])


class AngleInterpolator:
    """angle_interp for 1-D systems: linear in the single geometry parameter over TRAIN labels of the
    majority canonical sequence."""

    def __init__(self, featurizer: Featurizer):
        self.featurizer = featurizer

    def fit(self, train_records, train_canon_bank) -> "AngleInterpolator":
        _check_train(train_records)
        if len(train_records[0].geometry_params) != 1:
            raise ValueError("angle_interp is defined for 1-D geometry families only")
        labels = canonical_frame_angles(self.featurizer, train_records, train_canon_bank)
        self.seq = Counter(s for s, _ in labels).most_common(1)[0][0]
        pts = sorted((next(iter(r.geometry_params.values())), y)
                     for r, (s, y) in zip(train_records, labels) if s == self.seq)
        x, Y = np.array([p for p, _ in pts]), np.stack([y for _, y in pts])
        self.f = interp1d(x, Y, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
        return self

    def predict(self, record: HamiltonianRecord) -> GeneratedCircuit:
        r = next(iter(record.geometry_params.values()))
        return _to_circuit(record, self.featurizer, self.seq, self.f(r), "angle_interp")
