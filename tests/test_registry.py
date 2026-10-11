"""Lane B: model registry (models/registry.json), routing (A/B/C/D) and accounted generation."""
import copy
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from genqai import generate as gen
from genqai import registry as reg
from genqai.chemistry import SYSTEMS, SystemSpec, build_record
from genqai.data import FRESH_SPLIT, load_hamiltonians
from genqai.features import fit_featurizer
from genqai.pool import get_pool
from genqai.sim import Evaluator
from scripts.build_registry import build_cards, file_sha256

SPEC_FIELDS = ("name", "molecule", "geometry_params", "active_electrons", "num_active_orbitals", "mapping", "basis")


@dataclass(frozen=True)
class FrameSpec(SystemSpec):
    """A preset system whose atoms are placed in another frame (rotation R, translation t)."""

    R: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    t: tuple = (0.0, 0.0, 0.0)

    def atoms(self, params):
        return [(s, tuple(np.array(self.R) @ np.array(x) + np.array(self.t))) for s, x in SystemSpec.atoms(self, params)]


def framed(system, R, t=(0.0, 0.0, 0.0)):
    return FrameSpec(*[getattr(SYSTEMS[system], f) for f in SPEC_FIELDS], R=tuple(map(tuple, R)), t=tuple(t))


Z_TO_X = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))  # molecule axis z -> x
GENERIC_R = tuple(map(tuple, np.linalg.qr(np.array([[0.3, -0.5, 0.81], [0.2, 0.9, 0.3], [-0.93, 0.0, 0.4]]))[0]))


def rec(system, split, i=0):
    return load_hamiltonians(system, split)[i]


def by_r(system, split, r):
    return next(x for x in load_hamiltonians(system, split) if abs(x.geometry_params["r"] - r) < 0.05)


# ----------------------------------------------------------------------------------------- registry

def test_registry_rebuild_is_deterministic():
    committed = json.loads(reg.REGISTRY_PATH.read_text())
    a, b = reg.registry_json(build_cards(["h2"])), reg.registry_json(build_cards(["h2"]))
    assert a == b
    rebuilt = json.loads(a)["cards"]
    assert rebuilt == [c for c in committed["cards"] if c["system"] == "h2"]

    def keys(x):
        return [k for k, v in x.items() for k in [k, *keys(v)]] if isinstance(x, dict) else \
            [k for v in x for k in keys(v)] if isinstance(x, list) else []

    assert not [k for k in keys(json.loads(a)) if "time" in k or "date" in k]  # no timestamps


def test_registry_cards_are_consistent():
    cards = reg.load_registry()
    kinds = {s: {c.kind for c in cards if c.system == s} for s in ("h2", "lih4", "lih6", "h4")}
    common = {"transformer_sft", "transformer_grpo", "krr", "nn_reuse_gauge"}
    assert all(common <= k for k in kinds.values())
    assert "transformer_budget" in kinds["h4"] and "transformer_budget" in kinds["lih6"]
    assert "angle_interp" in kinds["lih4"] and "angle_interp" not in kinds["h4"]
    for c in cards:
        if c.checkpoint:
            assert file_sha256(reg.ROOT / c.checkpoint) == c.checkpoint_sha256
        assert c.signature["pauli_vocab_sha256"] == reg.vocab_sha256(fit_featurizer(c.system).keys)
        assert c.validation["val"]["n"] == len(load_hamiltonians(c.system, "val"))
        assert c.validation["test_fresh"]["display_only"] is True
    assert {c.signature["formula"] for c in cards} == {"H2", "HLi", "H4"}


# ------------------------------------------------------------------------------------------ routing

def test_presets_route_A_inside_and_B_outside_training_range():
    d = reg.route(None, by_r("lih4", "val", 1.44), None)
    assert (d.path, d.system, d.params) == ("A", "lih4", (1.44,))
    assert d.recommended in d.available_methods and f"lih4/{d.recommended}" in d.compatible
    out = by_r("lih4", FRESH_SPLIT, 4.3)  # beyond the r = 3.56 Å TRAIN edge
    d = reg.route(None, out, None)
    assert d.path == "B" and not d.ood["in_range"]["r"] and d.ood["ratio"] > 1
    assert d.recommended.endswith("+refine")
    h4_out = max(load_hamiltonians("h4", FRESH_SPLIT), key=lambda x: x.geometry_params["b"])
    assert reg.route(None, h4_out, None).path == "B"
    assert reg.route(None, rec("h4", "test_interval"), None).path == "A"


def test_h2_typed_along_x_axis_maps_to_h2_family():
    x = build_record(framed("h2", Z_TO_X, (2.0, -1.0, 0.5)), (0.9,))
    assert all(abs(xyz[1] + 1.0) < 1e-12 and abs(xyz[2] - 0.5) < 1e-12 for _, xyz in x.coordinates)  # along x
    assert reg.match_family(x) == ("h2", (0.9,))
    d = reg.route(None, x, None)
    assert d.path == "A" and d.params == (0.9,)
    mol = SimpleNamespace(atoms=tuple((s, tuple(v)) for s, v in x.coordinates), charge=0, multiplicity=1, basis="sto-3g")
    assert reg.match_family(x, mol) == ("h2", (0.9,))
    assert reg.match_family(x, replace_mol(mol, charge=1)) is None


def replace_mol(mol, **kw):
    return SimpleNamespace(**{**vars(mol), **kw})


def test_linear_h4_family_any_frame_and_asymmetric_rejected():
    params = reg.family_params("linear_h4", [(s, np.array(v)) for s, v in framed("h4", GENERIC_R, (1, 2, 3)).atoms((1.1, 1.37))])
    assert params == pytest.approx((1.1, 1.37), abs=1e-9)
    asym = [("H", np.array([z, 0.0, 0.0])) for z in (0.0, 1.0, 2.4, 3.5)]
    assert reg.family_params("linear_h4", asym) is None
    bent = [("H", np.array(v)) for v in ((0, 0, 0), (1, 0, 0), (2, 0.3, 0), (3, 0.3, 0))]
    assert reg.family_params("linear_h4", bent) is None


def unrelated_vocab_record():
    r = copy.deepcopy(rec("h2", "val"))
    r.pauli_terms = sorted(r.pauli_terms + [{"pauli": "YY", "coefficient": 0.05}], key=lambda t: t["pauli"])
    return r


def test_unrelated_hamiltonians_route_to_C_and_learned_methods_refuse(tmp_path, monkeypatch):
    other = unrelated_vocab_record()
    d = reg.route(None, other, None)
    assert d.path == "C" and d.compatible == [] and d.available_methods == ["adapt", "uccsd_vqe"]
    assert any("outside the trained vocabulary" in s for s in d.reasons)
    for m in ("krr", "transformer_sft", "nn_reuse_gauge+refine"):
        with pytest.raises(gen.GenerationError, match="no compatible trained generator"):
            gen.generate(other, m, d)
    wrong_q = replace(rec("lih4", "val"), num_qubits=5)
    assert reg.route(None, wrong_q, None).path == "C"
    square = build_record(SYSTEMS["h4"], (1.0, 1.0))
    square.coordinates = [["H", list(v)] for v in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    assert reg.route(None, square, None).path == "C"
    # path C: conventional solver runs, is accounted, and is logged as a teacher candidate
    monkeypatch.setattr(gen, "TEACHER_CANDIDATES_DIR", tmp_path)
    c, log = gen.generate(other, "adapt", d)
    assert log.validity and log.forward_passes == 0 and log.adapt_pool_gradient_evals > 0
    assert log.quantum_evals_to_propose == c.meta["energy_evals"] + c.meta["pool_gradient_evals"]
    rows = (tmp_path / f"{other.geometry_id}.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["circuit"]["operators"] == c.operators


def test_path_D_executes_nothing():
    report = SimpleNamespace(errors=["40 active orbitals exceeds the limit of 12"], warnings=[], exact_feasible=False,
                             limits={"max_active_orbitals": 12})
    d = reg.route(None, None, report)
    assert d.path == "D" and d.available_methods == [] and d.suggestions
    with pytest.raises(gen.GenerationError, match="path D"):
        gen.generate(rec("h2", "val"), "adapt", d)


def test_recommendation_uses_val_metrics_not_test_fresh(monkeypatch):
    cards = reg.load_registry()
    h4 = [c for c in cards if c.system == "h4"]
    d = reg.route(None, rec("h4", "val"), None)
    assert d.recommended == min(h4, key=reg.rank_key).kind == "krr"
    # a perfect test_fresh score must not change the choice ...
    doctored = copy.deepcopy(cards)
    for c in doctored:
        if c.id == "h4/nn_reuse_gauge":
            c.validation["test_fresh"].update({"frac_under_1.6mha": 1.0, "median_abs_err_mha": 0.0, "max_abs_err_mha": 0.0})
    monkeypatch.setattr(reg, "load_registry", lambda: doctored)
    assert reg.route(None, rec("h4", "val"), None).recommended == "krr"
    # ... while a better VAL score does
    for c in doctored:
        if c.id == "h4/nn_reuse_gauge":
            c.validation["val"].update({"frac_under_1.6mha": 1.0, "median_abs_err_mha": 0.0, "max_abs_err_mha": 0.0})
    assert reg.route(None, rec("h4", "val"), None).recommended == "nn_reuse_gauge"


# --------------------------------------------------------------------------------------- generation

@pytest.mark.parametrize("system,method", [
    ("lih4", "transformer_sft"), ("lih4", "transformer_grpo"), ("lih4", "krr"), ("lih4", "nn_reuse_gauge"),
    ("lih4", "angle_interp"), ("lih6", "transformer_budget"), ("h4", "krr"), ("h4", "transformer_sft"),
])
def test_oneshot_methods_valid_and_zero_evals(system, method):
    r = rec(system, "val")
    d = reg.route(None, r, None)
    c, log = gen.generate(r, method, d)
    assert log.validity and c.valid and len(c.operators) == len(c.angles) > 0
    assert all(0 <= o < len(get_pool(system)) for o in c.operators)
    assert log.quantum_evals_to_propose == 0 and log.refine_energy_evals == 0 and log.candidates == 1
    assert log.forward_passes == (0 if method == "nn_reuse_gauge" else 1)
    assert log.model_id == f"{system}/{method}" and not log.experimental
    if method.startswith("transformer"):
        assert log.details["decoder_steps"] == len(c.operators) + 1
    if method == "transformer_budget":
        assert c.meta["cnot_budget"] == reg.card_for("lih6/transformer_budget").config["default_cnot_budget"]
    if system == "lih4":  # VAL error matches what the registry recorded
        assert abs(1000 * (Evaluator(r).energy(c.operators, c.angles) - r.fci_energy)) <= \
            reg.card_for(log.model_id).validation["val"]["max_abs_err_mha"] + 1e-6


def test_bestofk_refine_and_solver_accounting():
    r = rec("lih4", "val")
    d = reg.route(None, r, None)
    c, log = gen.generate(r, "bestofk", d, k=4)
    assert (log.forward_passes, log.candidates, log.quantum_evals_to_propose) == (4, 4, 4)
    assert log.refine_energy_evals == 0 and "best-of-4" in log.notes[0]
    c, log = gen.generate(r, "transformer_grpo+bestofk", d, k=3)
    assert log.quantum_evals_to_propose == 3 and c.source == "transformer_grpo+bestofk"
    c, log = gen.generate(r, "krr+refine", d)
    assert log.quantum_evals_to_propose == 0 and log.refine_energy_evals > 0 and c.source == "krr+refine"
    assert abs(log.details["refined_energy"] - r.fci_energy) < 1e-6
    c, log = gen.generate(r, "uccsd_vqe", d)
    assert c.operators == list(range(len(get_pool("lih4")))) and log.quantum_evals_to_propose > 0
    assert log.adapt_pool_gradient_evals == 0
    c, log = gen.generate(r, "auto", d)
    assert log.method == d.recommended
    with pytest.raises(gen.GenerationError):
        gen.generate(r, "krr+bestofk", d)


def test_path_B_runs_but_is_experimental():
    r = by_r("lih4", FRESH_SPLIT, 4.3)
    d = reg.route(None, r, None)
    c, log = gen.generate(r, "krr", d)
    assert d.path == "B" and log.experimental and any("EXPERIMENTAL" in n for n in log.notes)
    c, log = gen.generate(r, d.recommended, d)
    assert log.experimental and log.refine_energy_evals > 0
    assert abs(log.details["refined_energy"] - r.fci_energy) * 1000 < 1.6


def test_torch_free_path_equals_torch_path(monkeypatch):
    for system in ("h2", "lih4", "lih6", "h4"):
        for r in load_hamiltonians(system, "val")[:2] + load_hamiltonians(system, FRESH_SPLIT)[-1:]:
            d = reg.route(None, r, None)
            ct, _ = gen.generate(r, "transformer_sft", d, backend="torch")
            cn, ln = gen.generate(r, "transformer_sft", d, backend="numpy")
            assert cn.operators == ct.operators and cn.valid == ct.valid, r.geometry_id
            assert np.abs(np.angle(np.exp(1j * (np.array(cn.angles) - np.array(ct.angles))))).max(initial=0) < 1e-4
            ev = Evaluator(r)
            assert abs(ev.energy(cn.operators, cn.angles) - ev.energy(ct.operators, ct.angles)) * 1000 < 0.01
            assert ln.forward_passes == 1 and ln.quantum_evals_to_propose == 0
    monkeypatch.setattr(gen, "torch_available", lambda: False)
    monkeypatch.setattr(reg, "torch_available", lambda: False)
    r = rec("h4", "val")
    d = reg.route(None, r, None)
    assert "transformer_grpo" not in d.available_methods and "transformer_sft" in d.available_methods
    c, log = gen.generate(r, "transformer_sft", d)
    assert log.details["backend"].startswith("numpy")
    with pytest.raises(gen.GenerationError, match="torch"):
        gen.generate(r, "transformer_grpo", d)


# ---------------------------------------------------------------------------------------- the gauge

def test_gauge_rotated_frame_flips_orbital_signs_but_features_and_energies_agree():
    """H4 built in a rotated + translated frame: PySCF returns MOs with other signs (Hamiltonian
    coefficients flip sign), the gauge-fixed features are identical, and KRR / transformer circuits
    mapped back into each record's own sign convention give the same energy."""
    params = (1.1, 1.37)
    pre, rot = build_record(SYSTEMS["h4"], params), build_record(framed("h4", GENERIC_R, (1.3, -2.0, 0.7)), params)
    cp = {t["pauli"]: t["coefficient"] for t in pre.pauli_terms}
    cr = {t["pauli"]: t["coefficient"] for t in rot.pauli_terms}
    assert set(cp) == set(cr)
    flipped = [k for k in cp if abs(cp[k]) > 1e-8 and np.sign(cp[k]) != np.sign(cr[k])]
    assert flipped, "expected the rotated frame to change some orbital signs"
    assert np.allclose(sorted(map(abs, cp.values())), sorted(map(abs, cr.values())), atol=1e-9)
    f = fit_featurizer("h4")
    assert np.abs(f([rot]) - f([pre])).max() < 1e-6
    assert (f.op_signs([rot]) != f.op_signs([pre])).any()  # the gauge map is non-trivial here
    d_pre, d_rot = reg.route(None, pre, None), reg.route(None, rot, None)
    assert d_rot.path == "A" and d_rot.params == pytest.approx(params, abs=1e-6)
    for m in ("krr", "transformer_sft", "nn_reuse_gauge"):
        c_pre, _ = gen.generate(pre, m, d_pre)
        c_rot, _ = gen.generate(rot, m, d_rot)
        assert c_pre.operators == c_rot.operators
        e_pre = Evaluator(pre).energy(c_pre.operators, c_pre.angles)
        e_rot = Evaluator(rot).energy(c_rot.operators, c_rot.angles)
        assert abs(e_pre - e_rot) * 1000 < 1e-6, m
