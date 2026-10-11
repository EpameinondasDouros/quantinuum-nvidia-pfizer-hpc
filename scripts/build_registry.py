"""Build models/registry.json: one ModelCard per available generator per preset system (lane B).

    PYTHONPATH=$PWD python -m scripts.build_registry [--systems h2 lih4 lih6 h4] [--check]

Per card: checkpoint + sha256 (transformers; KRR / NN reuse / interpolation are refitted
deterministically from the TRAIN labels, so their card records the fit configuration and the label
file's sha256 instead of a pickle), the system signature, the geometry family with its TRAIN range and
feature-distance threshold (genqai.registry), VAL metrics computed here by generating every VAL
geometry with genqai.generate.propose and scoring it with genqai.evaluate.score_circuit (used for
default selection), and test_fresh metrics copied from results_fresh/ (display only).
The output is deterministic (no timestamps; floats rounded); --check fails if the file would change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from genqai.chemistry import SYSTEMS
from genqai.data import FRESH_SPLIT, fresh_regime, hamiltonians_path, load_hamiltonians, results_dir
from genqai.evaluate import score_circuit, summarize
from genqai.features import fit_featurizer
from genqai.generate import (TRANSFORMER_STAGES, checkpoint_path, krr_config, numpy_export_path, propose,
                             train_set)
from genqai.pool import get_pool
from genqai.registry import (DISTANCE_FACTOR, GAUGE_CONVENTION, REGISTRY_PATH, ModelCard, hill_formula,
                             loo_nn_distances, registry_json, sha256_text, vocab_sha256)
from genqai.schema import EvalRecord, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ORDER = ("h2", "lih4", "lih6", "h4")
FRESH_FILES = {"transformer_sft": "sft_oneshot", "transformer_grpo": "grpo_oneshot", "transformer_budget": "budget_oneshot",
               "krr": "angle_regression", "nn_reuse_gauge": "nn_reuse_gauge", "angle_interp": "angle_interp"}
DECIMALS = 6


def _r(x: float) -> float:
    return round(float(x), DECIMALS)


def file_sha256(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def available_kinds(system: str) -> list[str]:
    kinds = [k for k in TRANSFORMER_STAGES if checkpoint_path(system, k).exists()]
    kinds += ["krr", "nn_reuse_gauge"]
    return kinds + (["angle_interp"] if len(SYSTEMS[system].geometry_params) == 1 else [])


def system_signature(system: str) -> dict:
    spec, rec = SYSTEMS[system], load_hamiltonians(system)[0]
    feat = fit_featurizer(system)
    return {"formula": hill_formula(s for s, _ in rec.coordinates), "basis": spec.basis.lower(), "charge": 0,
            "multiplicity": 1, "active": [spec.active_electrons, spec.num_active_orbitals], "mapping": spec.mapping,
            "num_qubits": rec.num_qubits, "pool_size": len(get_pool(system)),
            "pauli_vocab_sha256": vocab_sha256(feat.keys), "pauli_vocab_size": len(feat.keys),
            "gauge_convention": f"{GAUGE_CONVENTION}:"
                                + sha256_text(repr((feat.gauge_paulis, [_r(x) for x in feat.ref])))[:16]}


def system_family(system: str) -> dict:
    spec, train = SYSTEMS[system], train_set(system).records
    P = np.array([[r.geometry_params[k] for k in spec.geometry_params] for r in train])
    loo = loo_nn_distances(system)
    return {"kind": "linear_h4" if spec.molecule == "H4" else "diatomic", "params": list(spec.geometry_params),
            "elements": sorted(s for s, _ in train[0].coordinates),
            "train_range": {k: [_r(lo), _r(hi)] for k, lo, hi in zip(spec.geometry_params, P.min(0), P.max(0))},
            "feature_distance": {
                "space": "standardised gauge-fixed Pauli coefficients (genqai.features.fit_featurizer, TRAIN mean/std)",
                "train_loo_nn_max": _r(loo.max()), "train_loo_nn_median": _r(np.median(loo)),
                "threshold": _r(DISTANCE_FACTOR * loo.max()),
                "rule": f"nearest-TRAIN distance <= {DISTANCE_FACTOR:g} x max leave-one-out NN distance of TRAIN"}}


def _stats(evals: list[EvalRecord], split: str) -> dict:
    s = summarize(evals)[evals[0].method][split]
    return {"n": s["n"], "median_abs_err_mha": _r(s["median_abs_err_mha"]), "max_abs_err_mha": _r(s["max_abs_err_mha"]),
            "frac_under_1.6mha": _r(s["frac_under_1.6mha"]), "pct_under_1.6mha": _r(100 * s["frac_under_1.6mha"]),
            "valid_frac": _r(s["valid_frac"]), "mean_num_operators": _r(s["mean_num_operators"]),
            "mean_cnot_count": _r(s["mean_cnot_count"])}


def val_metrics(system: str, kind: str, config: dict) -> dict:
    """Generate (one-shot, greedy, torch backend) and score every VAL geometry."""
    evals = []
    for r in load_hamiltonians(system, "val"):
        c, cost = propose(r, kind, system, cnot_budget=config.get("default_cnot_budget"), backend="torch")
        evals.append(score_circuit(r, c, kind, model_forward_passes=cost["forward_passes"],
                                   energy_evals=cost["ranking_energy_evals"]))
    return {**_stats(evals, "val"), "quantum_evals_to_propose": 0,
            "source": "computed by scripts/build_registry.py: one-shot generation on the VAL split, exact statevector "
                      "energy (genqai.evaluate.score_circuit)"}


def fresh_metrics(system: str, kind: str) -> dict | None:
    p = results_dir(fresh=True) / system / f"{FRESH_FILES[kind]}.jsonl"
    if not p.exists():
        return None
    rows = load_jsonl(p, EvalRecord)
    extrap = {h.geometry_id for h in load_hamiltonians(system, FRESH_SPLIT) if fresh_regime(system, h.geometry_params) == "extrap"}
    out = {**_stats(rows, FRESH_SPLIT), "source": _rel(p), "display_only": True}
    for name, sel in (("interp", [e for e in rows if e.geometry_id not in extrap]),
                      ("extrap", [e for e in rows if e.geometry_id in extrap])):
        if sel:
            st = _stats(sel, FRESH_SPLIT)
            out[name] = {k: st[k] for k in ("n", "median_abs_err_mha", "max_abs_err_mha", "pct_under_1.6mha")}
    return out


def limitations(system: str, kind: str, family: dict, sig: dict, fresh: dict | None, config: dict) -> list[str]:
    rng = ", ".join(f"{k} in [{lo:g}, {hi:g}] Å" for k, (lo, hi) in family["train_range"].items())
    out = [f"fixed problem: {sig['formula']} {sig['basis']} CAS({sig['active'][0]}e, {sig['active'][1]}o) "
           f"{sig['mapping']} mapping, {sig['num_qubits']} qubits; any other molecule, active space or mapping "
           "has no compatible generator (path C)",
           f"validated only inside the TRAIN range ({rng}) and within the feature-distance threshold"]
    if fresh and "extrap" in fresh:
        x = fresh["extrap"]
        edge = ", ".join(f"{k} > {hi:g} Å" for k, (_, hi) in family["train_range"].items())
        verdict = "fails" if x["pct_under_1.6mha"] < 100 else "held on the committed points but is unvalidated"
        out.append(f"one-shot {verdict} beyond the training edge ({edge}): on the {x['n']} test_fresh extrapolation "
                   f"geometries median |err| {x['median_abs_err_mha']:.2f} mHa, max {x['max_abs_err_mha']:.2f} mHa, "
                   f"{x['pct_under_1.6mha']:.0f}% within 1.6 mHa ({fresh['source']}); use +refine or adapt there")
    if kind in TRANSFORMER_STAGES:
        out.append("checkpoint selected on VAL, so VAL metrics are in-sample for model selection")
    if kind == "transformer_sft":
        out.append("torch-free inference via the NumPy export (greedy one-shot only; best-of-K needs torch)")
    if kind == "transformer_grpo":
        out.append("needs torch; energy-reward GRPO did not improve held-out error over SFT (README finding 3)")
    if kind == "transformer_budget":
        out.append(f"needs torch; default CX budget {config['default_cnot_budget']} is the VAL-chosen operating point "
                   "(results/<system>/budget_oneshot.jsonl); other budgets via cnot_budget")
    if kind == "krr":
        out.append("needs scikit-learn; alpha/gamma/sequence rule chosen on VAL energies, so VAL metrics are "
                   "in-sample for that choice")
    if kind == "nn_reuse_gauge":
        out.append("copies the nearest TRAIN geometry's ADAPT circuit with gauge-mapped angles; no interpolation")
    if kind == "angle_interp":
        out.append("1-D only; linear in r, linear extrapolation beyond the end segments")
    return out


def _train_dataset(system: str, kind: str) -> str:
    n = len(train_set(system).records)
    ham = f"{_rel(hamiltonians_path(system))} (split=train, n={n})"
    if kind == "nn_reuse_gauge":
        return f"{ham} + data/{system}/teacher.jsonl (raw ADAPT circuits)"
    if kind == "transformer_grpo":
        return f"{ham}: GRPO on exact energies, initialised from checkpoints/{system}/sft.pt"
    if kind == "transformer_budget":
        return f"{ham}: budget-conditioned GRPO, initialised from checkpoints/{system}/budget_sft.pt"
    return f"{ham} + data/{system}/teacher_canonical.jsonl (canonical ADAPT labels)"


def build_cards(systems=SYSTEM_ORDER) -> list[ModelCard]:
    cards = []
    for system in systems:
        sig, fam = system_signature(system), system_family(system)
        labels = ROOT / "data" / system / "teacher_canonical.jsonl"
        for kind in available_kinds(system):
            config, ckpt, export = {}, checkpoint_path(system, kind), numpy_export_path(system, kind)
            if kind == "krr":
                config = {**{k: (_r(v) if isinstance(v, float) else v) for k, v in krr_config(system).items()},
                          "train_labels_sha256": file_sha256(labels), "fit": "genqai.regression.AngleRegressor"}
            elif kind == "angle_interp":
                config = {"train_labels_sha256": file_sha256(labels), "fit": "genqai.regression.AngleInterpolator"}
            elif kind == "nn_reuse_gauge":
                config = {"train_labels_sha256": file_sha256(ROOT / "data" / system / "teacher.jsonl")}
            elif kind == "transformer_budget":
                rows = load_jsonl(results_dir() / system / "budget_oneshot.jsonl", EvalRecord)
                config = {"default_cnot_budget": int(rows[0].extra["cnot_budget"])}
            if export is not None and export.exists():
                parts = [export.with_name(p) for p in json.loads(export.read_text())["parts"]]
                exp_sha = file_sha256(export, *parts)
            else:
                export, exp_sha = None, None
            fresh = fresh_metrics(system, kind)
            validation = {"val": val_metrics(system, kind, config)}
            if fresh:
                validation["test_fresh"] = fresh
            cards.append(ModelCard(
                id=f"{system}/{kind}", kind=kind, system=system,
                checkpoint=_rel(ckpt) if ckpt else None, checkpoint_sha256=file_sha256(ckpt) if ckpt else None,
                signature=sig, family=fam, train_dataset=_train_dataset(system, kind), validation=validation,
                modes=["oneshot", "bestofk", "refine"] if kind in TRANSFORMER_STAGES else ["oneshot", "refine"],
                limitations=limitations(system, kind, fam, sig, fresh, config), config=config,
                numpy_export=_rel(export) if export else None, numpy_export_sha256=exp_sha))
            print(f"{system:5s} {kind:20s} VAL n={validation['val']['n']} median={validation['val']['median_abs_err_mha']:.4f} "
                  f"max={validation['val']['max_abs_err_mha']:.4f} <1.6={validation['val']['pct_under_1.6mha']:.0f}%",
                  file=sys.stderr)
    return cards


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=list(SYSTEM_ORDER))
    p.add_argument("--out", default=str(REGISTRY_PATH))
    p.add_argument("--check", action="store_true", help="exit 1 if the output would differ from --out")
    a = p.parse_args(argv)
    text = registry_json(build_cards(a.systems))
    out = Path(a.out)
    if a.check:
        same = out.exists() and out.read_text() == text
        print("registry up to date" if same else f"{out} is stale: re-run scripts/build_registry.py")
        raise SystemExit(0 if same else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
