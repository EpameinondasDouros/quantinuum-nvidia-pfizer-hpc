"""Model registry and routing (workbench lane B).

models/registry.json (written by scripts/build_registry.py) holds one ModelCard per trained generator
per preset system. route() decides, for a problem the user built, which of them may run:

  A  validated     signature + Pauli vocabulary + geometry family match a trained system, every family
                   parameter lies inside the TRAIN range, and the Hamiltonian's distance to the nearest
                   TRAIN Hamiltonian in feature space is <= the system's threshold.
  B  experimental  compatible, but outside the TRAIN range or too far in feature space: one-shot
                   generators may run, flagged experimental; "<best>+refine" is recommended.
  C  conventional  no compatible trained generator, problem feasible: ADAPT-VQE / UCCSD-VQE only, and
                   their circuits are logged as potential teacher data (genqai.generate).
  D  no execution  the capability report has errors or exact simulation is infeasible.

Matching (match_family): a record belongs to trained system S only if (1) its full signature equals
S's (formula from the atoms, basis, charge, multiplicity, active electrons/orbitals, mapping, qubits,
pool size), (2) every Pauli label of its Hamiltonian is in S's vocabulary (genqai.features: union over
S's grid geometries; checked against the sha256 stored in the card), and (3) its atoms belong to S's
geometry family, in any orientation/translation: "diatomic" -> (r,) = interatomic distance;
"linear_h4" -> 4 H atoms collinear within GEOMETRY_TOL_A, with end spacings equal within
GEOMETRY_TOL_A -> (a, b) = (mean end spacing, middle spacing).

Feature-space distance: Euclidean distance, in the featurizer's standardised gauge-fixed coefficient
space (genqai.features.fit_featurizer: TRAIN mean/std), to the nearest TRAIN Hamiltonian. Threshold =
DISTANCE_FACTOR x the largest leave-one-out nearest-neighbour distance among the TRAIN Hamiltonians
(stored per system in the card): a new Hamiltonian may be no farther from the training set than the
most isolated training point is from the rest. On the committed data every VAL/test_interp/
test_interval/in-range test_fresh geometry passes (ratio <= 0.81) and the extrapolation geometries
beyond the LiH edge fail (ratio >= 1.03).

The recommended method is chosen from the cards' VAL metrics only (never test_fresh): highest
fraction of VAL geometries within 1.6 mHa, then lowest median |error| (rounded to 1e-3 mHa), then
lowest max |error|, then fewer quantum evaluations to propose, then RECOMMEND_TIEBREAK order.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from genqai.generate import LEARNED_KINDS, SOLVERS, TRANSFORMER_STAGES, torch_available, train_set
from genqai.schema import HamiltonianRecord

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "models" / "registry.json"
SCHEMA = "genqai.registry/v1"
GEOMETRY_TOL_A = 1e-3
DISTANCE_FACTOR = 1.0
PARAM_DECIMALS = 6
RECOMMEND_TIEBREAK = ("krr", "transformer_sft", "transformer_grpo", "transformer_budget", "angle_interp",
                      "nn_reuse_gauge")  # cheapest/simplest first among exact ties
GAUGE_CONVENTION = "orbital-parity-argmax-ref"  # genqai.features.Featurizer gauge fixing


@dataclass
class ModelCard:
    id: str
    kind: str  # "transformer_sft" | "transformer_grpo" | "transformer_budget" | "krr" | "nn_reuse_gauge" | "angle_interp"
    system: str
    checkpoint: str | None
    checkpoint_sha256: str | None
    signature: dict  # formula, basis, charge, multiplicity, active [ne, norb], mapping, num_qubits, pool_size,
    #                  pauli_vocab_sha256, pauli_vocab_size, gauge_convention
    family: dict  # kind, params, elements, train_range, feature_distance
    train_dataset: str
    validation: dict  # "val" (computed; used for selection) + "test_fresh" (committed results; display only)
    modes: list[str]
    limitations: list[str]
    config: dict = field(default_factory=dict)  # krr hyperparameters, budget default CX, ...
    numpy_export: str | None = None  # torch-free export (genqai.infer_numpy)
    numpy_export_sha256: str | None = None


@dataclass
class RoutingDecision:
    path: str  # "A" | "B" | "C" | "D"
    system: str | None
    params: tuple | None
    compatible: list[str]  # ModelCard ids
    recommended: str  # method id, chosen by VAL metrics (never by test)
    available_methods: list[str]
    reasons: list[str]
    ood: dict  # in_range per param, train_range, feature distance vs threshold
    suggestions: list[str] = field(default_factory=list)  # path D: what to change


# --------------------------------------------------------------------------- registry file

@lru_cache(maxsize=None)
def _load(path: str) -> tuple[ModelCard, ...]:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, got {data.get('schema')}")
    return tuple(ModelCard(**c) for c in data["cards"])


def load_registry(path: str | Path | None = None) -> list[ModelCard]:
    return list(_load(str(path or REGISTRY_PATH)))


def card_for(model_id: str) -> ModelCard:
    for c in load_registry():
        if c.id == model_id:
            return c
    raise KeyError(f"no model card {model_id!r} in {REGISTRY_PATH.relative_to(ROOT)}")


def registry_json(cards: list[ModelCard]) -> str:
    """Canonical serialisation (deterministic; no timestamps)."""
    data = {"schema": SCHEMA, "geometry_tol_angstrom": GEOMETRY_TOL_A, "distance_factor": DISTANCE_FACTOR,
            "cards": [asdict(c) for c in cards]}
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def _systems(cards: list[ModelCard]) -> dict[str, list[ModelCard]]:
    out: dict[str, list[ModelCard]] = {}
    for c in cards:
        out.setdefault(c.system, []).append(c)
    return out


# --------------------------------------------------------------------------- signatures

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def vocab_sha256(labels) -> str:
    return sha256_text(json.dumps(sorted(labels)))


def hill_formula(elements) -> str:
    n = Counter(elements)
    order = (["C", "H"] if "C" in n else []) + sorted(e for e in n if e not in (("C", "H") if "C" in n else ()))
    return "".join(f"{e}{n[e] if n[e] > 1 else ''}" for e in order)


def _atoms(record: HamiltonianRecord | None, mol) -> list[tuple[str, np.ndarray]]:
    src = mol.atoms if mol is not None else (record.coordinates if record is not None else [])
    return [(str(s), np.asarray(xyz, float)) for s, xyz in src]


def problem_signature(record: HamiltonianRecord | None = None, mol=None, config=None, report=None) -> dict:
    """Signature fields knowable for this problem (record > report/config > mol > defaults)."""
    atoms = _atoms(record, mol)
    sig: dict = {"formula": hill_formula(s for s, _ in atoms) if atoms else None}
    sig["basis"] = (record.basis if record is not None else getattr(mol, "basis", None) or "").lower() or None
    sig["charge"] = getattr(record, "charge", None)
    sig["multiplicity"] = getattr(record, "multiplicity", None)
    if sig["charge"] is None:
        sig["charge"] = int(getattr(mol, "charge", 0))
    if sig["multiplicity"] is None:
        sig["multiplicity"] = int(getattr(mol, "multiplicity", 1))
    if record is not None:
        sig.update(active=[record.active_electrons, len(record.active_orbitals)], mapping=record.mapping,
                   num_qubits=record.num_qubits)
        try:
            from genqai.pool import get_pool

            sig["pool_size"] = len(get_pool(record.system))
        except KeyError:  # system not registered in chemistry.SYSTEMS: (ne, norb, mapping) fix the pool
            pass
    else:
        if report is not None:
            ae = report.active_electrons
            sig.update(active=[int(sum(ae)) if isinstance(ae, (tuple, list)) else int(ae), int(report.active_orbitals)],
                       num_qubits=int(report.num_qubits))
        if config is not None:
            sig["mapping"] = config.mapping
    return {k: v for k, v in sig.items() if v is not None}


def family_params(kind: str, atoms: list[tuple[str, np.ndarray]]) -> tuple[float, ...] | None:
    """Geometry-family parameters of `atoms` (any orientation/translation), or None if not in the family."""
    X = np.array([x for _, x in atoms]) if atoms else np.zeros((0, 3))
    if kind == "diatomic":
        return (round(float(np.linalg.norm(X[1] - X[0])), PARAM_DECIMALS),) if len(X) == 2 else None
    if kind == "linear_h4":
        if len(X) != 4 or any(s != "H" for s, _ in atoms):
            return None
        c = X - X.mean(0)
        u = np.linalg.svd(c)[2][0]  # principal axis
        t = c @ u
        if np.abs(c - np.outer(t, u)).max() > GEOMETRY_TOL_A:  # not collinear
            return None
        d1, d2, d3 = np.diff(np.sort(t))
        if abs(d1 - d3) > GEOMETRY_TOL_A:
            return None
        return round(float((d1 + d3) / 2), PARAM_DECIMALS), round(float(d2), PARAM_DECIMALS)
    raise ValueError(kind)


@lru_cache(maxsize=None)
def system_vocab(system: str) -> frozenset[str]:
    return frozenset(train_set(system).featurizer.keys)


@lru_cache(maxsize=None)
def _train_features(system: str) -> np.ndarray:
    ts = train_set(system)
    return ts.featurizer(ts.records).astype(float)


def feature_distance(system: str, record: HamiltonianRecord) -> float:
    """Distance to the nearest TRAIN Hamiltonian in the standardised gauge-fixed feature space."""
    x = train_set(system).featurizer([record])[0].astype(float)
    return float(np.linalg.norm(_train_features(system) - x, axis=1).min())


def loo_nn_distances(system: str) -> np.ndarray:
    """Leave-one-out nearest-neighbour distance of every TRAIN Hamiltonian (threshold calibration)."""
    X = _train_features(system)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    return D.min(1)


def _match(record, mol, config=None, report=None, cards=None) -> tuple[str | None, tuple | None, list[str]]:
    """(system, params, reasons) of the first trained system this problem belongs to."""
    cards = load_registry() if cards is None else cards
    sig = problem_signature(record, mol, config, report)
    atoms = _atoms(record, mol)
    reasons = []
    for system, cs in _systems(cards).items():
        ref = cs[0].signature
        diff = [k for k in sig if k in ref and sig[k] != ref[k]]
        if diff:
            reasons.append(f"{system}: signature differs in {', '.join(f'{k} ({sig[k]} vs {ref[k]})' for k in diff)}")
            continue
        if record is not None:
            vocab = system_vocab(system)
            if vocab_sha256(vocab) != ref["pauli_vocab_sha256"]:
                raise RuntimeError(f"{system}: Pauli vocabulary changed since the registry was built; "
                                   "re-run scripts/build_registry.py")
            extra = {t["pauli"] for t in record.pauli_terms} - vocab
            if extra:
                reasons.append(f"{system}: {len(extra)} Pauli terms outside the trained vocabulary")
                continue
        params = family_params(cs[0].family["kind"], atoms)
        if params is None:
            reasons.append(f"{system}: geometry is not in the {cs[0].family['kind']} family")
            continue
        return system, params, [f"matches trained system {system} ({cs[0].family['kind']}, "
                                + ", ".join(f"{k} = {v:g} Å" for k, v in zip(cs[0].family["params"], params)) + ")"]
    return None, None, reasons


def match_family(record: HamiltonianRecord, mol=None) -> tuple[str, tuple[float, ...]] | None:
    """(trained system, family params) if signature, Pauli vocabulary and geometry family all match."""
    system, params, _ = _match(record, mol)
    return None if system is None else (system, params)


# --------------------------------------------------------------------------- routing

def runnable(card: ModelCard) -> bool:
    """Whether this runtime can execute the card (torch / NumPy export / scikit-learn)."""
    if card.kind in TRANSFORMER_STAGES:
        return torch_available() or (card.numpy_export is not None and (ROOT / card.numpy_export).exists())
    if card.kind == "krr":
        return importlib.util.find_spec("sklearn") is not None
    return True


def rank_key(card: ModelCard):
    v = card.validation["val"]
    return (-v["frac_under_1.6mha"], round(v["median_abs_err_mha"], 3), round(v["max_abs_err_mha"], 3),
            v.get("quantum_evals_to_propose", 0), RECOMMEND_TIEBREAK.index(card.kind))


def _methods(cards: list[ModelCard]) -> list[str]:
    out = []
    for c in cards:
        out += [c.kind, f"{c.kind}+refine"]
        if c.kind in TRANSFORMER_STAGES and torch_available():
            out.append(f"{c.kind}+bestofk")
    return out + list(SOLVERS)


def route(config=None, record: HamiltonianRecord | None = None, report=None) -> RoutingDecision:
    """Path A/B/C/D for a problem (config: genqai.molecule.ProblemConfig, report: CapabilityReport;
    both duck-typed and optional, record: the built Hamiltonian, optional before build_problem)."""
    if report is not None and (report.errors or not report.exact_feasible):
        why = list(report.errors) or ["exact simulation is not feasible within the workbench limits"]
        sugg = list(report.warnings) + ["reduce the active space (fewer active orbitals/electrons)",
                                        "use a smaller basis or a smaller molecule",
                                        f"limits applied: {report.limits}"]
        return RoutingDecision("D", None, None, [], "", [], why, {}, sugg)
    cards = load_registry()
    mol = getattr(config, "molecule", None)
    system, params, reasons = _match(record, mol, config, report, cards)
    if system is None:
        return RoutingDecision("C", None, None, [], "adapt", list(SOLVERS),
                               ["no compatible trained generator"] + reasons, {})
    cs = [c for c in cards if c.system == system]
    fam = cs[0].family
    in_range = {k: bool(lo <= v <= hi) for k, v, (lo, hi) in
                zip(fam["params"], params, (fam["train_range"][k] for k in fam["params"]))}
    ood: dict = {"in_range": in_range, "train_range": fam["train_range"]}
    ok = all(in_range.values())
    if not ok:
        reasons.append("outside the TRAIN range: " + ", ".join(
            f"{k} = {v:g} not in [{fam['train_range'][k][0]:g}, {fam['train_range'][k][1]:g}]"
            for k, v in zip(fam["params"], params) if not in_range[k]))
    if record is not None:
        d, thr = feature_distance(system, record), fam["feature_distance"]["threshold"]
        ood.update(feature_distance=d, threshold=thr, ratio=d / thr)
        if d > thr:
            ok = False
            reasons.append(f"Hamiltonian is {d / thr:.2f}x the TRAIN nearest-neighbour threshold away from "
                           "the training set in feature space")
    else:
        reasons.append("provisional: Hamiltonian not built yet (vocabulary and feature distance unchecked)")
    usable = [c for c in cs if runnable(c)]
    best = min(usable, key=rank_key).kind if usable else ""
    path = "A" if ok else "B"
    recommended = best if path == "A" else (f"{best}+refine" if best else "adapt")
    return RoutingDecision(path, system, params, [c.id for c in cs], recommended, _methods(usable), reasons, ood)
