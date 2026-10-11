"""Workstream A (phase 2): any molecule -> validated active-space problem -> HamiltonianRecord.

Inputs: a preset (exact preset geometry), SMILES (RDKit ETKDGv3 conformers, explicit H), XYZ text
(strict parser, coordinates kept verbatim) or an atom list. validate_molecule rejects impossible or
unsupported molecules with a MoleculeError (a user-facing message, never a traceback).

ProblemConfig fixes everything that defines the qubit Hamiltonian; its key (like MoleculeSpec.key)
is a hash of physical content only, so caches can use it. assess() is cheap (no SCF) and reports
qubits, limits and feasibility; inputs beyond the limits get errors with suggestions, the problem is
never shrunk silently. build_problem() registers a SystemSpec in chemistry.SYSTEMS and builds the
record with chemistry.build_record, so pools, the simulator and circuits work unchanged.

Preset routing: a molecule whose geometry is a preset family (H2, LiH, linear H4 with (a, b, a)
spacing; any orientation, sto-3g, neutral singlet) with the preset's active-space settings is built
as that preset at the canonical preset geometry (parameters rounded to 1e-6 Angstrom). Its record is
then identical to chemistry.build_record(SYSTEMS[preset], params), so trained models and their gauge
fixing apply. Gauge (orbital sign) conventions are only defined within one system's family: across
different molecules or active spaces there is no shared gauge, and operator IDs of different
systems are unrelated.

RDKit is imported lazily inside from_smiles (it is not available on Qollab); PySCF only for basis
lookups and orbital counts (gto), plus SCF inside build_record.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

from genqai import chemistry
from genqai.chemistry import PRESETS, SYSTEMS, SystemSpec, build_record, register_system, sector_dim
from genqai.schema import HamiltonianRecord

LIMITS = {
    "max_atoms": 30,
    "max_xyz_bytes": 64 * 1024,
    "max_smiles_chars": 500,
    "max_conformers": 20,
    "max_basis_functions": 150,  # one SCF in the browser stays seconds, not minutes
    "max_active_orbitals": 8,
    "max_qubits": 16,  # exact statevector simulation
    "auto_max_active_orbitals": 8,  # default (auto) active space ...
    "auto_max_qubits": 12,  # ... stays within this many qubits
    "min_atom_distance_angstrom": 0.3,
    "ionq_max_packed_qubits": 29,
}
MAPPINGS = ("parity", "jordan_wigner")
POOLS = ("uccsd",)
_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


class MoleculeError(ValueError):
    """Invalid or unsupported molecule / problem; the message is meant for the user."""


def _elements() -> list[str]:
    from pyscf.data.elements import ELEMENTS

    return ELEMENTS[1:]  # index 0 is the ghost atom "X"


def _Z(symbol: str) -> int:
    return _elements().index(symbol) + 1


def _core_orbitals(z: int) -> int:
    """Chemical (noble-gas) core: doubly occupied orbitals frozen by default."""
    for z_max, n in ((2, 0), (10, 1), (18, 5), (36, 9), (54, 18), (86, 27)):
        if z <= z_max:
            return n
    return 43


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class MoleculeSpec:
    atoms: tuple  # ((symbol, (x, y, z)), ...) in Angstrom
    charge: int = 0
    multiplicity: int = 1  # 2S+1; PySCF spin = multiplicity - 1
    basis: str = "sto-3g"
    name: str = ""
    source: str = "atoms"  # "preset" | "smiles" | "xyz" | "atoms"
    smiles: str | None = None
    conformer: dict | None = None  # {"seed", "method": "ETKDGv3", "optimizer", "energy", ...}
    preset: tuple | None = None  # ("h2", (0.74,)) when it came from a preset

    @property
    def formula(self) -> str:
        """Hill formula (C, H, then alphabetical; alphabetical without carbon)."""
        counts: dict[str, int] = {}
        for sym, _ in self.atoms:
            counts[sym] = counts.get(sym, 0) + 1
        order = (["C", "H"] if "C" in counts else []) + sorted(s for s in counts if "C" not in counts or s not in ("C", "H"))
        return "".join(s + (str(counts[s]) if counts[s] > 1 else "") for s in order)

    @property
    def n_electrons(self) -> int:
        return sum(_Z(sym) for sym, _ in self.atoms) - self.charge

    @property
    def spin(self) -> int:
        return self.multiplicity - 1

    @property
    def key(self) -> str:
        """Stable hash of the physical content: elements, coordinates rounded to 1e-6 A, charge, mult, basis."""
        atoms = [[sym, [f"{round(float(x), 6) + 0.0:.6f}" for x in xyz]] for sym, xyz in self.atoms]
        return _hash([atoms, self.charge, self.multiplicity, self.basis.lower()])


def _symbol(text: str) -> str:
    sym = text.strip()
    sym = sym[:1].upper() + sym[1:].lower()
    if sym not in _elements():
        raise MoleculeError(f"unknown element symbol {text!r}")
    return sym


def _normalize_atoms(atoms) -> tuple:
    out = []
    try:
        for sym, xyz in atoms:
            xyz = tuple(float(x) for x in xyz)
            if len(xyz) != 3 or not all(np.isfinite(xyz)):
                raise MoleculeError(f"atom {sym!r} needs three finite coordinates, got {xyz}")
            out.append((_symbol(str(sym)), xyz))
    except (TypeError, ValueError) as e:
        if isinstance(e, MoleculeError):
            raise
        raise MoleculeError(f"atoms must be (symbol, (x, y, z)) pairs in Angstrom: {e}") from None
    if not out:
        raise MoleculeError("the molecule has no atoms")
    return tuple(out)


def from_atoms(atoms, *, charge: int = 0, multiplicity: int = 1, basis: str = "sto-3g", name: str = "",
               **kw) -> MoleculeSpec:
    """Molecule from ((symbol, (x, y, z)), ...) in Angstrom. Validated (raises MoleculeError)."""
    if len(atoms) > LIMITS["max_atoms"]:
        raise MoleculeError(f"{len(atoms)} atoms exceeds the limit of {LIMITS['max_atoms']}; use a smaller molecule")
    mol = MoleculeSpec(_normalize_atoms(atoms), charge=charge, multiplicity=multiplicity, basis=basis, name=name, **kw)
    validate_molecule(mol)
    return mol


def from_preset(system: str, params: tuple[float, ...]) -> MoleculeSpec:
    """The exact geometry chemistry.SYSTEMS[system].atoms(params) uses."""
    if system not in PRESETS:
        raise MoleculeError(f"unknown preset {system!r}; presets are {', '.join(PRESETS)}")
    spec = SYSTEMS[system]
    params = tuple(float(p) for p in params)
    if len(params) != len(spec.geometry_params) or not all(p > 0 for p in params):
        raise MoleculeError(f"preset {system} needs positive {', '.join(spec.geometry_params)} (Angstrom)")
    return from_atoms(spec.atoms(params), basis=spec.basis, name=spec.molecule, source="preset",
                      preset=(system, params))


def from_xyz(text: str, *, charge: int = 0, multiplicity: int = 1, basis: str = "sto-3g", name: str = "") -> MoleculeSpec:
    """Strict XYZ: either the standard format (count line, comment line, count atom lines) or bare
    'Symbol x y z' lines, Angstrom. Coordinates are kept verbatim. Nothing in the text is executed."""
    if not isinstance(text, str):
        raise MoleculeError("XYZ input must be text")
    if len(text.encode()) > LIMITS["max_xyz_bytes"]:
        raise MoleculeError(f"XYZ text is larger than {LIMITS['max_xyz_bytes'] // 1024} KB")
    lines = text.strip().splitlines()
    if not lines:
        raise MoleculeError("XYZ input is empty")
    comment = ""
    if re.fullmatch(r"\d+", lines[0].strip()):
        n = int(lines[0])
        if n > LIMITS["max_atoms"]:
            raise MoleculeError(f"{n} atoms exceeds the limit of {LIMITS['max_atoms']}; use a smaller molecule")
        if n < 1 or len(lines) < 2 + n or any(l.strip() for l in lines[2 + n:]):
            raise MoleculeError(f"malformed XYZ: the header declares {n} atoms but {max(len(lines) - 2, 0)} lines follow the comment line")
        comment, body, first = lines[1].strip(), lines[2:2 + n], 3
    else:
        body, first = lines, 1
    atoms = []
    for i, line in enumerate(body, start=first):
        tok = line.split()
        if not tok and body is lines:
            continue  # blank line between bare atom lines
        if len(tok) != 4 or not re.fullmatch(r"[A-Za-z]{1,3}", tok[0]) or not all(_FLOAT.fullmatch(t) for t in tok[1:]):
            raise MoleculeError(f"malformed XYZ line {i}: expected 'Symbol x y z', got {line.strip()[:60]!r}")
        atoms.append((tok[0], tuple(float(t) for t in tok[1:])))
    if len(atoms) > LIMITS["max_atoms"]:
        raise MoleculeError(f"{len(atoms)} atoms exceeds the limit of {LIMITS['max_atoms']}; use a smaller molecule")
    return from_atoms(atoms, charge=charge, multiplicity=multiplicity, basis=basis, name=name or comment[:80],
                      source="xyz")


def from_smiles(smiles: str, *, charge: int | None = None, multiplicity: int | None = None, basis: str = "sto-3g",
                seed: int = 20261010, optimizer: str | None = None, num_conformers: int = 1) -> list[MoleculeSpec]:
    """One MoleculeSpec per RDKit ETKDGv3 conformer (explicit H, recorded seed), optionally MMFF/UFF-optimised.

    Defaults: charge = formal charge of the SMILES, multiplicity = radical electrons + 1.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeError("SMILES input is empty")
    smiles = smiles.strip()
    if len(smiles) > LIMITS["max_smiles_chars"]:
        raise MoleculeError(f"SMILES longer than {LIMITS['max_smiles_chars']} characters")
    if not isinstance(num_conformers, int) or not 1 <= num_conformers <= LIMITS["max_conformers"]:
        raise MoleculeError(f"num_conformers must be 1..{LIMITS['max_conformers']}")
    opt = optimizer.upper() if optimizer else None
    if opt not in (None, "MMFF", "UFF"):
        raise MoleculeError(f"optimizer must be None, 'MMFF' or 'UFF', got {optimizer!r}")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.rdBase import BlockLogs
    except ImportError:
        raise MoleculeError("SMILES input needs RDKit, which this runtime does not have; enter XYZ coordinates instead") from None

    _quiet = BlockLogs()  # RDKit parse errors become our MoleculeError instead of stderr noise
    m = Chem.MolFromSmiles(smiles)  # sanitize=True
    if m is None:
        raise MoleculeError(f"invalid SMILES {smiles!r}: RDKit could not parse or sanitize it")
    charge = Chem.GetFormalCharge(m) if charge is None else charge
    if multiplicity is None:
        multiplicity = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()) + 1
    mh = Chem.AddHs(m)
    if mh.GetNumAtoms() > LIMITS["max_atoms"]:
        raise MoleculeError(f"{smiles} has {mh.GetNumAtoms()} atoms with hydrogens, over the limit of {LIMITS['max_atoms']}")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=num_conformers, params=params))
    if not cids:
        raise MoleculeError(f"RDKit ETKDGv3 could not embed 3D coordinates for {smiles!r} (seed {seed}); enter XYZ instead")
    results = [(None, None)] * len(cids)
    if opt == "MMFF":
        if not AllChem.MMFFHasAllMoleculeParams(mh):
            raise MoleculeError(f"MMFF94 has no parameters for {smiles!r}; use optimizer='UFF' or None")
        results = AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=2000)
    elif opt == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(mh):
            raise MoleculeError(f"UFF has no parameters for {smiles!r}; use optimizer=None")
        results = AllChem.UFFOptimizeMoleculeConfs(mh, maxIters=2000)
    symbols = [a.GetSymbol() for a in mh.GetAtoms()]
    out = []
    for cid, (not_converged, energy) in zip(cids, results):
        pos = mh.GetConformer(cid).GetPositions()
        conformer = {"seed": seed, "method": "ETKDGv3", "optimizer": opt, "conformer_id": int(cid),
                     "num_conformers": len(cids), "energy": None if energy is None else float(energy),
                     "energy_unit": None if energy is None else "kcal/mol",
                     "converged": None if not_converged is None else not_converged == 0}
        atoms = [(s, tuple(float(x) for x in p)) for s, p in zip(symbols, pos)]
        out.append(from_atoms(atoms, charge=charge, multiplicity=multiplicity, basis=basis, name=smiles,
                              source="smiles", smiles=smiles, conformer=conformer))
    return out


_NAO: dict[str, int] = {}


def _num_basis_functions(mol: MoleculeSpec) -> int:
    if mol.key not in _NAO:
        from pyscf import gto

        pm = gto.M(atom=[(s, xyz) for s, xyz in mol.atoms], basis=mol.basis, unit="Angstrom", verbose=0,
                   charge=mol.charge, spin=mol.spin)
        _NAO[mol.key] = int(pm.nao_nr())
    return _NAO[mol.key]


def _fragments(mol: MoleculeSpec) -> int:
    """Connected components with bonds where d < 1.6 (r_cov,i + r_cov,j) (generous: only flags clear gaps)."""
    from pyscf.data.radii import BOHR, COVALENT

    xyz = np.array([p for _, p in mol.atoms])
    r = np.array([COVALENT[_Z(s)] * BOHR for s, _ in mol.atoms])
    bonded = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) < 1.6 * (r[:, None] + r[None])
    seen, n = set(), 0
    for i in range(len(xyz)):
        if i in seen:
            continue
        n, stack = n + 1, [i]
        while stack:
            j = stack.pop()
            if j not in seen:
                seen.add(j)
                stack.extend(np.flatnonzero(bonded[j]).tolist())
    return n


def validate_molecule(mol: MoleculeSpec) -> list[str]:
    """Raises MoleculeError on hard errors (limits, spin parity, elements missing from the basis);
    returns warnings."""
    from pyscf.gto import basis as pyscf_basis

    if not 1 <= len(mol.atoms) <= LIMITS["max_atoms"]:
        raise MoleculeError(f"a molecule needs 1..{LIMITS['max_atoms']} atoms, got {len(mol.atoms)}")
    if not isinstance(mol.charge, (int, np.integer)) or isinstance(mol.charge, bool):
        raise MoleculeError(f"charge must be an integer, got {mol.charge!r}")
    if not isinstance(mol.multiplicity, (int, np.integer)) or isinstance(mol.multiplicity, bool) or mol.multiplicity < 1:
        raise MoleculeError(f"multiplicity (2S+1) must be a positive integer, got {mol.multiplicity!r}")
    ne = mol.n_electrons
    if ne < 1:
        raise MoleculeError(f"charge {mol.charge:+d} leaves {ne} electrons")
    if mol.spin > ne or (ne - mol.spin) % 2:
        parity = "odd" if ne % 2 else "even"
        allowed = "2, 4, ..." if ne % 2 else "1, 3, ..."
        raise MoleculeError(f"{mol.formula} with charge {mol.charge:+d} has {ne} electrons ({parity}); multiplicity "
                            f"{mol.multiplicity} is impossible (allowed: {allowed} up to {ne + 1})")
    for sym in sorted({s for s, _ in mol.atoms}):
        try:
            ok = bool(pyscf_basis.load(mol.basis, sym))
        except Exception:  # PySCF raises BasisNotFoundError / KeyError / RuntimeError depending on the basis
            ok = False
        if not ok:
            raise MoleculeError(f"basis {mol.basis!r} has no functions for element {sym}")
    xyz = np.array([p for _, p in mol.atoms])
    if len(xyz) > 1:
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(xyz)) * 1e9
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < LIMITS["min_atom_distance_angstrom"]:
            raise MoleculeError(f"atoms {i + 1} and {j + 1} are only {d[i, j]:.3f} A apart (overlapping)")
    nao = _num_basis_functions(mol)
    if nao > LIMITS["max_basis_functions"]:
        raise MoleculeError(f"{nao} basis functions in {mol.basis} exceeds the limit of {LIMITS['max_basis_functions']}; "
                            "use a smaller molecule or basis")
    warnings = []
    if mol.smiles and "." in mol.smiles:
        warnings.append("the SMILES has several disconnected fragments ('.'); they are treated as one system")
    elif len(mol.atoms) > 1 and _fragments(mol) > 1:
        warnings.append("the geometry has non-bonded fragments; they are treated as one system")
    if mol.multiplicity > 1:
        warnings.append("open shell: ROHF reference; the UCCSD pool conserves S_z but not S^2")
    return warnings


@dataclass(frozen=True)
class ProblemConfig:
    molecule: MoleculeSpec
    active_electrons: int | tuple[int, int] | None = None  # None = auto
    active_orbitals: int | None = None
    active_orbital_indices: tuple[int, ...] | None = None  # 0-based MO indices, overrides auto
    mapping: str = "parity"  # "parity" (two-qubit reduced) | "jordan_wigner"
    frozen_core: bool = True
    pool: str = "uccsd"

    @property
    def key(self) -> str:
        ae = self.active_electrons
        return _hash([self.molecule.key, list(ae) if isinstance(ae, tuple) else ae, self.active_orbitals,
                      None if self.active_orbital_indices is None else sorted(self.active_orbital_indices),
                      self.mapping, self.frozen_core, self.pool])


def _preset_candidates(mol: MoleculeSpec) -> list[tuple[str, tuple[float, ...]]]:
    """Presets whose geometry family contains this molecule (lih4 before lih6 for LiH)."""
    if mol.charge != 0 or mol.multiplicity != 1 or mol.basis.lower() != "sto-3g":
        return []
    if mol.preset is not None:
        family = SYSTEMS[mol.preset[0]].molecule
        return [mol.preset] + [(s, mol.preset[1]) for s in PRESETS if SYSTEMS[s].molecule == family and s != mol.preset[0]]
    syms = sorted(s for s, _ in mol.atoms)
    xyz = np.array([p for _, p in mol.atoms])
    if syms in (["H", "H"], ["H", "Li"]):
        r = (round(float(np.linalg.norm(xyz[1] - xyz[0])), 6),)
        return [("h2", r)] if syms == ["H", "H"] else [("lih4", r), ("lih6", r)]
    if syms == ["H"] * 4:
        axis = xyz[np.argmax(np.linalg.norm(xyz - xyz[0], axis=1))] - xyz[0]
        axis = axis / np.linalg.norm(axis)
        t = (xyz - xyz[0]) @ axis
        if np.abs(xyz - xyz[0] - np.outer(t, axis)).max() > 1e-6:
            return []
        a, b, c = np.diff(np.sort(t))
        if abs(a - c) <= 1e-6:
            return [("h4", (round(float(a), 6), round(float(b), 6)))]
    return []


def preset_family(mol: MoleculeSpec) -> tuple[str, tuple[float, ...]] | None:
    """(preset, params) when the molecule lies in a preset's geometry family, else None."""
    found = _preset_candidates(mol)
    return found[0] if found else None


def _preset_route(config: ProblemConfig) -> tuple[str, tuple[float, ...]] | None:
    """The preset whose SystemSpec settings this config reproduces exactly, if any."""
    ae = config.active_electrons
    ne = sum(ae) if isinstance(ae, tuple) else ae
    for system, params in _preset_candidates(config.molecule):
        spec = SYSTEMS[system]
        if (ne, config.active_orbitals, config.active_orbital_indices, config.mapping, config.frozen_core, config.pool) == \
                (spec.active_electrons, spec.num_active_orbitals, None, spec.mapping, True, "uccsd"):
            return system, params
    return None


def default_config(mol: MoleculeSpec) -> ProblemConfig:
    """Preset families -> exactly the preset's SystemSpec settings; anything else -> auto active space."""
    fam = preset_family(mol)
    if fam is None:
        return ProblemConfig(mol)
    spec = SYSTEMS[fam[0]]
    return ProblemConfig(mol, spec.active_electrons, spec.num_active_orbitals, mapping=spec.mapping)


@dataclass
class CapabilityReport:
    formula: str
    n_electrons: int
    charge: int
    multiplicity: int
    num_qubits: int
    active_electrons: tuple[int, int]
    active_orbitals: int
    exact_feasible: bool  # FCI sector diagonalisation + statevector within limits
    est_pauli_terms: int | None
    ionq_packed_qubits: int | None  # groups x qubits of the packed single-circuit scheme (needs the record)
    fits_ionq: bool | None  # packed width <= 29
    limits: dict
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    preset: tuple | None = None  # (system, params) when build_problem will build that preset


def num_qubits(num_orbitals: int, mapping: str) -> int:
    return 2 * num_orbitals - (2 if mapping == "parity" else 0)


def _auto_max_orbitals(mapping: str) -> int:
    n = LIMITS["auto_max_active_orbitals"]
    while n > 1 and num_qubits(n, mapping) > LIMITS["auto_max_qubits"]:
        n -= 1
    return n


def _active_space(config: ProblemConfig, nao: int, notes: list[str]) -> tuple[int, int]:
    """(active electrons, active orbitals). Raises MoleculeError with a suggestion when impossible."""
    mol = config.molecule
    ne, s = mol.n_electrons, mol.spin
    ncore_chem = sum(_core_orbitals(_Z(sym)) for sym, _ in mol.atoms) if config.frozen_core else 0
    ae = config.active_electrons
    if isinstance(ae, tuple):
        if len(ae) != 2 or ae[0] - ae[1] != s:
            raise MoleculeError(f"active (n_alpha, n_beta) = {ae} must differ by 2S = {s} (every unpaired electron is active)")
        ae = ae[0] + ae[1]
    if config.active_orbital_indices is not None:
        idx = sorted(config.active_orbital_indices)
        norb = len(idx)
        if config.active_orbitals is not None and config.active_orbitals != norb:
            raise MoleculeError(f"active_orbitals={config.active_orbitals} but {norb} orbital indices were given")
        if len(set(idx)) != norb or not norb or idx[0] < 0 or idx[-1] >= nao:
            raise MoleculeError(f"active orbital indices must be distinct MO indices in 0..{nao - 1}")
        if config.frozen_core and idx[0] < ncore_chem:
            raise MoleculeError(f"orbital {idx[0]} is a chemical core orbital (0..{ncore_chem - 1}); set frozen_core=False to include it")
        if ae is None:  # aufbau occupations, as CASCI.sort_mo assumes
            ndocc = (ne - s) // 2
            ae = sum(2 if i < ndocc else 1 if i < ndocc + s else 0 for i in idx)
    else:
        nval_e, nval_orb = ne - 2 * ncore_chem, nao - ncore_chem
        if nval_e < 1:
            raise MoleculeError("no electrons outside the frozen core; set frozen_core=False")
        norb = config.active_orbitals
        if norb is None:
            norb = min(nval_orb, _auto_max_orbitals(config.mapping))
        if ae is None:
            n_docc, n_virt = (nval_e - s) // 2, nval_orb - (nval_e - s) // 2 - s
            if norb < s:
                raise MoleculeError(f"{s} unpaired electrons need at least {s} active orbitals")
            d = min(n_docc, (norb - s + 1) // 2)
            d = min(n_docc, norb - s - min(n_virt, norb - s - d))
            ae = 2 * d + s
        if config.active_orbitals is None and (norb < nval_orb or ae < nval_e):
            notes.append(f"auto active space: {ae} electrons in {norb} orbitals around the Fermi level "
                         f"({nval_e} valence electrons in {nval_orb} orbitals outside the frozen core)")
    n_beta = (ae - s) // 2
    ncore = (ne - ae) // 2
    if ae < 1 or (ae - s) % 2 or n_beta < 0 or (ne - ae) % 2 or ncore < 0:
        raise MoleculeError(f"{ae} active electrons is impossible for {ne} electrons with multiplicity {mol.multiplicity}")
    if n_beta + s > norb:
        raise MoleculeError(f"{ae} electrons do not fit in {norb} active orbitals")
    if ncore + norb > nao:
        raise MoleculeError(f"{norb} active orbitals plus {ncore} core orbitals exceed the {nao} orbitals of {mol.basis}")
    if config.frozen_core and ncore < ncore_chem:
        raise MoleculeError(f"{ae} active electrons would put chemical core electrons in the active space; "
                            f"use at most {ne - 2 * ncore_chem} or set frozen_core=False")
    return ae, norb


def est_pauli_terms(num_orbitals: int, qubits: int) -> int:
    """Rough upper estimate (N^4 scaling, N spin orbitals), capped at 4^qubits."""
    n = 2 * num_orbitals
    return int(min(4**qubits, n**4 // 16 + n**2 + 1))


def packed_width(record: HamiltonianRecord) -> int:
    """Qubits of the packed single-circuit IonQ measurement: QWC groups x qubits (genqai.measure)."""
    from genqai.measure import MeasurementPlan

    return MeasurementPlan.from_record(record).num_groups * record.num_qubits


def assess(config: ProblemConfig, record: HamiltonianRecord | None = None) -> CapabilityReport:
    """Cheap capability check (no SCF). Pass the built record to get the IonQ packed width."""
    mol = config.molecule
    limits = {**LIMITS, "fci_max_sector_dim": chemistry.FCI_MAX_DIM}
    errors, warnings = [], []
    rep = CapabilityReport(mol.formula, 0, mol.charge, mol.multiplicity, 0, (0, 0), 0, False, None, None, None,
                           limits, warnings, errors)
    try:
        warnings += validate_molecule(mol)
        rep.n_electrons = mol.n_electrons
        if config.mapping not in MAPPINGS:
            raise MoleculeError(f"mapping must be one of {MAPPINGS}, got {config.mapping!r}")
        if config.pool not in POOLS:
            raise MoleculeError(f"pool must be one of {POOLS}, got {config.pool!r}")
        ae, norb = _active_space(config, _num_basis_functions(mol), warnings)
    except MoleculeError as e:
        errors.append(str(e))
        return rep
    n_alpha = (ae + mol.spin) // 2
    rep.active_electrons, rep.active_orbitals = (n_alpha, ae - n_alpha), norb
    rep.num_qubits = q = num_qubits(norb, config.mapping)
    rep.est_pauli_terms = est_pauli_terms(norb, q)
    rep.preset = _preset_route(config)
    fix = (f"choose a smaller active space, e.g. ProblemConfig(mol, active_orbitals={_auto_max_orbitals(config.mapping)}) "
           "(electrons are then placed around the Fermi level)")
    if q < 1:
        errors.append(f"{norb} active orbital(s) give no qubits with the {config.mapping} mapping; use more orbitals")
    if norb > LIMITS["max_active_orbitals"]:
        errors.append(f"{norb} active orbitals exceeds the limit of {LIMITS['max_active_orbitals']}; {fix}")
    if q > LIMITS["max_qubits"]:
        errors.append(f"{q} qubits exceeds the exact-simulation limit of {LIMITS['max_qubits']}; {fix}")
    rep.exact_feasible = not errors and sector_dim(norb, rep.active_electrons) <= chemistry.FCI_MAX_DIM
    if not errors and not rep.exact_feasible:
        warnings.append(f"{sector_dim(norb, rep.active_electrons)} determinants is too many for exact FCI here; the "
                        "reference will be CCSD (HF where PySCF's CC module is missing), not exact")
    if record is not None:
        rep.ionq_packed_qubits = packed_width(record)
        rep.fits_ionq = rep.ionq_packed_qubits <= LIMITS["ionq_max_packed_qubits"]
        if not rep.fits_ionq:
            warnings.append(f"packed IonQ measurement needs {rep.ionq_packed_qubits} qubits "
                            f"(> {LIMITS['ionq_max_packed_qubits']}): no single-job IonQ run; local backends still work")
    return rep


def build_problem(config: ProblemConfig) -> HamiltonianRecord:
    """Register the SystemSpec (preset name for preset routes, else "mol-<key[:10]>") and build its record."""
    rep = assess(config)
    if rep.errors:
        raise MoleculeError("; ".join(rep.errors))
    if rep.preset is not None:
        spec, params = SYSTEMS[rep.preset[0]], rep.preset[1]
    else:
        mol = config.molecule
        idx = config.active_orbital_indices
        spec = register_system(SystemSpec(
            name=f"mol-{config.key[:10]}", molecule=mol.formula, geometry_params=(),
            active_electrons=sum(rep.active_electrons), num_active_orbitals=rep.active_orbitals,
            mapping=config.mapping, basis=mol.basis, charge=mol.charge, multiplicity=mol.multiplicity,
            fixed_atoms=mol.atoms, active_orbital_indices=None if idx is None else tuple(sorted(idx)),
            frozen_core=config.frozen_core))
        params = ()
    try:
        record = build_record(spec, params)
    except (RuntimeError, ValueError) as e:
        raise MoleculeError(f"could not build the Hamiltonian for {config.molecule.formula}: {e}") from None
    record.problem_key = config.key
    return record
