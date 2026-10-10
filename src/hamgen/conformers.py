"""Stage 1: generate imipramine conformers (paper Sec. 3.1, scaled down).

Three sources, as in the paper:
  * MD   - Langevin dynamics at 360 K, friction 1/ps, MACE-OFF forces, several seeds.
  * NEB  - images along nudged-elastic-band paths between RDKit (ETKDG) reference conformers.
  * OoD  - the highest-energy reference conformer, rattled with sigma = 0.05 A.
"""
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ase import Atoms, units
from ase.build import minimize_rotation_and_translation
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.mep import NEB
from ase.optimize import BFGS
from mace.calculators import mace_off
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

IMIPRAMINE_SMILES = "CN(C)CCCN1c2ccccc2CCc2ccccc21"
DATA = Path(__file__).resolve().parents[2] / "data"


def calc():
    return mace_off(model="small", device="cpu", default_dtype="float32")


def relax(atoms, fmax=0.05, steps=200):
    atoms = atoms.copy()
    atoms.calc = calc()
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def run_md(start, seed, n_steps, sample_every, dt_fs, temp_k=360.0, friction_per_ps=1.0):
    """One Langevin trajectory; returns snapshots after a short equilibration."""
    rng = np.random.default_rng(seed)
    atoms = start.copy()
    atoms.calc = calc()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp_k, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=temp_k,
                   friction=friction_per_ps / (1000 * units.fs), rng=rng, logfile=None)
    frames = []
    equil = n_steps // 5
    dyn.run(equil)

    def grab():
        a = atoms.copy()
        a.info["energy_eV"] = float(atoms.get_potential_energy())
        frames.append(a)

    dyn.attach(grab, interval=sample_every)
    dyn.run(n_steps - equil)
    return frames


def _atom_permutation(start, template):
    """perm[i] = index in `start` (paper XYZ order) of template atom i (graph isomorphism)."""
    import networkx as nx
    xyz = Chem.MolFromXYZBlock(
        f"{len(start)}\n\n" + "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in
                                       zip(start.get_chemical_symbols(), start.positions)))
    rdDetermineBonds.DetermineConnectivity(xyz)

    def graph(m):
        g = nx.Graph()
        for at in m.GetAtoms():
            g.add_node(at.GetIdx(), z=at.GetAtomicNum())
        g.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds())
        return g

    gm = nx.isomorphism.GraphMatcher(graph(template), graph(xyz),
                                     node_match=lambda a, b: a["z"] == b["z"])
    mapping = next(gm.isomorphisms_iter())
    return [mapping[i] for i in range(template.GetNumAtoms())]


def reference_conformers(start, n_refs, seed=42):
    """ETKDG conformers (RDKit), relaxed with MACE-OFF, sorted by energy."""
    molH = Chem.AddHs(Chem.MolFromSmiles(IMIPRAMINE_SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1
    cids = AllChem.EmbedMultipleConfs(molH, 4 * n_refs, params)
    perm = _atom_permutation(start, molH)  # keep the paper's atom order everywhere
    symbols = start.get_chemical_symbols()
    refs = []
    for cid in list(cids):
        raw = molH.GetConformer(cid).GetPositions()
        pos = np.zeros_like(raw)
        pos[perm] = raw
        a = Atoms(symbols, positions=pos)
        a = relax(a, fmax=0.1, steps=150)
        a.info["energy_eV"] = float(a.get_potential_energy())
        refs.append(a)
    refs.sort(key=lambda a: a.info["energy_eV"])
    # keep a structurally spread-out subset of n_refs
    keep = [refs[0]]
    for a in refs[1:]:
        if all(_rmsd(a, b) > 0.3 for b in keep):
            keep.append(a)
        if len(keep) == n_refs:
            break
    return keep


def _rmsd(a, b):
    a = a.copy()
    b = b.copy()
    minimize_rotation_and_translation(b, a)
    return float(np.sqrt(((a.positions - b.positions) ** 2).sum(1).mean()))


def neb_images(a, b, n_images, steps=60):
    b = b.copy()
    minimize_rotation_and_translation(a, b)
    images = [a.copy()] + [a.copy() for _ in range(n_images)] + [b.copy()]
    for im in images:
        im.calc = calc()
    neb = NEB(images, climb=False)
    neb.interpolate("idpp")
    BFGS(neb, logfile=None).run(fmax=0.1, steps=steps)
    out = []
    e_end = max(images[0].get_potential_energy(), images[-1].get_potential_energy())
    for im in images[1:-1]:
        e = float(im.get_potential_energy())
        if e - e_end > 5.0:  # unphysical path, discard (paper discards failed NEBs)
            return []
        c = Atoms(im.get_chemical_symbols(), positions=im.positions)
        c.info["energy_eV"] = e
        out.append(c)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-md", type=int, default=78)
    p.add_argument("--n-neb", type=int, default=16)
    p.add_argument("--n-ood", type=int, default=6)
    p.add_argument("--md-trajectories", type=int, default=5)
    p.add_argument("--md-steps", type=int, default=3750)
    p.add_argument("--md-dt-fs", type=float, default=0.5)
    p.add_argument("--n-refs", type=int, default=15)
    p.add_argument("--neb-images", type=int, default=6)
    p.add_argument("--out", default=str(DATA / "conformers.extxyz"))
    args = p.parse_args()

    start = relax(read(DATA / "imipramine.xyz"), fmax=0.02, steps=500)
    print("relaxed start geometry, E =", start.get_potential_energy(), "eV", flush=True)

    # --- MD
    per_traj = -(-args.n_md // args.md_trajectories)
    sample_every = (args.md_steps - args.md_steps // 5) // per_traj
    md = []
    for seed in range(args.md_trajectories):
        frames = run_md(start, seed, args.md_steps, sample_every, args.md_dt_fs)[:per_traj]
        for i, f in enumerate(frames):
            f.info.update(source="MD", traj=seed, frame=i)
        md += frames
        print(f"MD trajectory {seed}: {len(frames)} frames", flush=True)
    md = md[: args.n_md]

    # --- reference conformers (ETKDG) -> NEB + OoD
    refs = reference_conformers(start, args.n_refs)
    print(f"{len(refs)} reference conformers", flush=True)
    neb = []
    pairs = list(itertools.combinations(range(len(refs)), 2))
    np.random.default_rng(0).shuffle(pairs)
    for i, j in pairs:
        if len(neb) >= args.n_neb:
            break
        imgs = neb_images(refs[i], refs[j], args.neb_images)
        for k, im in enumerate(imgs):
            im.info.update(source="NEB", pair=f"{i}-{j}", frame=k)
        neb += imgs
        print(f"NEB {i}-{j}: {len(imgs)} images", flush=True)
    neb = neb[: args.n_neb]

    top = max(refs, key=lambda a: a.info["energy_eV"])
    rng = np.random.default_rng(1)
    ood = []
    for k in range(args.n_ood):
        a = Atoms(top.get_chemical_symbols(), positions=top.positions)
        a.rattle(stdev=0.05, rng=rng)
        a.calc = calc()
        a.info.update(source="OoD", frame=k, energy_eV=float(a.get_potential_energy()))
        ood.append(a)

    allc = md + neb + ood
    for idx, a in enumerate(allc):
        a.info["id"] = idx
        a.calc = None
    write(args.out, allc)
    print(f"wrote {len(allc)} conformers to {args.out} "
          f"(MD {len(md)}, NEB {len(neb)}, OoD {len(ood)})")


if __name__ == "__main__":
    main()
