"""Stage 2: conformer -> active-space qubit Hamiltonian (paper Sec. 3.1).

  geometry --PySCF RHF/6-31G--> orbitals --CASCI(ne, no)--> active-space integrals
           --Jordan-Wigner--> H = sum_i h_i P_i   (Pauli strings P_i, real coefficients h_i)
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ase.io import read
from openfermion import InteractionOperator, jordan_wigner
from openfermion.chem.molecular_data import spinorb_from_spatial
from pyscf import ao2mo, gto, mcscf, scf

DATA = Path(__file__).resolve().parents[2] / "data"

# (electrons, spatial orbitals) -> qubits = 2 * orbitals
ACTIVE_SPACES = {12: (6, 6), 14: (6, 7), 16: (8, 8)}


def build_hamiltonian(atoms, n_qubits=12, basis="6-31g", ref_mo=None):
    """Return dict with qubit Hamiltonian terms and reference energies for one geometry."""
    ne, no = ACTIVE_SPACES[n_qubits]
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                basis=basis, unit="Angstrom", spin=0, charge=0, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf)
        mf.kernel()

    mc = mcscf.CASCI(mf, no, ne)
    mo = mc.mo_coeff
    if ref_mo is not None:
        mo = _align_phases(mc, mo, ref_mo, mol)
    e_casci = mc.kernel(mo)[0]
    h1, e_core = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), no)  # chemist notation (pq|rs)

    one, two = spinorb_from_spatial(h1, np.asarray(eri.transpose(0, 2, 3, 1), order="C"))
    fermionic = InteractionOperator(e_core, one, 0.5 * two)
    qubit_op = jordan_wigner(fermionic)

    lo = mol.nelectron // 2 - ne // 2
    return {
        "terms": {k: float(np.real(v)) for k, v in qubit_op.terms.items()},
        "e_hf": float(mf.e_tot),
        "e_casci": float(e_casci),
        "e_core": float(e_core),
        "active_mo": mc.mo_coeff[:, lo:lo + no].copy(),
        "n_basis": mol.nao,
        "n_electrons_total": mol.nelectron,
    }


def _align_phases(mc, mo, ref_active_mo, mol):
    """Orbital signs are arbitrary; flip each active MO to agree with the reference conformer's.

    Same basis size/ordering across conformers (same atom order), so the AO-basis overlap with
    this geometry's S is a good proxy for orbital correspondence at small displacements."""
    no = mc.ncas
    n_act_e = sum(mc.nelecas)
    lo = mol.nelectron // 2 - n_act_e // 2
    mo = mo.copy()
    S = mol.intor("int1e_ovlp")
    for k in range(no):
        ov = ref_active_mo[:, k] @ S @ mo[:, lo + k]
        if ov < 0:
            mo[:, lo + k] *= -1
    return mo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conformers", default=str(DATA / "conformers.extxyz"))
    p.add_argument("--qubits", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    atoms_list = read(args.conformers, ":")[: args.limit]
    out = args.out or str(DATA / f"hamiltonians_{args.qubits}q.npz")
    results, ref_mo = [], None
    for i, a in enumerate(atoms_list):
        t = time.time()
        r = build_hamiltonian(a, args.qubits, ref_mo=ref_mo)
        ref_mo = ref_mo if ref_mo is not None else r["active_mo"]
        results.append(r)
        print(f"[{i + 1}/{len(atoms_list)}] {a.info['source']:>3}  E_HF={r['e_hf']:.6f}  "
              f"E_CASCI={r['e_casci']:.6f}  terms={len(r['terms'])}  {time.time() - t:.0f}s",
              flush=True)

    # Fixed canonical Pauli-string order shared by all conformers (union of all terms).
    def key(term):  # sort by weight then lexicographically
        return (len(term), term)

    all_terms = sorted({t for r in results for t in r["terms"]}, key=key)
    coeffs = np.array([[r["terms"].get(t, 0.0) for t in all_terms] for r in results])
    pauli = np.array(["".join("I" if q not in dict(t) else dict(t)[q]
                              for q in range(args.qubits)) for t in all_terms])
    np.savez(out, coeffs=coeffs, pauli=pauli,
             e_hf=[r["e_hf"] for r in results], e_casci=[r["e_casci"] for r in results],
             source=[a.info["source"] for a in atoms_list],
             conf_id=[a.info["id"] for a in atoms_list], n_qubits=args.qubits)
    print(f"saved {coeffs.shape} -> {out}")


if __name__ == "__main__":
    main()
