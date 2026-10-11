# Custom molecules: walkthrough

Any small molecule can go through GenQAI. Only four problems have trained circuit generators (H₂,
LiH 4q/6q, the H₄ chain), so a new molecule takes the conventional-solver route (path C), and the
workbench says so. Every number below was produced by the code on Oct 11, 2026; re-run the snippet to
reproduce it.

## In the workbench

`streamlit run app/workbench.py`, then on the Molecule tab:
1. Choose **SMILES**, type `O`, press **Embed conformers**. RDKit adds hydrogens and embeds an ETKDGv3
   conformer with seed 20261010; optionally relax it with MMFF/UFF or embed several and pick one.
   **XYZ** keeps uploaded coordinates verbatim. **Atoms** is an editable table.
2. Set charge, multiplicity, basis, frozen core, active electrons/orbitals and mapping under "Active
   space and mapping". Defaults freeze the chemical core and pick orbitals around the Fermi level,
   up to 8 orbitals / 12 qubits.
3. **Build Hamiltonian**. The capability table answers five separate questions: is the input
   accepted, is the Hamiltonian built, does a trained model cover it (with the reason), is a fallback
   solver available, and which backends can run it (including whether the packed single-circuit IonQ
   measurement fits in 29 qubits).
4. Generate (auto picks ADAPT on path C) → Grade → optionally Optimise → Execute → History & export.

## In Python

```python
from genqai import molecule as M
from genqai.registry import route
from genqai.generate import generate
from genqai.grading import grade
from genqai.execution import execute
from genqai.export import qiskit_code

mol = M.from_xyz("2\nHeH+\nHe 0 0 0\nH 0 0 0.772\n", charge=1)   # or M.from_smiles("O")[0]
config = M.default_config(mol)                                     # or M.ProblemConfig(mol, active_electrons=4, active_orbitals=4)
record = M.build_problem(config)
decision = route(config, record, M.assess(config, record))         # path "C": no trained generator
circuit, log = generate(record, "adapt", decision)                 # generate(record, "krr", ...) raises GenerationError here
report = grade(record, circuit, gen_log=log)
run = execute(record, circuit, "local_shots", shots=2000)
print(qiskit_code(record, circuit))
```

## Results (ADAPT on path C, graded against FCI of the same active-space Hamiltonian)

| Molecule | Input | Charge / mult. | Active space | Qubits | Pauli terms | HF (Ha) | FCI (Ha) | ADAPT ops | Quantum evals | Error | CX | IonQ packed fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HeH⁺ | XYZ | +1 / 1 | 2e / 2o | 2 | 9 | −2.841382 | −2.851024 | 3 | 66 | 0.0000 mHa | 4 | yes (8 qubits) |
| H₂O | SMILES `O` | 0 / 1 | 4e / 4o | 6 | 175 | −74.963550 | −74.971616 | 6 | 243 | 0.0005 mHa | 262 | no (48 groups) |
| OH• | SMILES `[OH]` | 0 / 2 | 5e (3α, 2β) / 4o | 6 | 93 | −74.363047 | −74.380209 | 5 | 168 | 0.0000 mHa | 202 | no (24 groups) |

FCI for all three matches native PySCF CASCI to 1e-8 (`tests/test_molecule.py`). The open-shell case
uses ROHF with a doubly occupied core; its UCCSD pool conserves S_z, not S².

**HeH⁺ on IonQ's Forte 1 noise-model simulator** (free, 1,000 shots, 4 measurement groups packed into
one 8-qubit circuit): E = −2.814929 ± 0.011048 Ha, **+36.1 ± 11.0 mHa** from FCI. Job
`01a12858-0690-7546-94e7-76cddf9f0e3f`; the record is `results/custom_heh+_ionq_forte1.json`. The
exact energy of the same circuit is at FCI, so the gap is shot noise plus IonQ's noise model.

## What the workbench refuses, and why

- Invalid SMILES or XYZ, an electron count inconsistent with the multiplicity, elements without the
  chosen basis, overlapping atoms, or files over the size limit: rejected with a message.
- Too many atoms, orbitals or qubits for exact simulation (path D): a capability report with
  suggestions such as a smaller active space. The problem is never shrunk silently.
- Neural or regression inference on a molecule outside the trained families: refused. A trained
  family entered by hand (e.g. H₂ along the x axis) is recognised and gets the trained models.
- IonQ jobs wider than 29 qubits: refused before submission. Paid hardware: never offered.
