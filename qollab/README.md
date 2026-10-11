# GenQAI: one-shot quantum chemistry circuits

**Team The Quantubians** · Qollab × IonQ Global Quantum Hackathon, Oct 9–11, 2026 · MIT licence ·
Code, data and the full local workbench: https://github.com/udsy19/genqai
**Demo video (2:23):** https://drive.google.com/file/d/1oOhMOhFqhNBd3pZ_hhlYvR5wS0-tAtYF/view · **Website:** https://udsy19.github.io/genqai/

## What it does

Quantinuum, NVIDIA and Pfizer's ADAPT-GQE trains a model on ADAPT-VQE circuits, so that for a new
molecular geometry it writes the **complete** ground-state circuit (every operator and rotation angle) in
one forward pass, skipping ADAPT's operator search and VQE loop. We rebuilt that pipeline in Qiskit at
hackathon scale: chemistry → ADAPT teacher → Hamiltonian-conditioned transformer → GRPO →
self-distillation → budget-conditioned generation → IonQ execution. We then benchmarked it against
strong baselines on geometries nobody looked at during development.

**Press Run** (pick a backend, fill in the form) and this project:
1. builds the molecular Hamiltonian for the bond length you type, live in your browser with PySCF, and
   prints HF and exact FCI energies;
2. generates the circuit with our trained model (NumPy, about 20 ms) with **zero quantum evaluations
   to propose it**;
3. scores it exactly against FCI, next to kernel-ridge regression, nearest-neighbour reuse and a live
   ADAPT-VQE run (with the quantum evaluations ADAPT needed);
4. measures its energy on the backend you chose. Qollab's IonQ gateway takes one single-circuit job per
   run, so every measurement basis is packed onto separate qubits of **one** circuit (H₂: 2 groups on
   4 qubits).

## How to run

- Backend: the **built-in simulator** (instant) or a free **IonQ simulator** (Forte 1 / Aria noise
  models). IonQ takes about 15 s.
- Form fields:
  - `system`: `h2` (2 qubits), `lih4` (4), `lih6` (6) or `h4` (8; uses `a`, `b`, the chain spacings a-b-a);
  - `r`: the bond length in Å, any value;
  - `shots`: shots per measurement group;
  - `live_adapt`: also run ADAPT-VQE for the cost comparison.
- IonQ fits **h2 only** (other systems need more than 29 qubits once packed, and the program says so).
  Use the built-in simulator for the others.
- If a second run straight after another fails with "file was not loaded … I/O error", reload the page
  and run again.

## What we found (untouched test geometries, scored once)

| H₄ chain, 8 qubits | median error | worst | within 1.6 mHa | CX | quantum evals |
| --- | --- | --- | --- | --- | --- |
| Fresh ADAPT-VQE | 0.000 mHa | 0.10 | 100% | 865 | 949 |
| Our transformer, one shot | 0.56 mHa | 14.6 | 65% | 923 | **0** |
| Transformer + angle refinement | 0.002 mHa | 0.57 | 100% | 923 | 48 |
| Kernel-ridge regression, one shot | **0.014 mHa** | **0.59** | **100%** | 923 | **0** |
| Copy nearest circuit, orbital signs not fixed | 4.2 mHa | 520 | 35% | 898 | 0 |

- **One-shot proposals work** for trained molecules: median 0.002 mHa (LiH) to 0.56 mHa (H₄) with zero
  quantum evaluations, where ADAPT needs 155–949 per geometry.
- **The key ingredient is fixing PySCF's orbital sign gauge.** With it, plain kernel-ridge regression
  matches or beats our transformer at every training-set size, so we say so. Without it, reusing the
  nearest circuit misses by up to 520 mHa.
- **RL (GRPO) and self-distillation did not improve held-out accuracy.** Budget-conditioned GRPO cut H₄
  circuits from 914 to about 720 CX, but regression on a shorter fixed circuit was better at the same
  gate count.
- **Pruning plus re-optimisation** reaches 100% chemical accuracy at about 600–670 CX on H₄ (from 923),
  at the cost of thousands of energy evaluations.
- **One-shot methods fail beyond the training range**; refinement or ADAPT recovers.
- **On IonQ's Forte 1 noise model**, H₂ (4 CX) measured +7.8 ± 5.8, +16.8 ± 6.0 and +25.0 ± 6.4 mHa
  across three runs, and a custom HeH⁺ +36.1 ± 11.0 mHa. LiH's 172-CX circuit is about 460 mHa off
  under noise. Gate count dominates on hardware.

We don't claim the transformer beats simple regression here, wall-clock speedups at 2–8 qubits, or
quantum advantage.

## Beyond this page

The GitHub repo also has a local workbench (`streamlit run app/workbench.py`):
- any small molecule from SMILES or XYZ (RDKit conformers), with charge, spin and active-space control;
- routing into validated / experimental / new-molecule / too-large paths, with ADAPT for molecules
  without a trained model;
- two-stage circuit grading with a Pareto front, optimisation, and exports (Qiskit, QASM, JSON, CSV).

There are 222 automated tests.

## Attribution

All code was written during the hackathon window, with no starter code. Inspired by the Quantinuum /
NVIDIA / Pfizer ADAPT-GQE work and ADAPT-VQE (Grimsley et al., 2019). Built on Qiskit, Qiskit Nature,
PySCF, qiskit-ionq and NumPy/SciPy. Design ideas (no code) came from Conditional_GQE, gqe-for-qsci, the
PennyLane GQE demo and nanoGPT; licences are in ATTRIBUTION.md in the repo.
