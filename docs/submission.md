# Submission material (SUBMITTED Oct 11, 2026)

Status: Qollab project published as v1.0 (public) and submitted to the Qollab × IonQ hackathon by team The Quantubians (theme: Quantum Impact). Demo video: https://drive.google.com/file/d/1oOhMOhFqhNBd3pZ_hhlYvR5wS0-tAtYF/view. Website: https://udsy19.github.io/genqai/. Public code: https://github.com/udsy19/genqai.

## Project title
GenQAI: one-shot quantum chemistry circuits, a molecular circuit workbench

## One-line description (≤ 240 chars, used on the Qollab card)
A model trained on ADAPT-VQE writes a molecule's whole ground-state circuit (operators + angles) in one
call, with no VQE loop. It runs in your browser on Qollab and measures on IonQ simulators, benchmarked honestly.

## Project page (Qollab write-up)

**What it does.** Quantinuum, NVIDIA and Pfizer's ADAPT-GQE trains a model on ADAPT-VQE circuits so
that, for a new molecular geometry, it writes the complete state-preparation circuit (every operator and
rotation angle) in one forward pass, skipping ADAPT's operator search. We rebuilt that pipeline in
Qiskit, at hackathon scale, and wrapped it in a workbench: molecule in → Hamiltonian → "does a trained
model cover this?" → circuit → independent grading → optimisation → execution on simulators or IonQ.

**Press Run.** This project builds the H₂ / LiH / H₄ Hamiltonian for the bond length you type, live,
with PySCF in your browser. The trained model (NumPy, about 20 ms) proposes the circuit with zero quantum
evaluations. The circuit is scored against exact diagonalisation and against kernel ridge,
nearest-neighbour reuse and a live ADAPT-VQE run (with its evaluation count), then measured on the
backend you picked. IonQ accepts one single-circuit job per run, so all measurement bases are packed
onto separate qubits of one circuit.

**What we measured** (on untouched geometries, drawn after development and scored once):
- One-shot proposals land at a median 0.002 mHa (LiH) to 0.56 mHa (H₄) from the exact energy, with
  zero quantum evaluations, where fresh ADAPT needs 155–949 per geometry. A short angle refinement takes
  every geometry within chemical accuracy at 9–20× less than ADAPT.
- The decisive ingredient is fixing PySCF's orbital sign gauge. With it, a plain kernel-ridge
  regressor matches or beats our transformer at every training-set size (H₄: 100% vs 65% within
  1.6 mHa), so kernel ridge is the workbench's default. Without it, reusing the nearest circuit misses
  by up to 520 mHa.
- RL (GRPO) and self-distillation did not improve held-out accuracy. Budget-conditioned GRPO cut H₄
  circuits from 914 to about 720 CX, but regression on a shorter fixed circuit is better at the same
  gate count.
- Pruning operators and re-optimising the angles gets H₄ to about 600–670 CX (from 923) and LiH-6q to
  162 (from 298) at 100% chemical accuracy, at the price of hundreds to thousands of energy evaluations.
- One-shot methods fail beyond the training range; the workbench labels those cases experimental and
  recommends refinement or ADAPT.
- On IonQ's Forte 1 noise model: H₂ (4 CX) +7.8 ± 5.8 and +16.8 ± 6.0 mHa (two Qollab runs at 1.33 Å) and +25.0 ± 6.4 mHa (API, 0.8 Å).
  Custom HeH⁺ gave +36.1 ± 11.0 mHa. LiH's 172-CX circuit is about 460 mHa off under noise. Gate count
  dominates.

**What we don't claim:** that the transformer beats simple regression here, wall-clock speedups at 2–8
qubits, or quantum advantage.

**Beyond this page** (local workbench, `streamlit run app/workbench.py`): SMILES/XYZ input with RDKit
conformers, charge/spin/active-space control, routing into validated / experimental / new-domain /
too-large paths, two-stage grading with a Pareto front, optimisation, exports (Qiskit, QASM, JSON, CSV).

Code, data, checkpoints, tests and figures: https://github.com/udsy19/genqai (MIT).
Attribution: ATTRIBUTION.md.

## Tags
quantum-chemistry, vqe, adapt-vqe, generative-ai, ionq, qiskit, benchmarking

## Video
Script and shot list: `docs/video_script.md`. **Not recorded yet.**

## Before publishing (manual steps for the team)
1. The draft holds the bundle as of commit `a249f96` (verified Oct 11 on the built-in simulator and IonQ
   Forte 1). Re-upload only if `qollab/` changes again (`git diff a249f96 -- qollab`).
2. Paste the page text above into the project card. The draft's write-up currently comes from
   `qollab/README.md`.
3. Record and upload the video, add the link.
4. Press Publish on Qollab and submit through the hackathon form before Oct 11, 7 PM ET.
