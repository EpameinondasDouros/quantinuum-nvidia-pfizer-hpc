# GenQAI demo video: script and recording plan (2:30)

Not recorded yet. This is the plan for whoever records it. Every number on screen must come from a run
made during the recording, or from a committed results file that is named on screen. Never retype a
number from this script. Placeholders like ⟨measured⟩ mean "read it off the screen".

## Before recording

- Chrome (Qollab needs WebAssembly JSPI). Log in to Qollab, open the draft project
  `qollab.xyz/u/udsy/genqai-one-shot-quantum-chemis`, and press Run once to warm the package cache.
- Locally: `streamlit run app/workbench.py` (repo root, venv active), browser at 1440×900, zoom 110%.
- Have `figures/fig7_budget_frontier.png`, `figures/fig8_learning_curve.png` and `figures/fig6_ionq.png` open in tabs.
- Close notifications. No API keys may be visible: the sidebar shows "IonQ key: present", never the key.

## Shot list

| Time | Screen | Voice-over (about 150 words a minute) |
| --- | --- | --- |
| 0:00–0:15 | README title, then the pipeline diagram | "Quantinuum, NVIDIA and Pfizer showed a transformer can write a molecule's quantum circuit in one shot, trained on ADAPT-VQE. We rebuilt that whole pipeline in Qiskit, and we tested it hard." |
| 0:15–0:40 | Workbench → Molecule tab: type SMILES `O` (water). Show the 3D conformer, formula, electrons, the active-space controls, the qubit estimate, and the capability states | "Any small molecule goes in: SMILES, XYZ or coordinates. GenQAI builds the conformer, the active-space Hamiltonian, and reports what it can and can't do. Water has no trained generator, so it's routed to ADAPT-VQE. We don't pretend the model generalises." |
| 0:40–1:05 | Switch to the H₄ preset at an unseen geometry. Generate tab: "auto" picks kernel ridge (show the reason), then run the transformer too. Show the operator/angle table and the circuit | "For a trained family, one forward pass gives every operator and angle, with zero quantum evaluations to propose it. Our default is kernel-ridge regression, because on untouched test geometries it beat our transformer. That's in the README." |
| 1:05–1:30 | Grade tab: energy error vs FCI, CNOTs, depth, measurement groups, shots for 1.6 mHa; the Pareto plot; then Optimize → min two-qubit gates, before/after | "Every circuit is graded independently: validity first, then accuracy against exact diagonalisation, then resources. A shallow but wrong circuit can never win." |
| 1:30–2:00 | Qollab tab: press Run on IonQ Forte 1 (free simulator). The console shows the live PySCF Hamiltonian, the 20 ms generation and the packed single-circuit job; read off the measured error and job id | "This runs in Qollab's browser. IonQ allows one single-circuit job, so we pack every measurement basis onto separate qubits of one circuit. H₂ on Forte 1's noise model: ⟨measured⟩ millihartree." |
| 2:00–2:20 | fig6_ionq, then fig7_budget_frontier | "Noise is the real cost: LiH's 172-CNOT circuit is hundreds of millihartree off under noise. Our RL cut H₄ circuits by about 20%, but a simpler regression is still better at the same gate count." |
| 2:20–2:30 | README findings section | "What works: fixing orbital signs, one-shot proposals for trained families, and honest grading. Everything is open and reproducible." |

## Must-not-say list

- That the transformer beats kernel ridge, that GRPO improved held-out accuracy, or quantum advantage.
- "Zero cost": say "zero quantum evaluations to propose the circuit".
- Any IonQ hardware claim. Everything shown runs on simulators.
