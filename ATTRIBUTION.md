# Attribution

This effort is supported by Qollab & IonQ.

The code was developed during the hackathon window with AI coding assistants (Claude Code).

GenQAI is released under the MIT licence (see `LICENSE`). All code in this repository was written
during the Qollab × IonQ hackathon (Oct 10–11, 2026). No starter code was used, and no source code
was copied from the projects listed under "Reference only".

## Scientific basis

- ADAPT-GQE: Quantinuum, NVIDIA and Pfizer, generative quantum AI for molecular circuit synthesis,
  [arXiv:2607.22468](https://arxiv.org/abs/2607.22468). We reproduce its architecture at small scale
  (Hamiltonian-conditioned transformer, ADAPT-VQE teacher, GRPO, self-distillation). The model
  sizes, molecules and hardware are different.
- ADAPT-VQE: Grimsley, Economou, Barnes, Mayhall, *Nat. Commun.* 10, 3007 (2019).
- GRPO: Shao et al., DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).

## Libraries used as dependencies (not vendored)

| Library | Licence | Used for |
| --- | --- | --- |
| [Qiskit](https://github.com/Qiskit/qiskit) | Apache-2.0 | Circuits, Pauli operators, transpilation, statevector reference |
| [Qiskit Nature](https://github.com/qiskit-community/qiskit-nature) | Apache-2.0 | Fermion-to-qubit mapping, UCCSD excitation pool, Hartree–Fock state |
| [Qiskit Algorithms](https://github.com/qiskit-community/qiskit-algorithms) | Apache-2.0 | `AdaptVQE` cross-check of our ADAPT teacher (tests only) |
| [qiskit-ionq](https://github.com/qiskit-community/qiskit-ionq) | Apache-2.0 | IonQ simulator execution, native-gate transpilation |
| [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) | Apache-2.0 | Local finite-shot and noisy simulation |
| [PySCF](https://github.com/pyscf/pyscf) | Apache-2.0 | RHF, CCSD, CASCI/FCI, active-space integrals |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Transformer generator and RL training |
| [NumPy](https://numpy.org) / [SciPy](https://scipy.org) | BSD-3-Clause | Linear algebra, BFGS |
| [Matplotlib](https://matplotlib.org) | Matplotlib (PSF-based) | Figures |

## Reference only (design ideas, no code copied)

| Project | Licence | What we took |
| --- | --- | --- |
| [Quantum-Buddies/Conditional_GQE](https://github.com/Quantum-Buddies/Conditional_GQE) | MIT | Overall conditional-GQE design: Hamiltonian conditioning, energy reward, caching |
| [moken20/gqe-for-qsci](https://github.com/moken20/gqe-for-qsci) | see repo | GRPO with replay for circuit generation |
| [PennyLane GQE demo](https://pennylane.ai/qml/demos/gqe_training) | Apache-2.0 | Energy-guided transformer training loop |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | MIT | Minimal decoder-only transformer layout |

If code from any of these is adapted later, it must be listed here with its file path, and its
copyright notice kept in that file.
