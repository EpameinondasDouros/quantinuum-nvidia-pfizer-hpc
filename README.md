# GenQAI: a molecular quantum-circuit workbench built on one-shot circuit generation

Qollab × IonQ Global Quantum Hackathon, Oct 10–11, 2026. MIT licence.

Quantinuum, NVIDIA and Pfizer's ADAPT-GQE trains a transformer on ADAPT-VQE circuits, so that for a
new molecular geometry it writes the whole circuit, operators and rotation angles, in one forward pass
instead of running ADAPT's operator search and VQE loop. We rebuilt the complete pipeline at small
scale in Qiskit: chemistry → ADAPT teacher → Hamiltonian-conditioned transformer → GRPO →
self-distillation → budget-conditioned generation → IonQ execution. We then benchmarked it against
every baseline a careful reviewer would ask for, on geometries nobody looked at during development.

GenQAI is now a workbench around that idea. Enter a molecule as a preset, SMILES, XYZ or coordinates.
It builds the active-space Hamiltonian, decides whether a trained model actually covers the problem,
proposes a complete circuit, grades it independently, optionally optimises it, runs it on a simulator
or IonQ's cloud simulators, and exports everything.

**Run it in the browser:** the Qollab project (`qollab/`) builds the Hamiltonian for any bond length
live with PySCF, generates the circuit with the transformer (NumPy, about 20 ms), scores it exactly,
compares it with ADAPT-VQE's cost, and measures it on the backend you pick, including IonQ's Forte 1
noise-model simulator.

## What we found

All numbers below are on **test_fresh**: 14–20 random geometries per molecule, drawn with a fixed seed
after development finished and scored once. "Quantum evals" counts energy evaluations plus ADAPT
pool gradients per geometry. Chemical accuracy is |E − E_FCI| ≤ 1.6 mHa.

| H₄ chain, 8 qubits | median error (mHa) | worst (mHa) | ≤ 1.6 mHa | CX | quantum evals |
| --- | --- | --- | --- | --- | --- |
| Hartree–Fock | 157 | 212 | 0% | 0 | 0 |
| CCSD (classical) | 0.10 | 4.1 | 80% | – | 0 |
| UCCSD-VQE | 0.60 | 4.4 | 80% | 736 | 37 |
| Fresh ADAPT-VQE (the method being replaced) | 0.000 | 0.10 | 100% | 865 | **949** |
| Warm-started ADAPT from nearest training geometry | 0.000 | 0.10 | 100% | 898 | 145 |
| Copy nearest training circuit (naive) | 4.2 | **520** | 35% | 898 | 0 |
| Copy nearest training circuit, orbital signs fixed | 1.05 | 4.3 | 65% | 898 | 0 |
| **Transformer, one shot** | 0.56 | 14.6 | 65% | 923 | **0** |
| Transformer, best of 16 | 0.24 | 12.8 | 80% | 923 | 16 |
| Transformer + angle refinement | 0.002 | 0.57 | 100% | 923 | 48 |
| Kernel-ridge angle regression, one shot (simple baseline) | **0.014** | **0.59** | **100%** | 923 | **0** |

1. **One-shot circuit generation works.** With zero quantum evaluations, the generated circuits land
   at a median of 0.002 mHa (LiH) to 0.56 mHa (H₄) from the exact energy. Fresh ADAPT needs 155–949
   quantum evaluations per geometry to get there. A short angle refinement (9–50 evaluations) takes
   every geometry to chemical accuracy: 9× cheaper than fresh ADAPT on LiH, 20× on H₄.
2. **The decisive ingredient is not the transformer, it's fixing the orbital sign gauge.** PySCF's
   orbital signs flip arbitrarily between neighbouring geometries, and the optimal angles flip with
   them. Fix that gauge (and give each molecule one canonical operator sequence), and a plain kernel
   ridge regressor on the Hamiltonian coefficients matches or beats the transformer at every
   training-set size (`figures/fig8_learning_curve.png`). Leave it unfixed and copying the nearest
   circuit misses by up to 520 mHa.
3. **RL did not beat imitation.** Energy-reward GRPO and self-distillation did not improve held-out
   error (`results/*/grpo_*.jsonl`). GRPO is what made **budget-conditioned** generation work: one
   model emits shorter circuits on request, about 720 CX instead of 914 on H₄. But regression on a
   fixed 14-operator prefix does better at the same gate count (`figures/fig7_budget_frontier.png`).
4. **Nothing one-shot extrapolates.** Beyond the training range (LiH at 4.1–4.4 Å) every one-shot
   method misses, by 6–150 mHa. Refinement and ADAPT don't. That is why LiH shows 71% for all one-shot
   methods: the 10 in-range points are all correct and the 4 extrapolated ones all fail.
5. **On IonQ, gate count is what matters.** H₂'s 4-CX circuit measured +7.8 ± 5.8 mHa (r = 1.33 Å,
   through Qollab, job `01a12826-5478-72af-b517-c3e60f663d79`) and +25.0 ± 6.4 mHa (r = 0.8 Å, via the
   API, job `01a12845-095c-73f9-b107-03d2aa0cef79`) on IonQ's Forte 1 noise model. Treat these as two
   samples, not a calibrated error. A custom molecule ran too: HeH⁺ (ADAPT circuit, 4 groups packed
   on 8 qubits) gave +36.1 ± 11.0 mHa, job `01a12858-0690-7546-94e7-76cddf9f0e3f`. LiH's 172-CX circuit is about 460 mHa off under similar noise
   (`figures/fig6_ionq.png`).
6. **Hardware-aware optimisation moves the gate frontier, but it isn't free.** Pruning operators and
   re-optimising the angles reaches 1.6 mHa on 100% of held-out geometries with H₄ at about 600–670 CX
   (from 923) and LiH-6q at 162 CX (from 298), below every zero-evaluation option at the same accuracy.
   It costs about 250–310 (LiH-6q) and 2,400–2,900 (H₄) energy evaluations per geometry, as much as or
   more than a fresh ADAPT run. So it's a different cost class from one-shot generation, and the
   starting generator stops mattering once evaluations are allowed (`results/hw_opt.md`,
   `figures/fig9_hw_optimization.png`).

What we do **not** claim: that the transformer beats simpler regressors here, that anything is faster
in wall-clock time at 2–8 qubits (exact diagonalisation is instant), or quantum advantage. At this
scale the honest contribution is a complete, open, Qiskit-native reproduction of the architecture,
an untouched-test benchmark against strong baselines, and the gauge-fixing finding, which any
learned or reuse-based circuit generator for molecules has to handle.

## Why "zero quantum evaluations" isn't zero cost

One-shot generation makes no quantum energy evaluations *to propose the circuit*: the operators and
angles come from one model call. Everything around it costs something, and the workbench reports each
part separately:
- building the Hamiltonian (classical SCF and integrals);
- training the model (the ADAPT teacher runs and model fitting, amortised over new geometries in
  `figures/fig5_breakeven.png`);
- grading against an exact reference;
- best-of-K ranking (K energy evaluations);
- refinement;
- measuring the energy on a device (shots × measurement groups).

## The workbench

```bash
streamlit run app/workbench.py        # local; full chemistry, RDKit, PyTorch, IonQ cloud simulators
```

| Step | What happens | Code |
| --- | --- | --- |
| Molecule | Preset, SMILES (RDKit ETKDGv3, seeded conformers, optional MMFF/UFF), XYZ (kept verbatim) or atom table; charge, multiplicity, basis, frozen core, active space, mapping, all validated. Open shells use ROHF | `genqai/molecule.py` |
| Capability report | Formula, electrons, qubits, whether exact FCI is feasible, packed IonQ width, limits. "Input accepted", "Hamiltonian built", "trained model covers it", "fallback available" and "backend can run it" are shown as separate answers | `molecule.assess` |
| Hamiltonian | Frozen-core active space from one RHF (only the PySCF parts that also run in the browser), FCI by sector diagonalisation, CCSD where available. Presets reproduce the committed data exactly | `genqai/chemistry.py` |
| Routing | **A**: a trained model covers it and it's inside the validated range. **B**: compatible but out of range, so labelled experimental, with refine or ADAPT recommended. **C**: no compatible model, so ADAPT/UCCSD runs and the circuit is saved as potential teacher data. **D**: too big, with suggestions instead of a silently smaller problem. The default method is the best on the VAL split; that's kernel ridge on every trained system | `genqai/registry.py`, `models/registry.json` |
| Generate | Transformer (SFT / GRPO / budget), kernel ridge, nearest-neighbour, ADAPT, UCCSD, with "+refine" and best-of-K. Each run logs forward passes, candidates and quantum evaluations to propose | `genqai/generate.py` |
| Grade | Stage 1: validity (qubits, operators, finite angles, norm, particle number and S_z, no energy below the exact ground state) and accuracy vs FCI, or labelled unverified when FCI isn't available. Stage 2: resources (gates, 2-qubit gates, depth, CX, IonQ-native counts where the compiler gives them, measurement groups, shots for a target error) ranked on a Pareto front among circuits that meet the target only | `genqai/grading.py` |
| Optimise | Angle refinement, exact merging of repeated operators, operator pruning with re-refinement, transpiler level 3, noise-aware selection. The original is kept and the same Hamiltonian is checked | `genqai/optimize.py` |
| Execute | Statevector, local shots, local noisy (our own approximation), IonQ cloud simulators (ideal, Forte 1, Aria 1). Paid hardware is never offered. Width is checked before anything is submitted | `genqai/backends.py`, `genqai/execution.py` |
| Export and history | Qiskit script (reproduces the statevector to 1e-10), OpenQASM 2/3, Hamiltonian/metadata JSON, measurement and benchmark CSV; experiments saved locally with a configuration hash | `genqai/export.py`, `genqai/store.py`, `genqai/jobs.py` |

Worked examples that run end to end (`tests/test_e2e.py`, `docs/custom_molecules.md`): HeH⁺ from XYZ,
water from SMILES and the OH radical (open shell) go to path C. ADAPT reaches FCI on each, and FCI
matches PySCF CASCI to 1e-8. H₂ typed in by hand along any axis is recognised as the trained H₂ family
(path A, kernel ridge). LiH beyond 4 Å goes to path B, labelled experimental.

What runs where: the Qollab bundle (`qollab/main.py`) runs in the browser on the trained presets, with
a live PySCF Hamiltonian, NumPy inference, grading and the IonQ run. Arbitrary molecules, RDKit
conformers, the full solver fallback and the workbench UI are local only: Qollab has no RDKit or
PyTorch, and its PySCF build lacks CASCI/FCI/CC. Nothing depends on a private server.

## Systems

| System | Active space | Mapping | Qubits | Geometries (train / val / test / test_fresh) |
| --- | --- | --- | --- | --- |
| `h2` | CAS(2e, 2o) | parity, 2-qubit reduced | 2 | 21 grid + 14 fresh |
| `lih4` | CAS(2e, 3o), frozen core | parity | 4 | 76 grid + 14 fresh |
| `lih6` | CAS(2e, 4o), σ + one π | parity | 6 | 46 grid + 14 fresh |
| `h4` | CAS(4e, 4o), linear chain (a, b, a) | Jordan–Wigner | 8 | 64 grid + 20 fresh |

STO-3G throughout. Splits are assigned per geometry before any labels exist: interpolation points, a
held-out contiguous interval, extrapolation beyond the range, and test_fresh.

## How to run

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                                  # see "Tests" below
.venv/bin/streamlit run app/workbench.py                       # the workbench

.venv/bin/python scripts/run_pipeline.py --quick               # H2 + LiH from scratch, about 90 s
.venv/bin/python scripts/run_pipeline.py --skip-train          # everything, using committed checkpoints
.venv/bin/python scripts/demo.py --system lih4 --geometry 2.47 [--ionq]
.venv/bin/python scripts/make_figures.py                       # figures/*.png + figures/summary.md
.venv/bin/python scripts/report_fair.py                        # the test / test_fresh tables
```

IonQ cloud runs need `IONQ_API_KEY` in the environment or in a gitignored `.env`.
`notebooks/genqai_demo.ipynb` is the local notebook version of the demo.

**On Qollab:** upload `qollab/` (rebuild it with `scripts/build_qollab_bundle.py`), press Run, pick a
backend, fill in `system`, `r` (or `a`, `b` for H₄), `shots`. Qollab runs Python in the browser on
Pyodide. Its PySCF build lacks CASCI/FCI/CC and its IonQ gateway takes one single-circuit job per run,
so we build the Hamiltonian from RHF integrals and pack all measurement groups into one circuit.
Details are in `docs/qollab_runtime_checklist.md`.

## Pipeline

| Stage | Code | Output |
| --- | --- | --- |
| Hamiltonians, HF / CCSD / FCI | `genqai/chemistry.py`, `scripts/make_hamiltonians.py` | `data/<s>/hamiltonians*.jsonl` |
| ADAPT-VQE teacher with per-step cost logs | `genqai/adapt.py`, `scripts/make_teacher.py` | `data/<s>/teacher*.jsonl` |
| Gauge-fixed canonical labels | `genqai/features.py`, `scripts/make_canonical_teacher.py` | `data/<s>/teacher_canonical.jsonl` |
| Transformer (operator head + von Mises angle head) | `genqai/model.py`, `scripts/train_sft.py` | `checkpoints/<s>/sft.pt` |
| GRPO, self-distillation, budget conditioning | `genqai/rl.py`, `scripts/train_grpo.py`, `scripts/budget_frontier.py` | `checkpoints/<s>/{grpo,distill,budget}.pt` |
| Baselines | `genqai/baselines.py`, `genqai/regression.py`, `scripts/run_baselines.py` | `results/<s>/*.jsonl`, `results_fresh/` |
| Scoring harness | `genqai/evaluate.py`, `genqai/sim.py` (exact, adjoint gradients) | EvalRecords |
| Grouped measurement, IonQ | `genqai/measure.py`, `scripts/run_ionq_demo.py` | `results/<s>/ionq_demo_*.json` |
| NumPy inference for the browser | `genqai/infer_numpy.py` | `qollab/checkpoints/` |
| Molecule input, routing, generation, grading, optimisation, execution | see "The workbench" | `runs/experiments/` (local) |
| Hardware-aware optimisation study | `scripts/run_hw_optimization.py` | `results/hw_opt.md`, `results*/<s>/hw_opt.jsonl` |
| Figures | `genqai/figures.py` | `figures/` |

Conventions (`genqai/schema.py`): Qiskit qubit order, total energies in Hartree, operator IDs index
`genqai.pool.get_pool(system)`, and a circuit is HF followed by exp(−iθ_t G_{op_t}) in order.

## Figures

| | |
| --- | --- |
| `fig1_pes` | Energy vs bond length with residuals against FCI |
| `fig2_test_errors` | Per-geometry error by method, with cost |
| `fig3_accuracy_cost` | Accuracy vs CX count and vs quantum evaluations |
| `fig4_training` | Held-out error during SFT / GRPO / distillation |
| `fig5_breakeven` | Cumulative cost vs number of new geometries |
| `fig6_ionq` | Measured energies on local and IonQ simulators |
| `fig7_budget_frontier` | Accuracy vs CX budget: budget policy vs truncated ADAPT vs regression |
| `fig8_learning_curve` | Accuracy vs training-set size |
| `fig9_hw_optimization` | Prune + refine vs zero-evaluation methods: CX, accuracy and evaluation cost |

## Tests

`.venv/bin/python -m pytest -q`: **222 passed, 4 skipped** (about 60 s, Oct 11, 2026). The skips:
- 3 need the optional `qiskit_qasm3_import` package to load exported QASM 3 back;
- 1 is the live IonQ job, which runs only with `GENQAI_LIVE_IONQ=1` and a key.

Coverage includes the preset benchmark Hamiltonians (exact to 1e-12 through the generic molecule path),
FCI = PySCF CASCI, packed = grouped = statevector measurement (to 1e-12 on analytic Hamiltonians),
error-bar calibration, routing paths A–D, refusal of incompatible models, grading status logic, the
optimisation invariants, export reproducibility, and the Streamlit app (headless).

## Known limitations

- Trained models cover four fixed active-space problems (H₂, LiH 4q/6q, H₄ chain) along their
  geometry families. Other molecules run through ADAPT/UCCSD, not a trained generator.
- One-shot accuracy holds inside the training range only; beyond it, refine or run ADAPT.
- Packed single-circuit IonQ measurement needs groups × qubits ≤ 29, which is in practice H₂ only.
  Larger systems run on local simulators.
- Open-shell pools conserve S_z but not S².
- Local noisy simulation is our own depolarising model, not IonQ's. IonQ numbers come from IonQ's simulators.
- Optimisation evaluation counts treat one BFGS gradient call as one evaluation; on hardware,
  parameter-shift gradients cost about 2 × the number of angles.

## Credits

Reused components and licences are listed in `ATTRIBUTION.md`. The team tracker is section 0 of
`docs/genqai-implementation-plan.md`.
