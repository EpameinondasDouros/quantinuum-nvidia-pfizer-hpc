# GenQAI: one-shot quantum chemistry circuits, reproduced and stress-tested

Qollab × IonQ Global Quantum Hackathon, Oct 10–11, 2026. MIT licence.

Quantinuum, NVIDIA and Pfizer's ADAPT-GQE trains a transformer on ADAPT-VQE circuits, so that for a
new molecular geometry it writes the whole circuit, operators and rotation angles, in one forward pass
instead of running ADAPT's operator search and VQE loop. We rebuilt the complete pipeline at small
scale in Qiskit: chemistry → ADAPT teacher → Hamiltonian-conditioned transformer → GRPO →
self-distillation → budget-conditioned generation → IonQ execution. We then benchmarked it against
every baseline a careful reviewer would ask for, on geometries nobody looked at during development.

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
5. **On IonQ, gate count is what matters.** H₂'s 4-CX circuit measures +7.8 ± 5.8 mHa on IonQ's
   Forte 1 noise model via Qollab. LiH's 172-CX circuit is about 460 mHa off under the same kind of noise
   (`figures/fig6_ionq.png`).

What we do **not** claim: that the transformer beats simpler regressors here, that anything is faster
in wall-clock time at 2–8 qubits (exact diagonalisation is instant), or quantum advantage. At this
scale the honest contribution is a complete, open, Qiskit-native reproduction of the architecture,
an untouched-test benchmark against strong baselines, and the gauge-fixing finding, which any
learned or reuse-based circuit generator for molecules has to handle.

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
.venv/bin/python -m pytest -q                                  # 79 tests, about 40 s

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

## Credits

Reused components and licences are listed in `ATTRIBUTION.md`. The team tracker is section 0 of
`docs/genqai-implementation-plan.md`.
