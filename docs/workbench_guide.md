# GenQAI workbench guide

The workbench is a local Streamlit app for the full loop: molecule in, Hamiltonian, routing to a trained
generator or a conventional solver, circuit generation, grading, optimization, execution on simulators or
IonQ, comparison, and export.

## Run it

```bash
pip install -e ".[app]"                 # streamlit + py3Dmol (rdkit, pyscf, qiskit come with the base install)
PYTHONPATH=$PWD streamlit run app/workbench.py
```

IonQ cloud simulators need `IONQ_API_KEY` (or `QISKIT_IONQ_API_TOKEN`) in the environment or in `.env`.
The key is never displayed. Paid QPU backends are listed but can't be selected.

## Where the numbers come from

Every panel says where its numbers come from:

- **Live computation in this session**: Hamiltonians, generated circuits, grades, executions.
- **Committed results files**: the Benchmarks tab only (`figures/`, `results_fresh/`).
- **Not available in this build**: a workbench module (lane) isn't installed. The panel says which one.
  Nothing is mocked or filled with placeholder numbers.

## Sidebar: environment

The sidebar lists which libraries this environment has: RDKit, full PySCF (including CCSD), PyTorch,
py3Dmol, Qiskit Aer, qiskit-ionq, and whether an IonQ key is present. It also lists which workbench
modules are installed (`genqai.molecule`, `registry`, `generate`, `grading`, `optimize`, `backends`,
`execution`) and the features that follow from them.

## Tabs

**Molecule.** You can enter a molecule four ways:

- a preset (H2, LiH 4q/6q, H4 chain), with geometry sliders
- a SMILES string, with an RDKit ETKDGv3 conformer seed, an optional MMFF/UFF optimizer and a conformer
  picker
- an XYZ upload or paste (size-limited)
- the atom editor

Charge, multiplicity, basis, active space, mapping and frozen core sit under "Active space and mapping".
"Use defaults" gives presets exactly the trained system's settings. The cheap capability report shows:

- formula and electrons
- estimated qubits
- whether an exact reference is feasible
- the IonQ packed-width fit, once the Hamiltonian is built

The capability table separates five states:

- input accepted
- Hamiltonian built
- trained-model inference supported (routing path A validated, B experimental/out of domain, C no
  compatible generator, D exceeds resources, with reasons)
- conventional fallback available
- backend execution supported

**Build Hamiltonian** runs in a background job. If you change the input after building, the table shows
"Hamiltonian built: no" until you build again. The 3D view uses py3Dmol.

**Generate.** Choose a method:

- auto: the registry's recommendation, picked on validation metrics, with the reasons shown
- transformer SFT/GRPO/budget, with a CX-budget slider for the budget model
- KRR
- nearest-neighbour reuse
- angle interpolation
- ADAPT-VQE
- UCCSD-VQE

You can also add `+refine` or best-of-K where the routing decision allows it. The tab shows:

- the operator/angle table
- an optional circuit diagram
- the generation log

The log statement is precise. *Quantum evaluations to propose* counts only what the method spent
proposing the circuit, and it is 0 for a genuine one-shot model. Building the Hamiltonian and grading
are separate costs. Refinement energy evaluations and ADAPT pool gradients are reported separately.

**Grade.** The energy, resources, cost and noise panels come from `genqai.grading.GradeReport`. Until
Lane C lands, the tab falls back to exact scoring via `genqai.evaluate.score_circuit`, labelled "basic
scoring". The tab also has:

- An accuracy-vs-CX plot of every circuit graded this session for this Hamiltonian, with the Pareto
  front highlighted when Lane C is present.
- An Optimize panel showing before/after reports for the objectives accuracy, min_2q, min_depth,
  min_cost, noise and pareto. It needs Lane C.

**Execute.** This tab shows the backend catalogue (`genqai.backends.list_backends`) with availability
and reasons. Only available, free backends can be selected. Set shots, seed and packing, then choose:

- **Dry run**: width check and plan, with nothing submitted.
- **Execute**: builds an `ExecutionRecord` with the job id, energy ± stderr and status.

The results table puts the exact statevector energy from the Grade tab next to the ideal and noisy runs.

**Benchmarks.** This tab shows committed results only. The per-method test_fresh table is recomputed
from `results_fresh/<system>/*.jsonl` with `genqai.evaluate.summarize`, and the committed figures appear
with their captions.

**History & export.** "Prepare exports" generates the following for the selected circuit:

- `circuit.py`: a standalone Qiskit script that rebuilds the exact circuit
- `circuit.qasm2` and `circuit.qasm3`: transpiled to cx/rz/sx/x
- `hamiltonian.json`
- `metadata.json`: config, method, routing, generation log, software versions and SHA-256 content
  hashes
- `benchmark.csv`
- one `measurements_<i>.csv` per execution

"Save experiment" writes everything to `runs/experiments/<utc>-<method>-<config hash>/` (gitignored). The
saved-experiments list can reload, download and delete them.

Duplicate submissions are blocked twice over:

- A button is disabled while a job of its kind is running.
- The shared `genqai.jobs.JobRunner` dedupes identical jobs by content key.

Long jobs show progress and a Cancel button.

## Local workbench vs the Qollab browser bundle

| | Local workbench (`app/workbench.py`) | Qollab bundle (`qollab/main.py`) |
|---|---|---|
| Runtime | CPython with full PySCF, RDKit, torch, Aer | Pyodide (browser); PySCF without CC, no RDKit/torch |
| Input | presets, SMILES, XYZ, atom editor | form fields: preset system and geometry |
| Generation | every registry method, refine, best-of-K, ADAPT/UCCSD | one-shot NumPy transformer export + baselines |
| Backends | statevector, local shots/noisy, IonQ cloud simulators | the QPU chosen in Qollab's Select-QPU dialog |
| Persistence | `runs/experiments/` + exports | printed output and a saved PNG |

The Qollab bundle keeps its own fixed `main()` flow and does not import the workbench. The workbench
changes nothing in it.

## Implementation notes

- `app/services.py` imports each lane lazily. A lane counts as present only if its file exists in this
  checkout's `genqai/` package. This stops an editable install of another checkout from leaking in.
- Qiskit's Rust extension crashes if it is first imported on a short-lived Streamlit script thread. The
  app imports it once on the job runner's long-lived worker thread before anything else.
- Chemistry jobs run in threads, not processes. `genqai.molecule.build_problem` registers the new system
  in the process-wide `chemistry.SYSTEMS`. `JobRunner(isolate=True)` gives a hard-cancellable process for
  picklable functions that don't need that.
