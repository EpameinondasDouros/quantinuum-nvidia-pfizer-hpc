# Workbench integration contracts (phase 2)

Every lane codes against these signatures. A signature may grow optional keyword arguments; anything
else needs a note to the lead first. Each file has exactly one owning lane; other lanes import from it
and never edit it. If you need a change in a file you don't own, put it in your final report.

Non-negotiable invariants:
- Preset results stay reproducible. `build_record(SYSTEMS["h2"], (r,))` etc. must give byte-identical
  Hamiltonians and energies to `data/*/hamiltonians*.jsonl`. `tests/test_core.py` checks this.
- The Qollab path keeps working: `qollab/main.py`, `genqai/demo.py`, `genqai/infer_numpy.py`,
  `genqai/measure.py` packing. Changes there must be additive. Re-run the bundle test
  (`tests/test_infer_numpy.py::test_qollab_bundle_in_sync`) after changing any bundled module.
- Zero quantum evaluations means "to *propose* the circuit". Hamiltonian construction, scoring,
  refinement and execution are counted separately and never folded into that zero.
- Never fabricate results. Every number shown comes from a computation in this run or a committed
  results file, and is labelled with which one it is.

## Lane A: molecular chemistry (owns `genqai/molecule.py`, `genqai/chemistry.py`, `genqai/pool.py`, `genqai/schema.py`)

```python
# genqai/molecule.py
class MoleculeError(ValueError): ...           # user-facing message, never a traceback

@dataclass(frozen=True)
class MoleculeSpec:
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]   # Angstrom
    charge: int = 0
    multiplicity: int = 1                       # 2S+1; PySCF spin = multiplicity - 1
    basis: str = "sto-3g"
    name: str = ""
    source: str = "atoms"                       # "preset" | "smiles" | "xyz" | "atoms"
    smiles: str | None = None
    conformer: dict | None = None               # {"seed", "method": "ETKDGv3", "optimizer": None|"MMFF"|"UFF", "energy"}
    preset: tuple[str, tuple[float, ...]] | None = None   # ("h2", (0.74,)) when it came from a preset
    # properties: formula, n_electrons, spin, key (stable sha of physical content)

def from_smiles(smiles: str, *, charge: int | None = None, multiplicity: int | None = None,
                basis="sto-3g", seed: int = 20261010, optimizer: str | None = None,
                num_conformers: int = 1) -> list[MoleculeSpec]     # one per conformer, ETKDGv3, explicit H
def from_xyz(text: str, *, charge=0, multiplicity=1, basis="sto-3g") -> MoleculeSpec   # coordinates kept verbatim
def from_atoms(atoms, **kw) -> MoleculeSpec
def from_preset(system: str, params: tuple[float, ...]) -> MoleculeSpec
def validate_molecule(mol: MoleculeSpec) -> list[str]   # raises MoleculeError on hard errors (parity, elements, basis), returns warnings

@dataclass(frozen=True)
class ProblemConfig:
    molecule: MoleculeSpec
    active_electrons: int | tuple[int, int] | None = None   # None = auto
    active_orbitals: int | None = None
    active_orbital_indices: tuple[int, ...] | None = None    # 0-based MO indices, overrides auto
    mapping: str = "parity"                                  # "parity" (two-qubit reduced) | "jordan_wigner"
    frozen_core: bool = True
    pool: str = "uccsd"                                      # fermionic singles+doubles, the only pool for now
    # property: key (sha over molecule.key + every field). Caches must use this key.

def default_config(mol: MoleculeSpec) -> ProblemConfig      # presets -> exactly the preset's SystemSpec settings
@dataclass
class CapabilityReport:
    formula: str; n_electrons: int; charge: int; multiplicity: int
    num_qubits: int; active_electrons: tuple[int, int]; active_orbitals: int
    exact_feasible: bool                 # FCI sector diag + statevector within limits
    est_pauli_terms: int | None
    ionq_packed_qubits: int | None       # groups x qubits for the packed single-circuit scheme (needs the Hamiltonian; None before)
    fits_ionq: bool | None               # packed width <= 29
    limits: dict                         # the limits applied (max atoms, max active orbitals, ...)
    warnings: list[str]; errors: list[str]
def assess(config: ProblemConfig) -> CapabilityReport        # cheap; no SCF unless needed for orbital counts
def build_problem(config: ProblemConfig) -> HamiltonianRecord
    # Registers a SystemSpec in chemistry.SYSTEMS (name "mol-<config.key[:10]>", or the preset name for presets)
    # so get_pool / Evaluator / build_circuit work unchanged. record.system == that name.
```

`schema.HamiltonianRecord` gains optional fields with defaults, so old jsonl still loads: `charge=0`,
`multiplicity=1`, `num_alpha=-1`, `num_beta=-1` (-1 = closed shell, active_electrons/2 each),
`problem_key=""`, `reference_method="FCI"`. `SystemSpec` gains optional `charge`, `multiplicity`,
`fixed_atoms`, `active_orbital_indices`, `frozen_core`. The `mapper_for`, `hf_bitstring` and
`get_pool` helpers handle (n_alpha, n_beta) != equal. Open shell uses ROHF.

## Lane B: model registry and routing (owns `genqai/registry.py`, `genqai/generate.py`, `models/registry.json`, `scripts/build_registry.py`)

```python
@dataclass
class ModelCard:
    id: str; kind: str        # "transformer_sft" | "transformer_grpo" | "transformer_budget" | "krr" | "nn_reuse_gauge" | "angle_interp"
    system: str; checkpoint: str | None; checkpoint_sha256: str | None
    signature: dict           # formula, basis, charge, multiplicity, active (ne, norb), mapping, pool size, pauli-vocab sha, gauge convention
    family: dict              # geometry family: {"kind": "diatomic"|"linear_h4", "params": [...], "train_range": {...}}
    train_dataset: str; validation: dict   # VAL-split metrics (used for default selection) + test_fresh metrics (display only)
    modes: list[str]; limitations: list[str]

def load_registry() -> list[ModelCard]
def match_family(record: HamiltonianRecord, mol: MoleculeSpec | None) -> tuple[str, tuple[float, ...]] | None
    # e.g. an H2 typed as XYZ at 0.9 Å with the default config -> ("h2", (0.9,)); requires the full signature to match
@dataclass
class RoutingDecision:
    path: str                 # "A" validated | "B" compatible but out of domain (experimental) | "C" no compatible generator | "D" exceeds resources
    system: str | None; params: tuple | None
    compatible: list[str]     # ModelCard ids
    recommended: str          # method id, chosen by VAL metrics (never by test)
    available_methods: list[str]
    reasons: list[str]; ood: dict   # in_range per param, feature-space distance vs train
def route(config: ProblemConfig, record: HamiltonianRecord | None, report: CapabilityReport) -> RoutingDecision

@dataclass
class GenerationLog:
    method: str; model_id: str | None; forward_passes: int; candidates: int
    model_time_s: float; quantum_evals_to_propose: int    # 0 for genuine one-shot
    refine_energy_evals: int; adapt_pool_gradient_evals: int; validity: bool; experimental: bool; notes: list[str]
def generate(record: HamiltonianRecord, method: str, decision: RoutingDecision, **opts) -> tuple[GeneratedCircuit, GenerationLog]
    # methods: "auto", "transformer_sft", "transformer_grpo", "transformer_budget" (opts cnot_budget), "krr", "nn_reuse_gauge",
    #          "adapt", "uccsd_vqe", plus suffix "+refine" and "bestofk" (opts k). Torch-free path via infer_numpy when torch is missing.
```

## Lane C: grading and optimization (owns `genqai/grading.py`, `genqai/optimize.py`, `scripts/run_hw_optimization.py`)

```python
@dataclass
class GradeReport:
    status: str               # "valid_meets_target" | "valid_misses_target" | "unverified_no_reference" | "invalid"
    checks: dict[str, bool | str]      # qubits, operator ids, finite params, norm, particle number <N>, <Sz>, variational bound
    energy: dict              # energy, reference, reference_kind ("FCI"|"CCSD"|"HF"), error_mha, thresholds {1.6,5,10}, correlation_recovered
    resources: dict           # qubits, ops, params, gates_total, gates_1q, gates_2q, depth, cx_qiskit, compiled_depth, ionq_native_2q (or None + reason),
                              # measurement_groups, shots_for_target (estimate), executions
    cost: dict                # model forward passes, quantum_evals_to_propose, refine evals, pool grads, runtimes
    noise: dict | None        # ideal vs shots vs noisy energies with stderr and noise-model metadata, when requested
    weighted_score: float | None; weights: dict | None   # convenience only; always shown with the components
def grade(record, circuit, *, gen_log=None, target_mha=1.6, shots: int | None = None,
          noise: str | None = None, seed=0) -> GradeReport
def pareto_front(reports: dict[str, GradeReport], objectives=("error_mha", "cx_qiskit", "quantum_evals")) -> list[str]
    # Stage 1 eligibility first: only status == "valid_meets_target" are ranked on efficiency
def optimize(record, circuit, objective: str, *, target_mha=1.6, max_evals=500, seed=0) -> OptimizationResult
    # objectives: "accuracy" | "min_2q" | "min_depth" | "min_cost" | "noise" | "pareto"
    # OptimizationResult: original, optimized (GeneratedCircuit), before/after GradeReport, steps[], evals_used, same_hamiltonian_check
```

## Lane D: backends and execution (owns `genqai/backends.py`, `genqai/measure.py`, `genqai/execution.py`)

```python
@dataclass
class BackendInfo:
    name: str  # "statevector" | "local_shots" | "local_noisy" | "ionq_simulator" | "ionq_simulator_forte-1" | "ionq_simulator_aria-1" | "injected"
    kind: str; available: bool; reason: str; max_qubits: int | None; needs_key: bool; paid: bool  # paid backends are never listed as available
def list_backends() -> list[BackendInfo]
@dataclass
class ExecutionRecord:
    execution_id: str; backend: str; noise_model: str | None; shots: int; packed: bool; num_groups: int; width: int
    energy: float; stderr: float; error_mha: float | None; job_id: str | None; compiled_qasm: str | None
    group_details: list[dict]; runtime_s: float; software_versions: dict; status: str; error: str | None
def execute(record, circuit, backend: str | object, *, shots: int, seed: int | None = None,
            packed: bool | None = None, dry_run: bool = False) -> ExecutionRecord
    # Width check BEFORE any submission; never submits a job wider than the backend; refuses paid backends.
```

## Lane E: workbench UI, export, persistence, jobs (owns `app/`, `genqai/export.py`, `genqai/store.py`, `genqai/jobs.py`)

```python
# genqai/export.py
def qiskit_code(record, circuit) -> str        # standalone script; recreates the exact circuit (asserted by test: statevector equality)
def qasm(record, circuit, version: int = 3) -> str   # after decomposition to a basis; raises if not representable
def hamiltonian_json(record) -> str; def metadata_json(experiment: dict) -> str
def measurements_csv(execution: ExecutionRecord) -> str; def benchmark_csv(grades: dict[str, GradeReport]) -> str
# genqai/store.py: experiments under runs/experiments/<experiment_id>/ (gitignored): config.json, record.json, circuits.jsonl,
#   grades.json, executions.jsonl, exports/; experiment_id = config hash + method + timestamp; list/load/delete
# genqai/jobs.py: JobRunner(max_workers=1): submit(key, fn, *args) dedupes identical keys, status, progress messages, cancel
#   (cooperative flag + process-based hard cancel for chemistry), never runs two identical jobs concurrently
```
`app/workbench.py` is a Streamlit app (`streamlit run app/workbench.py`). It calls only the public APIs above and
labels every panel with its source (live computation, committed results file, or not available in this environment).
