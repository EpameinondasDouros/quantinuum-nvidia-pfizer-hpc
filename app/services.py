"""Thin adapter between the workbench UI and the GenQAI library.

Lane modules (molecule, registry/generate, grading/optimize, backends/execution) are imported lazily; a
missing one makes its panel say "not available in this build" instead of showing anything made up.
The only non-lane computations here call existing public functions (genqai.demo, genqai.evaluate,
genqai.circuits) and are labelled as the legacy preset path.
"""
from __future__ import annotations

import importlib
import importlib.util
import time
from pathlib import Path

from genqai.export import to_jsonable

ROOT = Path(__file__).resolve().parent.parent

LANES = {
    "molecule": ("genqai.molecule", "Lane A (molecule input, active space, capability report)"),
    "registry": ("genqai.registry", "Lane B (model registry and routing)"),
    "generate": ("genqai.generate", "Lane B (unified circuit generation)"),
    "grading": ("genqai.grading", "Lane C (grading)"),
    "optimize": ("genqai.optimize", "Lane C (circuit optimization)"),
    "backends": ("genqai.backends", "Lane D (backend catalogue)"),
    "execution": ("genqai.execution", "Lane D (execution)"),
}

PRESET_RANGES = {  # slider ranges (Angstrom); the training grids are genqai.data.geometry_grid
    "h2": {"r": (0.4, 3.0, 0.74)},
    "lih4": {"r": (0.9, 4.5, 1.6)},
    "lih6": {"r": (0.9, 4.5, 1.6)},
    "h4": {"a": (0.6, 2.0, 1.0), "b": (0.6, 2.4, 1.2)},
}

METHOD_LABELS = {
    "auto": "auto (registry recommendation)",
    "transformer_sft": "transformer (SFT)",
    "transformer_grpo": "transformer (GRPO)",
    "transformer_budget": "transformer (CX budget)",
    "krr": "kernel ridge angle regression",
    "nn_reuse_gauge": "nearest-neighbour reuse (gauge-fixed)",
    "angle_interp": "angle interpolation",
    "adapt": "ADAPT-VQE (conventional)",
    "uccsd_vqe": "UCCSD-VQE (conventional)",
}

ZERO_EVAL_STATEMENT = (
    "Quantum evaluations to *propose* this circuit: {q}. Model forward passes: {f}. "
    "Building the Hamiltonian (classical SCF and integrals) and grading the circuit (energy, resources, "
    "noise) are separate costs and are not part of that number; refinement energy evaluations ({r}) and "
    "ADAPT pool-gradient evaluations ({p}) are counted separately."
)


def _present(modname: str) -> bool:
    """True when the module file is part of THIS checkout's genqai package. (find_spec is not enough: an
    editable install of another checkout can resolve a missing genqai.<x> to that checkout's file.)"""
    import genqai

    return (Path(genqai.__file__).parent / f"{modname.rsplit('.', 1)[1]}.py").is_file()


def lane(name: str):
    """The lane module, or None when it is not part of this build."""
    modname = LANES[name][0]
    return importlib.import_module(modname) if _present(modname) else None


def lane_status() -> dict[str, tuple[bool, str]]:
    return {k: (_present(m), desc) for k, (m, desc) in LANES.items()}


def environment() -> dict[str, tuple[bool, str]]:
    """{capability: (available, detail)} for the sidebar."""
    out = {}
    for mod, label in (("rdkit", "RDKit (SMILES -> 3D conformers)"), ("torch", "PyTorch (transformer checkpoints)"),
                       ("py3Dmol", "py3Dmol (3D viewer)"), ("qiskit_aer", "Qiskit Aer (shot/noisy simulation)"),
                       ("qiskit_ionq", "qiskit-ionq (IonQ cloud simulators)")):
        out[mod] = (importlib.util.find_spec(mod) is not None, label)
    try:
        from pyscf import cc, gto, scf  # noqa: F401  (cc is missing from the browser build)

        out["pyscf"] = (True, "PySCF full (SCF, CCSD references)")
    except (ImportError, OSError) as exc:
        out["pyscf"] = (False, f"PySCF full build not usable: {type(exc).__name__}")
    try:
        from genqai.measure import ionq_available

        key = ionq_available()
    except Exception:
        key = False
    out["ionq_key"] = (key, "IonQ API key in environment/.env (key value never shown)")
    return out


# ---------------------------------------------------------------- molecule / Hamiltonian

def preset_atoms(system: str, params: tuple[float, ...]):
    from genqai.chemistry import SYSTEMS

    return SYSTEMS[system].atoms(tuple(params))


def xyz_text(atoms) -> str:
    return f"{len(atoms)}\n\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in atoms) + "\n"


def viewer_html(atoms, width: int = 520, height: int = 320) -> str | None:
    if importlib.util.find_spec("py3Dmol") is None:
        return None
    import py3Dmol

    v = py3Dmol.view(width=width, height=height)
    v.addModel(xyz_text(atoms), "xyz")
    v.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.28}})
    v.setBackgroundColor("#efedea")  # paper, as the rest of the plate
    v.zoomTo()
    v.rotate(-60, "x")  # oblique camera: presets lie along z, which the default view looks straight down
    v.rotate(30, "y")
    return v._make_html()


def build_preset_legacy(ctx, system: str, params: tuple[float, ...]):
    """Legacy preset path (no Lane A): genqai.demo.get_record -> (record, how)."""
    from genqai.demo import get_record

    ctx.progress(f"building {system} {params} (stored grid geometry or live PySCF)")
    return get_record(system, params)


def build_problem_job(ctx, config):
    mol = lane("molecule")
    ctx.progress("running SCF and building the qubit Hamiltonian")
    t0 = time.perf_counter()
    rec = mol.build_problem(config)
    ctx.progress(f"Hamiltonian built in {time.perf_counter() - t0:.1f} s")
    return rec, f"built with genqai.molecule.build_problem in {time.perf_counter() - t0:.1f} s"


# ---------------------------------------------------------------- generation

def legacy_methods(system: str) -> list[str]:
    from genqai.chemistry import SYSTEMS
    from genqai.demo import discover_checkpoints

    if system not in SYSTEMS:
        return []
    return [f"legacy:{stage}" for stage in discover_checkpoints(system)]


def generate_job(ctx, record, method: str, decision, opts: dict):
    """-> (GeneratedCircuit, log dict). Lane B when present, else the legacy one-shot checkpoint path."""
    if method.startswith("legacy:"):
        from genqai.demo import discover_checkpoints, generate_oneshot

        stage = method.split(":", 1)[1]
        ctx.progress(f"one-shot generation with checkpoints/{record.system}/{stage}")
        circ, info = generate_oneshot(record, discover_checkpoints(record.system)[stage])
        log = {"method": method, "model_id": info["checkpoint"], "forward_passes": 1, "candidates": 1,
               "model_time_s": info["runtime_s"], "quantum_evals_to_propose": 0, "refine_energy_evals": 0,
               "adapt_pool_gradient_evals": 0, "validity": circ.valid, "experimental": False,
               "notes": [info["mode"], "legacy preset path (genqai.demo.generate_oneshot)"]}
        return circ, log
    gen = lane("generate")
    ctx.progress(f"generating with {method}")
    circ, log = gen.generate(record, method, decision, **opts)
    return circ, to_jsonable(log)


# ---------------------------------------------------------------- grading / optimization

def grade_job(ctx, record, circuit, log: dict | None, target_mha: float, shots, noise):
    grading = lane("grading")
    if grading is None:  # basic exact scoring with the shared harness
        from genqai.evaluate import score_circuit

        ctx.progress("scoring with genqai.evaluate.score_circuit (exact statevector)")
        e = score_circuit(record, circuit, circuit.source or "circuit",
                          model_forward_passes=(log or {}).get("forward_passes", 0))
        return {"basic": True, "status": "valid" if circuit.valid else "invalid",
                "energy": {"energy": e.energy, "reference": record.reference_energy,
                           "reference_kind": record.reference_method,
                           "error_mha": 1000.0 * (e.energy - record.reference_energy)},
                "resources": {"qubits": record.num_qubits, "ops": e.num_operators, "cx_qiskit": e.cnot_count,
                              "depth": e.depth},
                "cost": {"quantum_evals_to_propose": (log or {}).get("quantum_evals_to_propose")}}
    ctx.progress("grading (energy, resources, checks)")
    kw = {"gen_log": log, "target_mha": target_mha}
    if shots:
        kw["shots"] = int(shots)
    if noise:
        kw["noise"] = noise
    return grading.grade(record, circuit, **kw)


def pareto(reports: dict) -> list[str] | None:
    grading = lane("grading")
    if grading is None:
        return None
    return grading.pareto_front(reports)


def optimize_job(ctx, record, circuit, objective: str, target_mha: float, max_evals: int):
    ctx.progress(f"optimizing for {objective}")
    return lane("optimize").optimize(record, circuit, objective, target_mha=target_mha, max_evals=max_evals)


# ---------------------------------------------------------------- execution

def execute_job(ctx, record, circuit, backend: str, shots: int, seed, packed, dry_run: bool):
    ctx.progress(("dry run (width check) on " if dry_run else "executing on ") + backend)
    return lane("execution").execute(record, circuit, backend, shots=shots, seed=seed, packed=packed, dry_run=dry_run)


# ---------------------------------------------------------------- committed results

def committed_figures() -> list[tuple[Path, str]]:
    from genqai.demo import FIGURE_CAPTIONS

    return [(p, FIGURE_CAPTIONS.get(p.stem, p.stem.replace("_", " "))) for p in sorted((ROOT / "figures").glob("*.png"))]


def committed_summary(system: str) -> list[dict]:
    """Per-method test_fresh statistics recomputed from the committed results_fresh/<system>/*.jsonl."""
    from genqai.evaluate import summarize
    from genqai.figures import load_results

    evals = [e for recs in load_results(ROOT / "results_fresh", system).values() for e in recs]
    rows = []
    for method, by_split in summarize(evals).items():
        s = by_split["all"]
        rows.append({"method": method, "n": s["n"], "median |err| mHa": s["median_abs_err_mha"],
                     "max |err| mHa": s["max_abs_err_mha"], "< 1.6 mHa": s["frac_under_1.6mha"],
                     "mean CX": s["mean_cnot_count"], "energy evals": s["mean_energy_evals"],
                     "pool-grad evals": s["mean_pool_gradient_evals"],
                     "forward passes": s["mean_model_forward_passes"]})
    return sorted(rows, key=lambda r: r["median |err| mHa"])
