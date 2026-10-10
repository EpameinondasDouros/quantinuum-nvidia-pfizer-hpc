"""GenQAI on Qollab: a Hamiltonian-conditioned transformer writes a complete VQE circuit in ONE shot.

Run form fields: system (h2 | lih4 | lih6 | h4), bond length r in Angstrom (h4: a, b), shots per
measurement group, live_adapt (also run a fresh ADAPT-VQE for comparison). Qollab calls main();
the quantum backend is the global `backend` chosen in the Select-QPU dialog.
"""
import os
import sys
import time
import warnings

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, BASE)
warnings.filterwarnings("ignore")

from datafiles import DATA_FILES  # noqa: E402  (literal data paths: Qollab only ships files named in source)

CHEM_ACC_MHA = 1.6


def _step(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    return time.perf_counter()


def _done(t0, timings, name):
    timings[name] = time.perf_counter() - t0
    print(f"[{name}: {timings[name]:.2f} s]")


def _context(system):
    import json

    from genqai.data import DATA_DIR

    with open(DATA_DIR / "results_summary.json") as f:
        s = json.load(f)[system]["test_fresh"]
    names = {"sft_oneshot": "transformer one-shot", "angle_regression": "angle regression (KRR)",
             "nn_reuse_gauge": "nearest-neighbour reuse", "adapt": "fresh ADAPT-VQE", "ccsd": "CCSD", "hf": "HF"}
    print(f"Precomputed held-out results ({system}, {next(iter(s.values()))['n']} fresh random test geometries):")
    print(f"  {'method':<26}{'median |err| mHa':>17}{'chem. acc.':>11}{'quantum evals/geom':>20}")
    for m, st in s.items():
        evals = st["mean_pool_gradient_evals"] + st["mean_energy_evals"]
        print(f"  {names.get(m, m):<26}{st['median_abs_err_mha']:>17.3f}{100 * st['frac_under_1.6mha']:>10.0f}%{evals:>20.0f}")


def _plot(rec, evals, measured):
    import matplotlib

    if "pyodide" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(e.method, e.error_mha, 0.0) for e in evals if e.method != "fci" and e.error_mha == e.error_mha]
    if measured is not None:
        rows.append((f"measured on {measured.backend}", 1000 * (measured.energy - rec.fci_energy), 1000 * measured.stderr))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.6))
    y = range(len(rows))
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="o", color="#3b5bdb", capsize=4)
    ax.axvspan(-CHEM_ACC_MHA, CHEM_ACC_MHA, color="#2b8a3e", alpha=0.12, label="chemical accuracy (1.6 mHa)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("E - E_FCI (mHa, symlog)")
    ax.set_title(f"GenQAI {rec.system} {rec.geometry_params}: FCI = {rec.fci_energy:.6f} Ha")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("genqai_energies.png", dpi=120)
    print("figure saved: genqai_energies.png")


def main(system: str = "h2", r: float = 1.33, a: float = 1.1, b: float = 1.7, shots: int = 2000, live_adapt: bool = True):
    from genqai.circuits import build_circuit, gate_stats
    from genqai.demo import compare, comparison_table, describe_circuit, generate_all, get_record, training_context
    from genqai.pool import get_pool

    timings, rec, circ, gens, evals, measured = {}, None, None, {}, [], None
    system = system.strip().lower()
    if system not in ("h2", "lih4", "lih6", "h4"):
        print(f"unknown system {system!r}: choose h2, lih4, lih6 or h4")
        return
    params = (a, b) if system == "h4" else (r,)
    missing = [p for p in DATA_FILES if not os.path.exists(os.path.join(BASE, p))]
    if missing:
        print(f"WARNING: {len(missing)} bundled data files not found under {BASE}: {missing[:4]} ...")
    print("GenQAI: Hamiltonian-conditioned transformer -> complete operator+angle circuit in one shot")
    print("(reproduction of the Quantinuum/NVIDIA/Pfizer ADAPT-GQE idea; NumPy inference, no PyTorch)")
    if system == "h4":
        print("note: h4 is 8 qubits; PySCF in the browser and the live ADAPT run are slower (~minutes).")

    t0 = _step(f"1. Hamiltonian for {system} at {params} (PySCF, STO-3G active space)")
    try:
        rec, how = get_record(system, params)
        print(f"source: {how}")
        print(f"{rec.num_qubits} qubits, {len(rec.pauli_terms)} Pauli terms, {rec.mapping} mapping")
        print(f"HF   = {rec.hf_energy:.6f} Ha\nCCSD = {rec.ccsd_energy:.6f} Ha\nFCI  = {rec.fci_energy:.6f} Ha  (exact reference)")
        ctx = training_context(rec)
        rng = ", ".join(f"{k} in [{lo:g}, {hi:g}]" for k, (lo, hi) in ctx["range"].items())
        print(f"training range: {rng} -> this geometry is {'INSIDE' if ctx['inside_range'] else 'OUTSIDE (extrapolation)'}")
        print(f"nearest training geometry: {ctx['nearest']} (distance {ctx['distance']:.3f} A)")
    except Exception as exc:
        print(f"FAILED to obtain a Hamiltonian: {type(exc).__name__}: {exc}")
        return
    _done(t0, timings, "hamiltonian")

    t0 = _step("2. One-shot circuit generation (transformer, greedy decoding, 0 energy evaluations)")
    try:
        gens = generate_all(rec)
        circ, info = next(iter(gens.values()))
        st = gate_stats(build_circuit(rec, circ.operators, circ.angles))
        print(f"{info['mode']} from {info['checkpoint']}: model time {1000 * info['runtime_s']:.0f} ms")
        print(f"{len(circ.operators)} operators chosen from a pool of {len(get_pool(system))}; "
              f"{st['cnot_count']} CX, depth {st['depth']} (Qiskit CX basis, all-to-all)")
        print(describe_circuit(circ))
    except Exception as exc:
        print(f"generation failed: {type(exc).__name__}: {exc}")
    _done(t0, timings, "generation")

    t0 = _step("3. Exact scoring vs FCI and baselines (statevector; baselines read TRAINING labels only)")
    try:
        evals = compare(rec, gens, live_adapt=live_adapt)
        print(comparison_table(rec, evals))
        print("sft_oneshot = the transformer; nn_reuse_gauge = copy nearest training circuit; angle_regression ="
              " kernel ridge on the same features; adapt = ADAPT-VQE from scratch (pool grads + energy evals"
              " are the quantum cost the transformer avoids).")
    except Exception as exc:
        print(f"scoring failed: {type(exc).__name__}: {exc}")
    _done(t0, timings, "scoring")
    try:
        _context(system)
    except Exception as exc:
        print(f"(no precomputed context: {exc})")

    t0 = _step("4. Quantum execution of the generated circuit on the selected backend")
    qpu = globals().get("backend")
    if qpu is None or circ is None:
        print("no backend injected (or no circuit): skipping. On Qollab pick a backend in Select QPU.")
    else:
        try:
            from genqai.measure import MeasurementPlan, measure_on_backend

            plan = MeasurementPlan.from_record(rec)
            ionq = "ionq" in (type(qpu).__module__ + str(getattr(qpu, "name", ""))).lower()
            print(f"backend: {qpu}")
            print(f"{plan.num_groups} qubit-wise-commuting measurement groups x {shots} shots, "
                  f"ALL submitted as ONE job (one backend.run call)")
            if ionq:
                print("IonQ: circuits are compiled server-side to IonQ native gates (GPI/GPI2 + MS/ZZ).")
            measured = measure_on_backend(rec, circ.operators, circ.angles, qpu, shots, plan=plan)
            err = 1000 * (measured.energy - rec.fci_energy)
            print(f"backend name: {measured.backend}   job id: {measured.meta.get('job_id', '-')}")
            print(f"measured E = {measured.energy:.6f} +- {measured.stderr:.6f} Ha  ({measured.total_shots} shots)")
            exact = next((e.energy for e in evals if e.method == circ.source), None)
            print(f"error vs FCI = {err:+.2f} +- {1000 * measured.stderr:.2f} mHa"
                  + (f"\n(exact energy of this circuit: {exact:.6f} Ha; the difference is shot noise"
                     " plus the backend's noise model, if any)" if exact is not None else ""))
        except Exception as exc:
            import traceback

            print(f"quantum execution failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    _done(t0, timings, "quantum")

    try:
        _plot(rec, evals, measured)
    except Exception as exc:
        print(f"figure skipped: {type(exc).__name__}: {exc}")
    print("\ntimings: " + ", ".join(f"{k} {v:.1f} s" for k, v in timings.items()))
