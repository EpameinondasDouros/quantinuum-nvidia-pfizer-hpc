"""Terminal version of notebooks/genqai_demo.ipynb: one unseen geometry, end to end.

    PYTHONPATH=$PWD python scripts/demo.py --system h2 --geometry 1.33
    PYTHONPATH=$PWD python scripts/demo.py --system lih4 --geometry 2.47 --shots 20000
    PYTHONPATH=$PWD python scripts/demo.py --system h4 --geometry 1.05 1.37 --no-adapt --no-measure

Builds the Hamiltonian live (PySCF), generates a circuit with every checkpoints/<system>/*.pt in one
shot, scores it exactly against HF / CCSD / FCI / nearest-neighbour / fresh ADAPT, then measures the
generated circuit with grouped Pauli measurements on local simulators and, when IONQ_API_KEY is set
and --ionq is given, on the IonQ cloud simulator (ideal + noise models).
"""
import argparse
import warnings
from pathlib import Path

from genqai.circuits import build_circuit, gate_stats
from genqai.demo import FIGURE_CAPTIONS, ROOT, compare, comparison_table, describe_circuit, generate_all, get_record
from genqai.measure import ionq_available, run_backends, runs_table

warnings.filterwarnings("ignore", category=DeprecationWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="h2", choices=["h2", "lih4", "lih6", "h4"])
ap.add_argument("--geometry", nargs="+", type=float, default=[1.33], help="r in Å (h4: a b)")
ap.add_argument("--no-adapt", action="store_true", help="skip the live ADAPT-VQE comparison run")
ap.add_argument("--no-measure", action="store_true", help="skip shot-based measurement")
ap.add_argument("--ionq", action="store_true", help="also run on the IonQ cloud simulator (needs IONQ_API_KEY)")
ap.add_argument("--shots", type=int, default=4000, help="total shots over all measurement groups, per backend")
ap.add_argument("--ionq-noise-models", nargs="+", default=["aria-1", "forte-1"])
a = ap.parse_args()

rec, how = get_record(a.system, a.geometry)
print(f"== {a.system} {rec.geometry_params}: {how}")
print(f"   {rec.num_qubits} qubits, {len(rec.pauli_terms)} Pauli terms | HF {rec.hf_energy:.6f}  "
      f"CCSD {rec.ccsd_energy:.6f}  FCI {rec.fci_energy:.6f} Ha")

gens = generate_all(rec)
for name, (circ, info) in gens.items():
    st = gate_stats(build_circuit(rec, circ.operators, circ.angles))
    print(f"\n== {name}: {info['mode']} from {info['checkpoint']} ({1000 * info['runtime_s']:.0f} ms, "
          f"{len(circ.operators)} ops, {st['cnot_count']} CX, depth {st['depth']})")
    print(describe_circuit(circ))

print("\n== Exact scoring (statevector)")
print(comparison_table(rec, compare(rec, gens, live_adapt=not a.no_adapt)))

if not a.no_measure:
    circ = next(iter(gens.values()))[0]
    backends = ["local_shots", "local_noisy"]
    if a.ionq:
        if ionq_available():
            backends += ["ionq_ideal", "ionq_noisy"]
        else:
            print("\n--ionq given but no IONQ_API_KEY found (env or .env): local backends only")
    print(f"\n== Measured energy of {circ.source} ({a.shots} shots per backend)")
    print(runs_table(rec, run_backends(rec, circ.operators, circ.angles, backends, shots=a.shots,
                                       ionq_noise_models=a.ionq_noise_models)))

print("\n== Committed figures")
for name, cap in FIGURE_CAPTIONS.items():
    p = ROOT / "figures" / f"{name}.png"
    if p.exists():
        print(f"{Path(p).relative_to(ROOT)}: {cap}")
