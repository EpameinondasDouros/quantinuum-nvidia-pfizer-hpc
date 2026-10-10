"""Measured energy of one circuit on exact / local / IonQ backends -> results/<system>/ionq_demo_<id>.json.

  python scripts/run_ionq_demo.py --system h2 --geometry-index 0 --shots 4000
  python scripts/run_ionq_demo.py --system lih4 --circuits runs/gen.jsonl --ionq-noise-models aria-1 forte-1

--geometry-index indexes the TEST geometries (splits test_*) in dataset order. --shots is the total
budget over all measurement groups. IonQ backends run only when IONQ_API_KEY is set (env or .env).
"""
import argparse
import json
import warnings
from pathlib import Path

from genqai.circuits import build_circuit, gate_stats
from genqai.data import DATA_DIR, load_hamiltonians
from genqai.measure import MeasurementPlan, compile_state, ionq_native_stats, run_backends, runs_table
from genqai.schema import GeneratedCircuit, load_jsonl

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="h2", choices=["h2", "lih4", "lih6", "h4"])
ap.add_argument("--geometry-index", type=int, default=0)
ap.add_argument("--circuits", help="jsonl of GeneratedCircuit records (default: ADAPT teacher)")
ap.add_argument("--backend", nargs="+", default=None,
                help="default: exact exact_probs local_shots local_noisy (+ ionq_ideal ionq_noisy if key)")
ap.add_argument("--shots", type=int, default=20_000, help="total shots over all groups")
ap.add_argument("--allocation", default="uniform", choices=["uniform", "weighted"])
ap.add_argument("--ionq-noise-models", nargs="+", default=["aria-1"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="output suffix: results/<system>/ionq_demo_<id><tag>.json")
args = ap.parse_args()

rec = load_hamiltonians(args.system, "test")[args.geometry_index]
src = Path(args.circuits) if args.circuits else DATA_DIR / args.system / "teacher.jsonl"
circ = {c.geometry_id: c for c in load_jsonl(src, GeneratedCircuit)}[rec.geometry_id]
plan = MeasurementPlan.from_record(rec)
qc = build_circuit(rec, circ.operators, circ.angles)
gates = gate_stats(qc)
native = {d: ionq_native_stats(compile_state(qc), d) for d in ("aria-1", "forte-1")}

runs = run_backends(rec, circ.operators, circ.angles, args.backend, shots=args.shots, allocation=args.allocation,
                   ionq_noise_models=args.ionq_noise_models, seed=args.seed, plan=plan)
print(f"\n{args.system} {rec.geometry_id} {rec.geometry_params} split={rec.split} source={circ.source} "
      f"ops={len(circ.operators)} groups={plan.num_groups}")
print(f"FCI {rec.fci_energy:.6f} Ha | CX (Qiskit basis) {gates['cnot_count']} | native 2q (qiskit-ionq local): "
      + ", ".join(f"{d} {v['native_2q_count']} {v['native_2q_gate'].upper()}" for d, v in native.items()))
print(runs_table(rec, runs))

out = ROOT / "results" / args.system / f"ionq_demo_{rec.geometry_id}{args.tag}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "geometry_id": rec.geometry_id, "system": args.system, "geometry_params": rec.geometry_params,
    "split": rec.split, "fci_energy": rec.fci_energy, "circuit": {"source": circ.source, "file": str(src),
    "operators": circ.operators, "angles": circ.angles}, "num_groups": plan.num_groups,
    "qiskit_gate_stats": gates, "ionq_native_stats": native, "shots_requested": args.shots,
    "runs": [{**e.to_dict(), "error_mha": 1000 * (e.energy - rec.fci_energy)} for e in runs],
}, indent=1))
print(f"saved {out.relative_to(ROOT)}")
