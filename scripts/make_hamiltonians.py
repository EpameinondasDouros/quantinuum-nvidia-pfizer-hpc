"""Build data/<system>/hamiltonians.jsonl for the requested systems."""
import argparse
from collections import Counter

from genqai.data import build_dataset, hamiltonians_path
from genqai.schema import dump_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["h2", "lih4", "lih6", "h4"])
args = ap.parse_args()
for s in args.systems:
    recs = build_dataset(s)
    dump_jsonl(recs, hamiltonians_path(s))
    print(s, len(recs), "qubits", recs[0].num_qubits, "terms", len(recs[0].pauli_terms), dict(Counter(r.split for r in recs)))
