"""All baselines for every geometry of every system: results/<system>/<method>.jsonl."""
import argparse
import time
from pathlib import Path

from genqai.baselines import METHODS, run_baselines
from genqai.schema import dump_jsonl

RESULTS = Path(__file__).resolve().parent.parent / "results"

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["h2", "lih4", "lih6", "h4"])
ap.add_argument("--methods", nargs="*", default=list(METHODS))
args = ap.parse_args()
for s in args.systems:
    t0 = time.perf_counter()
    for m, evals in run_baselines(s, args.methods).items():
        dump_jsonl(evals, RESULTS / s / f"{m}.jsonl")
    print(f"{s}: {len(args.methods)} methods in {time.perf_counter() - t0:.1f}s")
