"""All baselines for every geometry of every system: results/<system>/<method>.jsonl.

--fresh: score on the untouched test_fresh geometries instead -> results_fresh/<system>/<method>.jsonl.
"""
import argparse
import time

from genqai.baselines import METHODS, run_baselines
from genqai.data import results_dir
from genqai.schema import dump_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["h2", "lih4", "lih6", "h4"])
ap.add_argument("--methods", nargs="*", default=list(METHODS))
ap.add_argument("--fresh", action="store_true")
args = ap.parse_args()
for s in args.systems:
    t0 = time.perf_counter()
    for m, evals in run_baselines(s, args.methods, fresh=args.fresh).items():
        dump_jsonl(evals, results_dir(args.fresh) / s / f"{m}.jsonl")
    print(f"{s}: {len(args.methods)} methods in {time.perf_counter() - t0:.1f}s")
