"""Fresh ADAPT-VQE teacher labels for every geometry: data/<system>/teacher.jsonl.

--fresh: ADAPT on the test_fresh geometries -> data/<system>/teacher_fresh.jsonl. These are only the
"adapt" baseline's own runs on those geometries; nothing ever trains on them.
"""
import argparse

import numpy as np

from genqai.adapt import run_adapt
from genqai.data import DATA_DIR, FRESH_SPLIT, load_hamiltonians
from genqai.schema import dump_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["h2", "lih4", "lih6", "h4"])
ap.add_argument("--max-ops", type=int, default=32)
ap.add_argument("--grad-tol", type=float, default=1e-4)
ap.add_argument("--fresh", action="store_true")
args = ap.parse_args()
for s in args.systems:
    recs = load_hamiltonians(s, FRESH_SPLIT if args.fresh else None)
    labels = [run_adapt(r, max_ops=args.max_ops, grad_tol=args.grad_tol) for r in recs]
    dump_jsonl(labels, DATA_DIR / s / ("teacher_fresh.jsonl" if args.fresh else "teacher.jsonl"))
    err = np.array([l.meta["error_mha"] for l in labels])
    n = np.array([len(l.operators) for l in labels])
    conv = sum(l.meta["converged"] for l in labels)
    print(f"{s}: n={len(labels)} converged={conv} ops[min/med/max]={n.min()}/{int(np.median(n))}/{n.max()} "
          f"err_mHa[med/max]={np.median(err):.4f}/{err.max():.4f} chem_acc={np.mean(np.abs(err)<=1.6):.2f} "
          f"t={sum(l.meta['runtime_s'] for l in labels):.1f}s")
