"""Markdown tables per system/split from results/<system>/*.jsonl."""
import argparse
from pathlib import Path

from genqai.baselines import METHODS
from genqai.evaluate import summarize
from genqai.schema import EvalRecord, load_jsonl

RESULTS = Path(__file__).resolve().parent.parent / "results"
SPLITS = ["train", "val", "test_interp", "test_interval", "test_extrap", "test"]

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["h2", "lih4", "lih6", "h4"])
ap.add_argument("--splits", nargs="*", default=SPLITS)
args = ap.parse_args()


def fmt(x, p=2):
    return "-" if x != x else f"{x:.{p}f}"


def order(path):  # known baselines first, any other method (model, RL) after
    return (METHODS.index(path.stem), "") if path.stem in METHODS else (len(METHODS), path.stem)


for s in args.systems:
    files = sorted((RESULTS / s).glob("*.jsonl"), key=order)
    summ = summarize([e for p in files for e in load_jsonl(p, EvalRecord)])
    print(f"\n## {s}\n")
    print("| split | method | n | med abs err mHa | max abs err mHa | <1.6 | <5 | ops | CX | depth "
          "| pool-grad evals | energy evals | opt iters | runtime s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for split in args.splits:
        for m, by in summ.items():
            if split not in by:
                continue
            r = by[split]
            print(f"| {split} | {m} | {r['n']} | {fmt(r['median_abs_err_mha'], 3)} "
                  f"| {fmt(r['max_abs_err_mha'], 3)} | {r['frac_under_1.6mha']:.2f} | {r['frac_under_5mha']:.2f} "
                  f"| {r['mean_num_operators']:.1f} | {fmt(r['mean_cnot_count'], 1)} | {fmt(r['mean_depth'], 1)} "
                  f"| {r['mean_pool_gradient_evals']:.0f} | {r['mean_energy_evals']:.0f} "
                  f"| {r['mean_optimizer_iterations']:.0f} | {r['mean_runtime_s']:.3f} |")
