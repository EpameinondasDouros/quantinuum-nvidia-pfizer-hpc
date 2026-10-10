"""Fair comparison tables (B4): every method on the original test splits (results/) and, separately,
on the untouched test_fresh geometries (results_fresh/), scored once with frozen checkpoints.

    PYTHONPATH=$PWD python -m scripts.report_fair [systems...]
"""
import argparse

from genqai.data import FRESH_SPLIT, fresh_regime, load_hamiltonians, results_dir
from genqai.evaluate import summarize
from genqai.figures import load_results, method_order

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="*", default=["h2", "lih4", "lih6", "h4"])
args = ap.parse_args()


def fmt(x, p=3):
    return "-" if x != x else f"{x:.{p}f}"


def table(evals_by_method, split, extrap_ids=None):
    summ = summarize([e for v in evals_by_method.values() for e in v])
    head = ("| method | n | median abs err | max abs err | <1.6 | <5 | <10 | mean CX | energy evals | pool-grad evals |"
            + (" extrap median / max |" if extrap_ids else ""))
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    for m in method_order(summ):
        if m == "fci" or split not in summ[m]:
            continue
        r = summ[m][split]
        row = (f"| {m} | {r['n']} | {fmt(r['median_abs_err_mha'])} | {fmt(r['max_abs_err_mha'])} "
               f"| {100 * r['frac_under_1.6mha']:.0f}% | {100 * r['frac_under_5mha']:.0f}% "
               f"| {100 * r['frac_under_10mha']:.0f}% | {fmt(r['mean_cnot_count'], 0)} "
               f"| {r['mean_energy_evals']:.1f} | {r['mean_pool_gradient_evals']:.1f} |")
        if extrap_ids:
            x = summarize([e for e in evals_by_method[m] if e.geometry_id in extrap_ids])[m][split]
            row += f" {fmt(x['median_abs_err_mha'])} / {fmt(x['max_abs_err_mha'])} |"
        print(row)


for s in args.systems:
    print(f"\n## {s}\n\n### test (original grid splits; seen during label/training tuning)\n")
    table(load_results(results_dir(), s), "test")
    fresh = load_results(results_dir(fresh=True), s)
    if not fresh:
        continue
    extrap = {h.geometry_id for h in load_hamiltonians(s, FRESH_SPLIT) if fresh_regime(s, h.geometry_params) == "extrap"}
    print(f"\n### {FRESH_SPLIT} (untouched; {len(extrap)} of "
          f"{len(load_hamiltonians(s, FRESH_SPLIT))} geometries beyond the grid edge)\n")
    table(fresh, FRESH_SPLIT, extrap)
