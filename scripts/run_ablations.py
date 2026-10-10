"""X1 learning curve and D6 ablations, scored on the untouched test_fresh geometries (old grid test
splits alongside, for reference only). Every checkpoint / hyperparameter is chosen on VAL.

    PYTHONPATH=$PWD python -m scripts.run_ablations curve [--systems h4 lih6] [--jobs 3]
    PYTHONPATH=$PWD python -m scripts.run_ablations table [--systems h4 lih6] [--jobs 3]

curve: nested random TRAIN subsets N in {5, 10, 20, all} x 3 subset seeds (genqai.data.train_subset).
  For each subset the canonical labels are rebuilt from the subset's ADAPT runs alone
  (make_canonical_teacher --train-subset), then: sft_oneshot (SFT retrained with the committed
  checkpoint's config and seed; N=all is the committed checkpoint), angle_regression (KRR refit, VAL
  hyperparameters), nn_reuse_gauge, angle_interp (1-D systems). Teacher cost per subset = sum of ADAPT
  pool-gradient + energy evaluations over its geometries; the canonical relabelling's own energy
  evaluations are recorded separately. -> results/learning_curve.jsonl
table: SFT one-shot with Hamiltonian vs geometry conditioning, gauge on/off, canonical vs raw labels
  (each 3 training seeds, the VAL-best kept), KRR with the same feature/label variants, and the NN
  baselines. -> results/ablations.jsonl, results/ablations.md
Labels and checkpoints are written under runs/ (git-ignored).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from genqai.baselines import load_train_set, run_method
from genqai.data import FRESH_SPLIT, load_hamiltonians, load_teacher, train_subset
from genqai.evaluate import summarize
from genqai.features import fit_featurizer
from genqai.model import GQEPolicy
from genqai.schema import dump_jsonl
from scripts.generate import generate

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SIZES = (5, 10, 20, None)  # None = all TRAIN geometries
SUBSET_SEEDS = (0, 1, 2)
TRAIN_SEEDS = (0, 1, 2)
EVAL_SPLITS = (FRESH_SPLIT, "test")


def sft_argv(system: str, **override) -> list[str]:
    """train_sft argv: the committed checkpoint's own arguments with `override` applied."""
    args = {**GQEPolicy.load(ROOT / "checkpoints" / system / "sft.pt").extra["args"], **override}
    argv = []
    for k, v in args.items():
        if v is None or v is False:
            continue
        argv.append("--" + k.replace("_", "-"))
        if v is not True:
            argv.append(str(v))
    return argv


def run_jobs(jobs: list[tuple[Path, list[str]]], n_parallel: int) -> None:
    """Run `python -m scripts.<...>` argvs whose output file does not exist yet."""
    todo = [(out, argv) for out, argv in jobs if not out.exists()]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

    def one(job):
        out, argv = job
        log = out.with_suffix(".stdout")
        with log.open("w") as f:
            subprocess.run([sys.executable, "-m", *argv], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                           check=True)
        print(f"done {out.relative_to(ROOT)}", flush=True)

    for out, _ in todo:
        out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    with ThreadPoolExecutor(n_parallel) as ex:
        list(ex.map(one, todo))


def stats(evals) -> dict[str, dict]:
    """{split: metrics} for EvalRecords of one method on EVAL_SPLITS. Each split is summarised on its
    own: summarize's pseudo-split "test" (any test_*) would otherwise also swallow test_fresh."""
    keys = ("n", "valid_frac", "median_abs_err_mha", "mean_abs_err_mha", "max_abs_err_mha", "frac_under_1.6mha",
            "mean_cnot_count", "mean_num_operators")
    out = {}
    for sp in EVAL_SPLITS:
        s = summarize([e for e in evals if (e.split == FRESH_SPLIT) == (sp == FRESH_SPLIT)])[evals[0].method][sp]
        out[sp] = {k: s[k] for k in keys}
    return out


def eval_records(system: str):
    return load_hamiltonians(system, FRESH_SPLIT) + load_hamiltonians(system, "test")


def score_sft(ckpt: Path, records) -> dict[str, dict]:
    policy = GQEPolicy.load(ckpt)
    torch.manual_seed(0)
    return stats([generate(policy, r, "oneshot")[1] for r in records])


def score_baseline(method: str, train, records) -> dict[str, dict]:
    return stats([run_method(method, r, train) for r in records])


def val_best(ckpts: list[Path]) -> Path:
    """Checkpoint with the lowest VAL mean one-shot |error| (train_sft's own selection metric)."""
    return min(ckpts, key=lambda p: GQEPolicy.load(p).extra["best"]["val_mean_abs_mha"])


# ------------------------------------------------------------------------------ learning curve

def curve(systems, n_parallel):
    from scripts import make_canonical_teacher

    cells = [(s, n, seed) for s in systems for n in SIZES for seed in (SUBSET_SEEDS if n else (0,))]
    tag = lambda s, n, seed: f"{s}_n{n or 'all'}_s{seed}"  # noqa: E731
    for s, n, seed in cells:
        out = RUNS / "canon" / tag(s, n, seed)
        if not out.with_suffix(".jsonl").exists():
            make_canonical_teacher.main(["--system", s, "--out", str(out)]
                                        + (["--train-subset", str(n), "--subset-seed", str(seed)] if n else []))
    run_jobs([(RUNS / "curve" / f"{tag(s, n, seed)}.pt",
               ["scripts.train_sft", *sft_argv(s, train_subset=n, subset_seed=seed,
                                               targets=RUNS / "canon" / tag(s, n, seed),
                                               out=RUNS / "curve" / f"{tag(s, n, seed)}.pt",
                                               log=RUNS / "curve" / f"{tag(s, n, seed)}.json")])
              for s, n, seed in cells if n], n_parallel)

    rows = []
    for s, n, seed in cells:
        recs = eval_records(s)
        canon = str(RUNS / "canon" / tag(s, n, seed))
        train = load_train_set(s, n, seed, canon=canon)
        teacher, labels = load_teacher(s), load_teacher(s, canon)
        ids = [r.geometry_id for r in train.records]
        cost = {"n_train": len(ids), "teacher_pool_gradient_evals": sum(teacher[g].meta["pool_gradient_evals"] for g in ids),
                "teacher_energy_evals": sum(teacher[g].meta["energy_evals"] for g in ids),
                "relabel_energy_evals": sum(labels[g].meta["relabel_energy_evals"] for g in ids)}
        cost["teacher_evals"] = cost["teacher_pool_gradient_evals"] + cost["teacher_energy_evals"]
        ckpt = RUNS / "curve" / f"{tag(s, n, seed)}.pt" if n else ROOT / "checkpoints" / s / "sft.pt"
        methods = {"sft_oneshot": lambda: score_sft(ckpt, recs)}
        for m in ("angle_regression", "nn_reuse_gauge", "angle_interp"):
            if m != "angle_interp" or len(recs[0].geometry_params) == 1:
                methods[m] = lambda m=m: score_baseline(m, train, recs)
        for m, f in methods.items():
            for split, st in f().items():
                rows.append({"system": s, "method": m, "subset_seed": seed if n else None, "split": split,
                             **cost, **st})
            r = rows[-2]
            print(f"{s} N={len(ids):2d} seed {seed} {m:17s} fresh med {r['median_abs_err_mha']:.3g} "
                  f"<1.6 {100 * r['frac_under_1.6mha']:.0f}%", flush=True)
    dump_jsonl(rows, ROOT / "results" / "learning_curve.jsonl")


# ----------------------------------------------------------------------------------- ablations

SFT_VARIANTS = {  # name -> (conditioning, gauge, labels, train_sft overrides)
    "full": ("hamiltonian", "on", "canonical", {}),
    "geometry": ("geometry", "n/a", "canonical", {"cond": "geometry"}),
    "no_gauge": ("hamiltonian", "off", "canonical", {"no_gauge": True}),
    "raw_labels": ("hamiltonian", "on", "raw ADAPT", {"targets": "teacher"}),
}
KRR_VARIANTS = {  # name -> (conditioning, gauge, labels, load_train_set kwargs builder)
    "full": ("hamiltonian", "on", "canonical", lambda s: {}),
    "geometry": ("geometry", "n/a", "canonical", lambda s: {"featurizer": fit_featurizer(s, "geometry")}),
    "no_gauge": ("hamiltonian", "off", "canonical", lambda s: {"featurizer": fit_featurizer(s, gauge=False)}),
    "raw_labels": ("hamiltonian", "on", "raw ADAPT", lambda s: {"canon": "teacher"}),
}
NN_VARIANTS = {  # baselines.METHODS name -> (conditioning, gauge, labels)
    "nn_reuse": ("geometry NN", "off", "raw ADAPT"),
    "nn_reuse_gauge": ("geometry NN", "on", "raw ADAPT"),
    "nn_reuse_canon": ("geometry NN", "on", "canonical"),
}


def table(systems, n_parallel):
    ck = lambda s, v, seed: RUNS / "ablation" / f"{s}_{v}_s{seed}.pt"  # noqa: E731
    run_jobs([(ck(s, v, seed), ["scripts.train_sft", *sft_argv(s, **ov, seed=seed, out=ck(s, v, seed),
                                                                 log=ck(s, v, seed).with_suffix(".json"))])
              for s in systems for v, (*_, ov) in SFT_VARIANTS.items() for seed in TRAIN_SEEDS], n_parallel)
    rows = []

    def add(s, model, variant, cond, gauge, labels, st, **extra):
        for split, r in st.items():
            rows.append({"system": s, "model": model, "variant": variant, "conditioning": cond, "gauge": gauge,
                         "labels": labels, "split": split, **extra, **r})
        r = st[FRESH_SPLIT]
        print(f"{s} {model:12s} {variant:11s} fresh med {r['median_abs_err_mha']:.3g} max {r['max_abs_err_mha']:.3g} "
              f"<1.6 {100 * r['frac_under_1.6mha']:.0f}%", flush=True)

    for s in systems:
        recs = eval_records(s)
        add(s, "transformer", "committed", "hamiltonian", "on", "canonical",
            score_sft(ROOT / "checkpoints" / s / "sft.pt", recs), train_seed=GQEPolicy.load(
                ROOT / "checkpoints" / s / "sft.pt").extra["args"]["seed"])
        for v, (cond, gauge, labels, _) in SFT_VARIANTS.items():
            best = val_best([ck(s, v, seed) for seed in TRAIN_SEEDS])
            seeds = {seed: GQEPolicy.load(ck(s, v, seed)).extra["best"]["val_mean_abs_mha"] for seed in TRAIN_SEEDS}
            add(s, "transformer", v, cond, gauge, labels, score_sft(best, recs),
                train_seed=int(best.stem.rsplit("_s", 1)[1]), val_mean_abs_mha_by_seed=seeds)
        for v, (cond, gauge, labels, kw) in KRR_VARIANTS.items():
            add(s, "kernel ridge", v, cond, gauge, labels, score_baseline("angle_regression", load_train_set(s, **kw(s)), recs))
        train = load_train_set(s)
        for m, (cond, gauge, labels) in NN_VARIANTS.items():
            add(s, "nearest nbr", m, cond, gauge, labels, score_baseline(m, train, recs))
    dump_jsonl(rows, ROOT / "results" / "ablations.jsonl")
    (ROOT / "results" / "ablations.md").write_text(ablation_markdown(rows))


def _f(x: float) -> str:
    return "-" if x != x else (f"{x:.2g}" if x < 0.01 else f"{x:.3g}")


def ablation_markdown(rows) -> str:
    lines = ["# D6 ablations (one-shot, zero energy evaluations at inference)", "",
             "Generated by `python -m scripts.run_ablations table`. |err| = |E − E_FCI| (mHa) of the one-shot "
             f"circuit. Primary: the untouched `{FRESH_SPLIT}` geometries; the last two columns are the original "
             "grid test splits (seen while the label recipe was tuned), for reference only. Transformer variants: "
             "same SFT config as the committed checkpoint, 3 training seeds, VAL-best kept. Kernel ridge: VAL-chosen "
             "hyperparameters. Gauge = orbital-phase sign fixing (genqai.features); n/a for geometry inputs, which "
             "carry no MO-sign information.", ""]
    for s in dict.fromkeys(r["system"] for r in rows):
        fresh = [r for r in rows if r["system"] == s and r["split"] == FRESH_SPLIT]
        old = {(r["model"], r["variant"]): r for r in rows if r["system"] == s and r["split"] == "test"}
        lines += [f"## {s} ({fresh[0]['n']} test_fresh / {next(iter(old.values()))['n']} old-test geometries)", "",
                  "| model | variant | conditioning | gauge | labels | median | max | < 1.6 mHa | valid | mean CX "
                  "| old test median | old test < 1.6 |", "|---" * 12 + "|"]
        for r in fresh:
            o = old[(r["model"], r["variant"])]
            lines.append(f"| {r['model']} | {r['variant']} | {r['conditioning']} | {r['gauge']} | {r['labels']} "
                         f"| {_f(r['median_abs_err_mha'])} | {_f(r['max_abs_err_mha'])} "
                         f"| {100 * r['frac_under_1.6mha']:.0f}% | {100 * r['valid_frac']:.0f}% "
                         f"| {r['mean_cnot_count']:.0f} | {_f(o['median_abs_err_mha'])} "
                         f"| {100 * o['frac_under_1.6mha']:.0f}% |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["curve", "table"])
    ap.add_argument("--systems", nargs="+", default=["h4", "lih6"])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    (curve if a.what == "curve" else table)(a.systems, a.jobs)
