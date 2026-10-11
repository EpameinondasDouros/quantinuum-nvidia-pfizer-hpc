"""Does hardware-aware optimisation (prune + refine, genqai.optimize) beat the non-neural accuracy-vs-CX
frontier? (Lane C research question.)

Starting circuits, per held-out geometry:
  sft     the committed SFT transformer's one-shot circuit (results[_fresh]/<s>/gen_oneshot.jsonl)
  krr     kernel-ridge angle regression on the full canonical sequence (genqai.baselines TrainSet, refit here;
          TRAIN labels + VAL Hamiltonians only)
  adapt   the fresh ADAPT teacher run (data/<s>/teacher.jsonl, teacher_fresh.jsonl for test_fresh)
Each is optimised with optimize(objective="min_2q", target 1.6 mHa) in two modes:
  oracle  removals accepted while |E - E_FCI| <= target (uses the exact reference, which the workbench has
          for these small active spaces)
  blind   removals accepted while E - E_start <= tol (no reference used); tol chosen on VAL
Hyperparameters (selection rule, beam, tol) are chosen per (start, mode) on VAL only: the configuration
with the lowest mean final CX among those that keep every VAL geometry within the target (else the one
with the most VAL geometries within target). TEST (grid) and test_fresh are then scored once.

Comparators (zero evaluations unless noted) come from results/<s>/budget_frontier.jsonl
(scripts/budget_frontier.py): KRR on fixed prefixes / pruned labels / ADAPT prefixes at their VAL-chosen
level, plus KRR-prefix and truncated ADAPT at the SAME per-geometry CX as each optimised circuit.
Prune + refine spends energy evaluations: a different cost class from one-shot generation.

Rows -> results/<s>/hw_opt.jsonl (VAL tuning + grid test) and results_fresh/<s>/hw_opt.jsonl; summary table
-> results/hw_opt.md.  CX = Qiskit CX-basis count at transpile optimisation level 1 (the frontier's
convention); level-3 counts are recorded too.

    PYTHONPATH=$PWD python -m scripts.run_hw_optimization --systems h4 lih6
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from genqai.baselines import load_train_set
from genqai.data import FRESH_SPLIT, load_hamiltonians, load_teacher, results_dir
from genqai.optimize import optimize
from genqai.schema import GeneratedCircuit, dump_jsonl, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
TARGET = 1.6
STARTS = ("sft", "krr", "adapt")
MAX_EVALS = 20_000
CONFIGS = {
    "oracle": [{"selection": s, "beam": b} for s, b in itertools.product(("energy", "energy_per_cx"), (3, 6))],
    "blind": [{"selection": s, "beam": b, "reference_free_tol_mha": t}
              for s, b, t in itertools.product(("energy", "energy_per_cx"), (3, 6), (0.25, 0.5, 1.0, 1.5))],
}
FRONTIER_CURVES = ("regress_prefix", "regress_pruned", "regress_adapt", "budget", "adapt")
GROUPS = {"TEST (grid)": lambda s: s.startswith("test_") and s != FRESH_SPLIT, "TEST_FRESH": lambda s: s == FRESH_SPLIT}


def start_circuits(system: str, records) -> dict[str, dict[str, GeneratedCircuit]]:
    """{start: {geometry_id: circuit}} with each start's cost to PROPOSE the circuit in meta['propose_evals']."""
    sft = {}
    for fresh in (False, True):
        for c in load_jsonl(results_dir(fresh) / system / "gen_oneshot.jsonl", GeneratedCircuit):
            c.meta["propose_evals"] = 0
            sft[c.geometry_id] = c
    reg = load_train_set(system).regressor
    krr = {}
    for r in records:
        c = reg.predict(r)
        c.meta["propose_evals"] = 0
        krr[r.geometry_id] = c
    adapt = {**load_teacher(system), **load_teacher(system, "teacher_fresh")}
    for c in adapt.values():
        c.meta["propose_evals"] = c.meta["energy_evals"] + c.meta["pool_gradient_evals"]
    ids = {r.geometry_id for r in records}
    return {k: {g: c for g, c in v.items() if g in ids} for k, v in (("sft", sft), ("krr", krr), ("adapt", adapt))}


def run_one(args) -> dict:
    record, circuit, start, mode, cfg, phase = args
    warnings.filterwarnings("ignore")
    res = optimize(record, circuit, "min_2q", target_mha=TARGET, max_evals=MAX_EVALS,
                   grade_kw={"ionq_native": False}, **cfg)
    b, a = res.before, res.after
    return {"phase": phase, "start": start, "mode": mode, "config": cfg, "geometry_id": record.geometry_id,
            "split": record.split, "start_status": b.status, "start_err_mha": b.energy["error_mha"],
            "start_cx": b.resources["cx_qiskit"], "start_ops": len(circuit.operators),
            "status": a.status, "err_mha": a.energy["error_mha"], "cx": a.resources["cx_qiskit"],
            "cx_l3": res.compiled_level3["optimized"]["cx_qiskit"], "compiled_depth": a.resources["compiled_depth"],
            "n_ops": len(res.optimized.operators), "evals_used": res.evals_used, "gradient_evals": res.gradient_evals,
            "propose_evals": circuit.meta["propose_evals"], "same_hamiltonian_ok": res.same_hamiltonian_check["ok"],
            "operators": res.optimized.operators, "angles": res.optimized.angles}


def choose(rows) -> dict:
    """VAL rule: lowest mean CX among configs with every VAL geometry within target, else most within target."""
    by = {}
    for r in rows:
        by.setdefault(str(r["config"]), []).append(r)
    def key(k):
        v = by[k]
        ok = np.mean([x["status"] == "valid_meets_target" for x in v])
        return (-ok, np.mean([x["cx"] for x in v]))
    best = min(by, key=key)
    return by[best][0]["config"]


def val_operating_point(rows, curve):
    """budget_frontier.py's rule: smallest mean VAL CX level with every VAL geometry within target."""
    lv_ok = []
    for lv in sorted({r["level"] for r in rows if r["curve"] == curve}):
        v = [r for r in rows if r["curve"] == curve and r["level"] == lv and r["split"] == "val"]
        if v and max(abs(r["err_mha"]) for r in v) <= TARGET:
            lv_ok.append((np.mean([r["cnot_count"] for r in v]), lv))
    return min(lv_ok)[1] if lv_ok else None


def matched(frontier, curve, gid, cx):
    """|err| of `curve` at the largest-CX level whose CX <= cx on geometry gid (None if no level fits)."""
    pts = [r for r in frontier if r["curve"] == curve and r["geometry_id"] == gid and r["cnot_count"] <= cx]
    return abs(max(pts, key=lambda r: r["cnot_count"])["err_mha"]) if pts else None


def _line(name, cls, cx, err, evals, extra=""):
    e = np.abs(np.asarray(err, float))
    return (f"| {name} | {cls} | {len(e)} | {np.mean(cx):.0f} | {np.median(e):.3g} | {np.max(e):.3g} "
            f"| {100 * np.mean(e <= TARGET):.0f}% | {np.mean(evals):.0f} |{extra}")


def summarize(system, rows, frontier, chosen) -> list[str]:
    out = [f"## {system}", "", "VAL-chosen optimiser settings: " + "; ".join(
        f"{s}/{m}: {chosen[(s, m)]}" for s, m in sorted(chosen)), ""]
    for gname, pred in GROUPS.items():
        test = [r for r in rows if r["phase"] == "test" and pred(r["split"])]
        if not test:
            continue
        out += [f"### {system} — {gname}", "",
                "| circuit | cost class | n | mean CX | median abs err (mHa) | max abs err | <= 1.6 mHa | mean quantum evals "
                "| same-CX KRR-prefix <= 1.6 | same-CX truncated ADAPT <= 1.6 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for s in STARTS:
            base = [r for r in test if r["start"] == s and r["mode"] == "oracle"]
            out.append(_line(f"{s} (as generated)", "one-shot" if s != "adapt" else "ADAPT run",
                             [r["start_cx"] for r in base], [r["start_err_mha"] for r in base],
                             [r["propose_evals"] for r in base], " – | – |"))
            for m in ("oracle", "blind"):
                sel = [r for r in test if r["start"] == s and r["mode"] == m]
                cmp_ = []
                for curve in ("regress_prefix", "adapt"):
                    errs = [matched(frontier, curve, r["geometry_id"], r["cx"]) for r in sel]
                    cmp_.append(f"{100 * np.mean([e is not None and e <= TARGET for e in errs]):.0f}%")
                out.append(_line(f"{s} + prune/refine ({m})", "+ energy evals", [r["cx"] for r in sel],
                                 [r["err_mha"] for r in sel], [r["propose_evals"] + r["evals_used"] for r in sel],
                                 f" {cmp_[0]} | {cmp_[1]} |"))
        for curve in FRONTIER_CURVES:
            lv = val_operating_point(frontier, curve)
            sel = [r for r in frontier if r["curve"] == curve and r["level"] == lv and pred(r["split"])]
            if lv is None or not sel:
                out.append(f"| {curve} (VAL-chosen) | – | 0 | no level keeps every VAL geometry within target | | | | | | |")
                continue
            cls = "ADAPT run, truncated" if curve == "adapt" else "one-shot"
            out.append(_line(f"{curve} @ level {lv} (VAL-chosen)", cls, [r["cnot_count"] for r in sel],
                             [r["err_mha"] for r in sel], [r["energy_evals"] + r["pool_gradient_evals"] for r in sel],
                             " – | – |"))
        out.append("")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["h4", "lih6"])
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    md = ["# Hardware-aware optimisation vs the non-neural CX frontier (lane C)", "",
          "Generated by `python -m scripts.run_hw_optimization`. Target 1.6 mHa vs FCI of the active-space "
          "Hamiltonian. CX = Qiskit CX-basis count, transpile level 1 (same convention as "
          "`results/<s>/budget_frontier.jsonl`). Quantum evals = evaluations to propose the circuit + energy "
          "evaluations spent by prune/refine (Evaluator calls; one BFGS gradient call counts as one, as everywhere "
          "in this repo — parameter-shift on hardware would cost 2 x n_params per gradient). 'oracle' prunes "
          "against the exact reference; 'blind' only against the energy increase from its own start (tol on VAL). "
          "Same-CX columns: fraction of geometries where the KRR-prefix curve / truncated ADAPT at the largest "
          "level with CX <= the optimised circuit's CX is within 1.6 mHa (per-geometry level choice, i.e. an "
          "optimistic comparator).", ""]
    for system in a.systems:
        val, grid_test = load_hamiltonians(system, "val"), load_hamiltonians(system, "test")
        fresh = load_hamiltonians(system, FRESH_SPLIT)
        starts = start_circuits(system, val + grid_test + fresh)
        rows, chosen = [], {}
        with ProcessPoolExecutor(a.workers) as pool:
            tune = [(r, starts[s][r.geometry_id], s, m, cfg, "val") for s in STARTS for m in CONFIGS
                    for cfg in CONFIGS[m] for r in val]
            rows += list(pool.map(run_one, tune, chunksize=2))
            for s, m in itertools.product(STARTS, CONFIGS):
                chosen[(s, m)] = choose([r for r in rows if r["start"] == s and r["mode"] == m])
            jobs = [(r, starts[s][r.geometry_id], s, m, chosen[(s, m)], "test") for s, m in chosen
                    for r in grid_test + fresh]
            rows += list(pool.map(run_one, jobs, chunksize=2))
        assert all(r["same_hamiltonian_ok"] for r in rows)
        dump_jsonl([r for r in rows if r["split"] != FRESH_SPLIT], results_dir(False) / system / "hw_opt.jsonl")
        dump_jsonl([r for r in rows if r["split"] == FRESH_SPLIT], results_dir(True) / system / "hw_opt.jsonl")
        frontier = load_jsonl(ROOT / "results" / system / "budget_frontier.jsonl")
        md += summarize(system, rows, frontier, chosen)
        print("\n".join(md[-60:]))
    (ROOT / "results" / "hw_opt.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
