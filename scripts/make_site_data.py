"""Fill the static site (site/) from the committed result files, so every number on it comes from a logged run.

Writes site/data.js (the DATA object the paper's SVG charts are drawn from) and rewrites every
`<!-- gen:NAME -->...<!-- /gen:NAME -->` block in site/index.html and site/benchmarks.html with a table
built from results/, results_fresh/, results/hw_opt.md and results/ablations.md.
Separate from make_site_cover.py: that one renders images, this one emits page data and tables.

    .venv/bin/python -m scripts.make_site_data
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np

from genqai.data import FRESH_SPLIT, fresh_regime, load_hamiltonians, results_dir
from genqai.evaluate import summarize
from genqai.figures import CHEM_ACC, SYSTEMS, load_results, method_order

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
RES = ROOT / "results"

SYSTEM_NAME = {"h2": "H₂ · 2 qubits", "lih4": "LiH · 4 qubits", "lih6": "LiH · 6 qubits", "h4": "H₄ chain · 8 qubits"}
BASE = {
    "hf": "Hartree–Fock", "ccsd": "CCSD (classical)", "uccsd_vqe": "UCCSD-VQE",
    "adapt": "Fresh ADAPT-VQE", "adapt_warm": "Warm-started ADAPT (nearest training geometry)",
    "nn_reuse": "Copy nearest training circuit (naive)", "nn_refine": "Copy nearest (naive) + refine",
    "nn_reuse_gauge": "Copy nearest, orbital signs fixed", "nn_refine_gauge": "Copy nearest, signs fixed + refine",
    "nn_reuse_canon": "Copy nearest, canonical sequence", "nn_refine_canon": "Copy nearest, canonical + refine",
    "angle_regression": "Kernel-ridge angle regression, one shot", "angle_interp": "Angle interpolation, one shot",
}
FAMILY = {"sft": "Transformer (SFT)", "grpo": "Transformer (GRPO)", "distill": "Transformer (self-distilled)"}
MODE = {"oneshot": "one shot", "bestofk": "best of 16", "refine": "+ angle refinement"}
OURS = {"sft_oneshot", "grpo_oneshot", "distill_oneshot", "budget_oneshot"}

# Qollab runs that exist only in docs/qollab_runtime_checklist.md (the browser keeps no result file).
QOLLAB_RUNS = [
    {"system": "H₂", "geometry": "r = 1.33 Å", "circuit": "transformer", "backend": "Qollab · built-in simulator",
     "error": 6.6, "stderr": 5.8, "job": "–", "source": "docs/qollab_runtime_checklist.md"},
    {"system": "H₂", "geometry": "r = 1.33 Å", "circuit": "transformer", "backend": "Qollab · IonQ Forte 1 noise model",
     "error": 7.8, "stderr": 5.8, "job": "01a12826-5478-72af-b517-c3e60f663d79", "source": "docs/qollab_runtime_checklist.md"},
    {"system": "H₂", "geometry": "r = 1.33 Å", "circuit": "transformer", "backend": "Qollab · built-in simulator (re-run)",
     "error": -0.04, "stderr": 5.78, "job": "–", "source": "docs/qollab_runtime_checklist.md"},
    {"system": "H₂", "geometry": "r = 1.33 Å", "circuit": "transformer", "backend": "Qollab · IonQ Forte 1 noise model (re-run)",
     "error": 16.75, "stderr": 5.96, "job": "01a1285e-639a-7445-8d01-aa0365bdc241", "source": "docs/qollab_runtime_checklist.md"},
]
BACKEND_NAME = {"exact": "exact statevector", "exact_probs": "exact probabilities (grouped)",
                "local_shots": "local shots", "local_noisy": "local noisy (our depolarising model, not IonQ's)",
                "ionq_ideal": "IonQ ideal simulator", "ionq_noisy/aria-1": "IonQ Aria 1 noise model",
                "ionq_noisy/forte-1": "IonQ Forte 1 noise model"}


def label(m: str) -> str:
    if m in BASE:
        return BASE[m]
    if m.startswith("budget_sft"):
        return "Budget-conditioned transformer (SFT only), one shot"
    if m.startswith("budget"):
        return "Budget-conditioned transformer (GRPO), one shot"
    fam, _, mode = m.partition("_")
    return f"{FAMILY.get(fam, fam)}, {MODE.get(mode, mode)}" if mode != "refine" else f"{FAMILY.get(fam, fam)} {MODE[mode]}"


def num(x: float) -> str:
    if x != x:
        return "–"
    if x < 0.0005:
        return "&lt; 0.001"
    if x < 1:
        return f"{x:.3f}"
    if x < 100:
        return f"{x:.3g}"
    return f"{x:.0f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def table(head: list[str], rows: list[tuple[list[str], str]], caption: str) -> str:
    """rows: (cells, row class). First cell is text, the rest are numbers (td.n)."""
    out = [f'<div class="tcap">{caption}</div>', '<table class="bench many">' if len(head) >= 7 else '<table class="bench">',
           "<thead><tr>" + "".join(f'<th scope="col">{esc(h)}</th>' for h in head) + "</tr></thead>", "<tbody>"]
    for cells, cls in rows:
        c = f' class="{cls}"' if cls else ""
        out.append(f"<tr{c}><td>{cells[0]}</td>" + "".join(f'<td class="n">{v}</td>' for v in cells[1:]) + "</tr>")
    out += ["</tbody>", "</table>"]
    return '<div class="scroll wide" role="region" tabindex="0" aria-label="' + esc(re.sub("<[^>]+>", "", caption)) + '">' \
        + "\n".join(out) + "</div>"


# ------------------------------------------------------------------------------------------ per-system tables

def system_stats(system: str):
    test = summarize([e for v in load_results(results_dir(), system).values() for e in v])
    fresh_recs = load_results(results_dir(fresh=True), system)
    fresh = summarize([e for v in fresh_recs.values() for e in v])
    hams = load_hamiltonians(system, FRESH_SPLIT)
    extrap = {h.geometry_id for h in hams if fresh_regime(system, h.geometry_params) == "extrap"}
    ext = {m: summarize([e for e in v if e.geometry_id in extrap]).get(m, {}).get(FRESH_SPLIT) for m, v in fresh_recs.items()}
    return test, fresh, ext, fresh_recs, len(hams), len(extrap)


def method_rows(summ, split, ext=None):
    rows = []
    for m in method_order(summ):
        if m == "fci" or split not in summ[m]:
            continue
        r = summ[m][split]
        q = r["mean_energy_evals"] + r["mean_pool_gradient_evals"]
        cells = [f"{esc(label(m))} <code>{m}</code>", str(r["n"]), num(r["median_abs_err_mha"]), num(r["max_abs_err_mha"]),
                 pct(r["frac_under_1.6mha"]), pct(r["frac_under_5mha"]), pct(r["frac_under_10mha"]),
                 "–" if r["mean_cnot_count"] != r["mean_cnot_count"] else f"{r['mean_cnot_count']:.0f}", f"{q:.0f}"]
        if ext is not None:
            x = ext.get(m)
            cells.append(f"{num(x['median_abs_err_mha'])} / {num(x['max_abs_err_mha'])}" if x else "–")
        rows.append((cells, "ours" if m in OURS else ""))
    return rows


HEAD = ["Method", "n", "Median |err| mHa", "Worst mHa", "≤ 1.6", "≤ 5", "≤ 10", "Mean CX", "Quantum evals"]


def benchmark_tables() -> str:
    parts = []
    for s in SYSTEMS:
        test, fresh, ext, _, n_fresh, n_ext = system_stats(s)
        parts.append(f'<h3 id="{s}">{SYSTEM_NAME[s]}</h3>')
        parts.append(table(HEAD + ["Extrapolated median / worst"], method_rows(fresh, FRESH_SPLIT, ext),
                           f"{s} · test_fresh · {n_fresh} untouched geometries, {n_ext} beyond the grid edge · scored once"))
        parts.append(table(HEAD, method_rows(test, "test"),
                           f"{s} · test (grid: interp + interval + extrap) · seen while labels and training were tuned"))
    return "\n".join(parts)


# README's H4 headline table, rebuilt from the files (same rows and order as README.md).
H4_ROWS = ["hf", "ccsd", "uccsd_vqe", "adapt", "adapt_warm", "nn_reuse", "nn_reuse_gauge", "sft_oneshot", "sft_bestofk",
           "sft_refine", "angle_regression"]
LIH_ROWS = ["hf", "adapt", "nn_reuse", "sft_oneshot", "sft_refine", "angle_regression"]


def headline(system: str, keep: list[str], caption: str) -> str:
    _, fresh, ext, _, n, n_ext = system_stats(system)
    rows = []
    for m in keep:
        r = fresh[m][FRESH_SPLIT]
        q = r["mean_energy_evals"] + r["mean_pool_gradient_evals"]
        cx = "–" if r["mean_cnot_count"] != r["mean_cnot_count"] or m == "ccsd" else f"{r['mean_cnot_count']:.0f}"
        rows.append(([esc(label(m)), num(r["median_abs_err_mha"]), num(r["max_abs_err_mha"]), pct(r["frac_under_1.6mha"]),
                      cx, f"{q:.0f}"], "ours" if m in OURS else ""))
    return table(["Method", "Median |err| mHa", "Worst mHa", "≤ 1.6 mHa", "CX", "Quantum evals"], rows,
                 caption.format(n=n, ext=n_ext))


# ---------------------------------------------------------------------------- markdown tables (hw_opt, ablations)

def md_tables(path: Path) -> list[tuple[str, list[list[str]]]]:
    """[(nearest heading, rows)] for every pipe table in a markdown file."""
    out, heading, cur = [], "", None
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if cur is None:
                cur = (heading, [])
                out.append(cur)
            cur[1].append(cells)
        else:
            cur = None
    return out


def md_cell(c: str) -> str:
    c = esc(c)
    c = re.sub(r"`([^`]+)`", r"<code>\1</code>", c)
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", c)


def md_to_bench(heading: str, rows: list[list[str]], caption: str, keep=None) -> str:
    head, body = rows[0], rows[1:]
    if keep:
        body = [r for r in body if any(r[0].startswith(k) for k in keep)]
    width = len(head)
    out = []
    for r in body:
        r = (r + [""] * width)[:width]
        out.append(([md_cell(r[0])] + [md_cell(c) for c in r[1:]], "ours" if "prune/refine" in r[0] else ""))
    return table(head, out, caption)


def hw_tables(keep=None, only_fresh=False, first_table=None) -> str:
    parts = []
    for heading, rows in md_tables(RES / "hw_opt.md"):
        if only_fresh and "TEST_FRESH" not in heading:
            continue
        cap = heading.replace("—", "·").replace("TEST_FRESH", "test_fresh").replace("TEST (grid)", "test (grid)")
        tag = f"Table {first_table + len(parts)} · " if first_table else ""
        parts.append(md_to_bench(heading, rows, f"{tag}{cap} · results/hw_opt.md", keep))
    return "\n".join(parts)


def ablation_tables() -> str:
    parts = []
    for heading, rows in md_tables(RES / "ablations.md"):
        head, body = rows[0], rows[1:]
        out = [([md_cell(" · ".join(r[:2]))] + [md_cell(c) for c in r[2:]], "") for r in body]
        parts.append(table(["Model · variant"] + head[2:], out, f"{heading} · results/ablations.md"))
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------- IonQ runs

def ionq_runs() -> list[dict]:
    runs = []
    for s in SYSTEMS:
        for p in sorted((RES / s).glob("ionq_demo_*.json")):
            d = json.loads(p.read_text())
            g = ", ".join(f"{k} = {v:g} Å" for k, v in d["geometry_params"].items())
            for r in d["runs"]:
                lab = r["meta"].get("label", r["backend"])
                if lab in ("exact", "exact_probs"):
                    continue
                runs.append({"system": {"h2": "H₂", "lih4": "LiH (4q)"}.get(s, s), "geometry": g,
                             "circuit": f"{d['circuit']['source']} · {d['qiskit_gate_stats']['cnot_count']} CX",
                             "backend": BACKEND_NAME.get(lab, lab), "error": r["error_mha"], "stderr": 1000 * r["stderr"],
                             "shots": r["total_shots"], "groups": d["num_groups"],
                             "job": r["meta"].get("ionq_job_id") or "–", "source": str(p.relative_to(ROOT))})
    for p in sorted(RES.glob("h2/execution_ionq_*.json")) + [RES / "custom_heh+_ionq_forte1.json"]:
        d = json.loads(p.read_text())
        heh = "heh" in p.name
        runs.append({"system": "HeH⁺ (custom)" if heh else "H₂", "geometry": "He–H 0.772 Å" if heh else "r = 0.8 Å",
                     "circuit": f"adapt · packed {d['num_groups']} groups on {d['width']} qubits",
                     "backend": f"IonQ {d['noise_model'].replace('forte-1', 'Forte 1')} noise model (API)",
                     "error": d["error_mha"], "stderr": 1000 * d["stderr"], "shots": d["shots"], "groups": d["num_groups"],
                     "job": d["job_id"], "source": str(p.relative_to(ROOT))})
    return runs


def ionq_table(runs) -> str:
    rows = []
    for r in runs + QOLLAB_RUNS:
        rows.append(([f"{r['system']} · {esc(r['geometry'])}", esc(r["circuit"]), esc(r["backend"]),
                      f"{r['error']:+.1f} ± {r['stderr']:.1f}", str(r.get("shots", "–")), f"<code>{esc(r['job'])}</code>",
                      f"<code>{esc(r['source'])}</code>"], "ours" if "Forte 1" in r["backend"] else ""))
    return table(["System · geometry", "Circuit", "Backend", "E − E_FCI mHa", "Total shots", "IonQ job", "Record"], rows,
                 "Every measured run · error = measured energy − FCI, ± one standard error")


# --------------------------------------------------------------------------------------------- chart data

def chart_data() -> dict:
    out = {"chemAcc": CHEM_ACC}
    # Fig: H4 test_fresh per-geometry |err| by method
    _, fresh, _, recs, _, _ = system_stats("h4")
    strip = []
    for m in ["hf", "nn_reuse", "nn_reuse_gauge", "uccsd_vqe", "ccsd", "sft_oneshot", "sft_bestofk", "angle_regression",
              "sft_refine", "adapt"]:
        errs = sorted(abs(e.error_mha) for e in recs[m])
        q = fresh[m][FRESH_SPLIT]
        strip.append({"m": m, "label": label(m), "errs": [round(x, 6) for x in errs],
                      "median": q["median_abs_err_mha"], "evals": round(q["mean_energy_evals"] + q["mean_pool_gradient_evals"]),
                      "within": q["frac_under_1.6mha"]})
    out["h4Strip"] = strip
    # Fig: learning curve (test_fresh), median over subset seeds
    lc = [json.loads(line) for line in (RES / "learning_curve.jsonl").read_text().splitlines() if line.strip()]
    curves = {}
    for s in ("h4", "lih6"):
        rs = [r for r in lc if r["system"] == s and r["split"] == FRESH_SPLIT]
        ns = sorted({r["n_train"] for r in rs})
        curves[s] = {"n": ns, "teacher": [round(np.mean([r["teacher_evals"] for r in rs if r["n_train"] == n])) for n in ns],
                     "series": {}}
        for m in ("sft_oneshot", "angle_regression", "nn_reuse_gauge"):
            pts = [[r["median_abs_err_mha"] for r in rs if r["method"] == m and r["n_train"] == n] for n in ns]
            curves[s]["series"][m] = [{"mid": float(np.median(p)), "lo": min(p), "hi": max(p), "k": len(p)} for p in pts]
    out["learning"] = curves
    # Fig: accuracy vs CX frontier (test_fresh): frontier curves + one-shot points + prune/refine
    front = {}
    for s in ("h4", "lih6"):
        seen = {}
        for p in sorted((RES / s).glob("*_frontier.jsonl")):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                seen[(r["curve"], r["level"], r["geometry_id"])] = r
        rows = [r for r in seen.values() if r["split"] == FRESH_SPLIT]
        curves_s = {}
        for c in ("regress_prefix", "adapt", "budget"):
            pts = []
            for lv in sorted({r["level"] for r in rows if r["curve"] == c}):
                sel = [r for r in rows if r["curve"] == c and r["level"] == lv]
                pts.append({"cx": float(np.mean([r["cnot_count"] for r in sel])),
                            "within": float(np.mean(np.abs([r["err_mha"] for r in sel]) <= CHEM_ACC)), "n": len(sel)})
            curves_s[c] = pts
        hw = [json.loads(line) for line in (ROOT / "results_fresh" / s / "hw_opt.jsonl").read_text().splitlines()]
        hw = [r for r in hw if r["phase"] == "test" and r["split"] == FRESH_SPLIT]
        pr = {}
        for start in ("sft", "krr", "adapt"):
            for mode in ("oracle", "blind"):
                sel = [r for r in hw if r["start"] == start and r["mode"] == mode]
                pr[f"{start}_{mode}"] = {
                    "cx": float(np.mean([r["cx"] for r in sel])),
                    "within": float(np.mean([r["status"] == "valid_meets_target" for r in sel])),
                    "evals": float(np.mean([r["evals_used"] for r in sel])),
                    "startCx": float(np.mean([r["start_cx"] for r in sel])),
                    "startWithin": float(np.mean(np.abs([r["start_err_mha"] for r in sel]) <= CHEM_ACC))}
        front[s] = {"curves": curves_s, "prune": pr}
    out["frontier"] = front
    # Fig: IonQ measured errors
    runs = ionq_runs()
    pick = [r for r in runs if r["backend"].startswith("IonQ") and "ideal" not in r["backend"]]
    q = [r for r in QOLLAB_RUNS if "IonQ" in r["backend"]]
    out["ionq"] = [{"label": f"{r['system']} · {r['geometry']}", "backend": r["backend"], "err": round(r["error"], 2),
                    "se": round(r["stderr"], 2), "job": r["job"]} for r in q + pick]
    return out


def replace_blocks(path: Path, blocks: dict[str, str]) -> None:
    text = path.read_text()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- gen:{name} -->)(.*?)(<!-- /gen:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"{path.name}: no gen:{name} block")
        text = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    path.write_text(text)


def main():
    data = chart_data()
    (SITE / "data.js").write_text("/* Generated by scripts/make_site_data.py from results/ and results_fresh/. Do not edit. */\n"
                                  "const DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                                                               default=lambda x: None if x != x else x) + ";\n")
    replace_blocks(SITE / "index.html", {
        "h4": headline("h4", H4_ROWS, "Table 2 · H₄ chain, 8 qubits · test_fresh, {n} untouched geometries ({ext} beyond the grid edge) · scored once"),
        "lih": headline("lih4", LIH_ROWS, "Table 3 · LiH, 4 qubits · test_fresh, {n} geometries ({ext} extrapolated)") + "\n"
               + headline("lih6", LIH_ROWS, "Table 4 · LiH, 6 qubits · test_fresh, {n} geometries ({ext} extrapolated)"),
        "hw": hw_tables(keep=["sft (as", "sft + prune/refine (oracle)", "krr (as", "krr + prune/refine (oracle)",
                              "adapt (as", "regress_prefix", "adapt @"], only_fresh=True, first_table=5),
    })
    replace_blocks(SITE / "benchmarks.html", {
        "systems": benchmark_tables(), "hw": hw_tables(), "ablations": ablation_tables(), "ionq": ionq_table(ionq_runs()),
    })
    print("wrote site/data.js and the generated tables in site/index.html, site/benchmarks.html")


if __name__ == "__main__":
    main()
