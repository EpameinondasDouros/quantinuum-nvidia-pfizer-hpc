"""Paper figures + README results table from results/<system>/*.jsonl (Workstream F).

    make_all(results_dir, out_dir) -> {figure_name: path or skip reason}

Methods are discovered from the EvalRecord jsonl files present; anything that is not a classical/quantum
baseline (genqai.baselines.METHODS) is treated as a generator method and drawn in the generator style
(warm hues by inference mode, marker by training family). Missing inputs are skipped, never faked.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from genqai.baselines import METHODS  # noqa: E402
from genqai.data import FRESH_SPLIT, hamiltonians_path, load_hamiltonians  # noqa: E402
from genqai.evaluate import summarize  # noqa: E402
from genqai.schema import EvalRecord, load_jsonl  # noqa: E402

SYSTEMS = ("h2", "lih4", "lih6", "h4")
SYSTEM_LABEL = {"h2": "H$_2$ (2q)", "lih4": "LiH (4q)", "lih6": "LiH (6q)", "h4": "H$_4$ (8q)"}
TEST_SPLITS = ("test_interp", "test_interval", "test_extrap")
CHEM_ACC = 1.6  # mHa
ERR_FLOOR = 1e-6  # mHa; |errors| below this are drawn at the floor (numerically exact)

# chart chrome (reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "#e4eefb"  # chemical-accuracy band: a pale wash of the blue ramp
# fixed method -> color (color follows the entity, never its rank)
BASE_COLOR = {
    "fci": INK, "hf": MUTED, "ccsd": "#008300",
    "adapt": "#2a78d6", "adapt_warm": "#1baf7a", "nn_reuse": "#eda100", "nn_refine": "#e87ba4",
    "uccsd_vqe": "#6250d6", "nn_reuse_gauge": "#b07c00", "nn_refine_gauge": "#c2508a",
    "nn_reuse_canon": "#7a5600", "nn_refine_canon": "#93306a", "angle_regression": "#0b8a8a",
    "angle_interp": "#5aa9a9",
}
BASE_LABEL = {
    "fci": "FCI", "hf": "HF", "ccsd": "CCSD", "adapt": "ADAPT (fresh)", "adapt_warm": "ADAPT (warm)",
    "nn_reuse": "NN reuse (naive)", "nn_refine": "NN refine (naive)", "uccsd_vqe": "UCCSD-VQE",
    "nn_reuse_gauge": "NN reuse (sign-fixed)", "nn_refine_gauge": "NN refine (sign-fixed)",
    "nn_reuse_canon": "NN reuse (canonical)", "nn_refine_canon": "NN refine (canonical)",
    "angle_regression": "Angle KRR", "angle_interp": "Angle interp",
}
# generator: hue by inference mode (warm family), marker by training family
GEN_MODE_COLOR = {"oneshot": "#eb6834", "refine": "#e34948", "bestofk": "#a8431c"}
GEN_FAMILY_MARKER = {"sft": "D", "grpo": "^", "distill": "s", "budget": "v"}
FIG2_ORDER = ("nn_reuse", "nn_refine", "adapt_warm", "adapt", "uccsd_vqe")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


# ----------------------------------------------------------------------------------------- data

def is_generator(method: str) -> bool:
    return method not in METHODS


def style(method: str) -> dict:
    """Color/marker/label for any method name; generator methods share one consistent style."""
    if not is_generator(method):
        return {"color": BASE_COLOR[method], "marker": "o", "label": BASE_LABEL[method]}
    fam, _, mode = method.partition("_")
    color = next((c for k, c in GEN_MODE_COLOR.items() if mode.startswith(k)), "#c25a2a")
    return {"color": color, "marker": GEN_FAMILY_MARKER.get(fam, "P"), "label": method.replace("_", " ")}


def method_order(methods) -> list[str]:
    base = [m for m in FIG2_ORDER if m in methods]
    rest = [m for m in METHODS if m in methods and m not in base]
    return base + rest + sorted(m for m in methods if is_generator(m))


def load_results(results_dir: Path, system: str) -> dict[str, list[EvalRecord]]:
    """{method: [EvalRecord]} for every jsonl in results/<system> that holds EvalRecords."""
    out = {}
    for p in sorted((Path(results_dir) / system).glob("*.jsonl")):
        with p.open() as f:
            first = f.readline()
        if not first.strip() or not {"method", "error_mha", "geometry_id"} <= set(json.loads(first)):
            continue  # not an EvalRecord file (e.g. generated circuits)
        for e in load_jsonl(p, EvalRecord):
            out.setdefault(e.method, []).append(e)
    return out


def geometry_params(system: str) -> dict[str, dict[str, float]]:
    """Grid geometries plus, when built, the test_fresh ones (results_fresh/ figures)."""
    recs = load_hamiltonians(system)
    if hamiltonians_path(system, fresh=True).exists():
        recs += load_hamiltonians(system, FRESH_SPLIT)
    return {h.geometry_id: h.geometry_params for h in recs}


def train_logs(results_dir: Path, system: str) -> dict[str, dict]:
    """{family: log} from results/<system>/<family>_train_log.json."""
    return {p.name.removesuffix("_train_log.json"): json.loads(p.read_text())
            for p in sorted((Path(results_dir) / system).glob("*_train_log.json"))}


def _is_test(e: EvalRecord) -> bool:
    return e.split.startswith("test")


def _abs_err(e: EvalRecord) -> float:
    return max(abs(e.error_mha), ERR_FLOOR)


def _evals(e: EvalRecord) -> int:
    return e.energy_evals + e.pool_gradient_evals


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _floor_note(ax):
    ax.text(1.0, 1.01, f"|err| < {ERR_FLOOR:g} mHa (numerically exact) drawn at {ERR_FLOOR:g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)


# ------------------------------------------------------------------------------- figure 1: PES

def _h4_slice(params, recs) -> float:
    """The a value whose b-slice contains the most distinct splits (ties -> most test points)."""
    by_a = defaultdict(list)
    for e in recs:
        by_a[params[e.geometry_id]["a"]].append(e.split)
    return max(by_a, key=lambda a: (len(set(by_a[a])), sum(s.startswith("test") for s in by_a[a]), -a))


def fig_pes(data, params, out_dir):
    systems = [s for s in SYSTEMS if "fci" in data.get(s, {})]
    if not systems:
        return None
    fig, axes = plt.subplots(2, len(systems), figsize=(3.4 * len(systems), 5.6), sharex="col",
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08, "wspace": 0.32}, squeeze=False)
    for j, s in enumerate(systems):
        res, prm = data[s], params[s]
        if s == "h4":
            a0 = _h4_slice(prm, res["fci"])
            keep = lambda e, a0=a0: prm[e.geometry_id]["a"] == a0  # noqa: E731
            xkey, xlabel, title = "b", "b (Å)", f"{SYSTEM_LABEL[s]}, slice a = {a0:.2f} Å"
        else:
            keep, xkey, xlabel, title = (lambda e: True), "r", "bond length r (Å)", SYSTEM_LABEL[s]
        ax, rx = axes[0, j], axes[1, j]
        fci = {e.geometry_id: e for e in res["fci"] if keep(e)}
        xs = {g: prm[g][xkey] for g in fci}
        # test-geometry markers: shaded contiguous spans + hairlines for isolated interp points
        for split, kind in (("test_interval", "span"), ("test_extrap", "span"), ("test_interp", "line")):
            gx = sorted(xs[g] for g, e in fci.items() if e.split == split)
            if not gx:
                continue
            for a in (ax, rx):
                if kind == "span":
                    step = np.diff(sorted(xs.values())).min() / 2 if len(xs) > 1 else 0.05
                    a.axvspan(gx[0] - step, gx[-1] + step, color="#f0efec", lw=0, zorder=0)
                else:
                    for x in gx:
                        a.axvline(x, color=AXIS, lw=0.8, zorder=0)
        for split, lab in (("test_interval", "interval"), ("test_extrap", "extrap")):
            gx = [xs[g] for g, e in fci.items() if e.split == split]
            if gx:
                ax.text(np.mean(gx), 0.98, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=7, color=MUTED)
        order = sorted(fci, key=xs.get)
        x = np.array([xs[g] for g in order])
        ax.plot(x, [fci[g].energy for g in order], color=INK, lw=1.5, label="FCI", zorder=3)
        methods = [m for m in ("hf", "ccsd", "adapt") if m in res] + sorted(m for m in res if is_generator(m))
        rx.axhspan(-CHEM_ACC, CHEM_ACC, color=BAND, lw=0, zorder=0.5)
        for m in methods:
            st = style(m)
            by = {e.geometry_id: e for e in res[m] if e.geometry_id in fci}
            g = [k for k in order if k in by and not math.isnan(by[k].energy)]
            xm = np.array([xs[k] for k in g])
            em = np.array([by[k].energy for k in g])
            err = np.array([by[k].error_mha for k in g])
            if m in ("hf", "ccsd"):
                ax.plot(xm, em, color=st["color"], lw=1.2, label=st["label"], zorder=2)
                rx.plot(xm, err, color=st["color"], lw=1.2, zorder=2)
            else:
                kw = dict(marker=st["marker"], ms=4.5 if m == "adapt" else 5, mfc=st["color"],
                          mec=SURFACE, mew=0.8, ls="none", color=st["color"], zorder=4)
                ax.plot(xm, em, label=st["label"], **kw)
                rx.plot(xm, err, **kw)
        rx.set_yscale("symlog", linthresh=0.1, linscale=0.6)
        for a in (ax, rx):
            a.grid(axis="x", visible=False)  # vertical hairlines are reserved for test_interp geometries
        rx.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_title(title, loc="left")
        rx.set_xlabel(xlabel)
        if j == 0:
            ax.set_ylabel("energy (Ha)")
            rx.set_ylabel("E − E$_{FCI}$ (mHa, symlog)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Patch(color=BAND, lw=0), Patch(color="#f0efec", lw=0), Line2D([], [], color=AXIS, lw=0.8)]
    labels += ["±1.6 mHa (chem. accuracy)", "test interval / extrap", "test interp"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Potential-energy curves and residual vs FCI", x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(bottom=0.16)
    return _save(fig, out_dir, "fig1_pes")


# ------------------------------------------------------------------- figure 2: test error strips

def fig_test_errors(data, out_dir):
    systems = [s for s in SYSTEMS if s in data]
    if not systems:
        return None
    fig, axes = plt.subplots(len(systems), 1, figsize=(10, 2.9 * len(systems)), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, s in zip(axes[:, 0], systems):
        res = data[s]
        methods = [m for m in method_order(res) if m not in ("hf", "ccsd", "fci")]
        ticklabels = []
        for i, m in enumerate(methods):
            t = [e for e in res[m] if _is_test(e)]
            err = np.array([_abs_err(e) for e in t if not math.isnan(e.error_mha)])
            st = style(m)
            if err.size:
                ax.boxplot([err], positions=[i], widths=0.5, showfliers=False, whis=(0, 100),
                           medianprops={"color": INK, "lw": 1.5}, boxprops={"color": AXIS},
                           whiskerprops={"color": AXIS}, capprops={"color": AXIS})
                ax.scatter(i + rng.uniform(-0.17, 0.17, err.size), err, s=16, color=st["color"],
                           marker=st["marker"], edgecolors=SURFACE, linewidths=0.6, zorder=3)
            med = np.median(err) if err.size else math.nan
            med_txt = f"< {ERR_FLOOR:g}" if med <= ERR_FLOOR else f"{med:.2g}"
            ticklabels.append(f"{st['label']}\nmed {med_txt} mHa\nE {np.mean([e.energy_evals for e in t]):.0f}"
                              f" · G {np.mean([e.pool_gradient_evals for e in t]):.0f}")
        ax.axhspan(ERR_FLOOR / 2, CHEM_ACC, color=BAND, lw=0, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(ERR_FLOOR / 2, None)
        ax.set_xticks(range(len(methods)), ticklabels, fontsize=7.5)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_ylabel("|E − E$_{FCI}$| (mHa)")
        n = len({e.geometry_id for m in methods for e in res[m] if _is_test(e)})
        ax.set_title(f"{SYSTEM_LABEL[s]} — {n} test geometries", loc="left")
        ax.grid(axis="x", visible=False)
        _floor_note(ax)
    fig.text(0.01, -0.005, "Shaded: below 1.6 mHa. E = mean energy evaluations, G = mean pool-gradient "
             "evaluations per test geometry.", fontsize=7.5, color=INK2)
    fig.suptitle("Per-geometry error on held-out test geometries", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(h_pad=1.2)
    return _save(fig, out_dir, "fig2_test_errors")


# --------------------------------------------------------------------- figure 3: accuracy vs cost

SYSTEM_MARKER = {"h2": "o", "lih4": "s", "lih6": "^", "h4": "D"}


def fig_accuracy_cost(data, out_dir):
    """Small multiples: one column per system; top x = mean CX, bottom x = quantum evaluations."""
    pts = defaultdict(list)
    for s in SYSTEMS:
        for m in method_order(data.get(s, {})):
            if m in ("hf", "ccsd", "fci"):
                continue
            t = [e for e in data[s][m] if _is_test(e) and not math.isnan(e.error_mha)]
            cx = [e.cnot_count for e in t if e.cnot_count is not None]
            if t and cx:
                pts[s].append((m, np.mean(cx), np.mean([_evals(e) for e in t]),
                               max(np.median([abs(e.error_mha) for e in t]), ERR_FLOOR)))
    if not pts:
        return None
    systems = [s for s in SYSTEMS if s in pts]
    fig, axes = plt.subplots(2, len(systems), figsize=(3.2 * len(systems), 6.4), sharey=True, squeeze=False)
    for j, s in enumerate(systems):
        # methods often share one circuit size: dodge only points with the same rounded CX so none hide
        step = 0.025 * max(p[1] for p in pts[s])
        clusters, xpos = [], {}
        for m, cx, *_ in sorted(pts[s], key=lambda p: p[1]):
            if clusters and cx - clusters[-1][-1][1] < step:
                clusters[-1].append((m, cx))
            else:
                clusters.append([(m, cx)])
        for c in clusters:
            centre = np.mean([cx for _, cx in c])
            for k, (m, _) in enumerate(c):
                xpos[m] = centre + step * (k - (len(c) - 1) / 2) if len(c) > 1 else c[0][1]
        for m, cx, ev, err in pts[s]:
            st = style(m)
            for ax, x in ((axes[0, j], xpos[m]), (axes[1, j], ev)):
                ax.scatter(x, err, s=52, color=st["color"], marker=st["marker"], edgecolors=SURFACE,
                           linewidths=1.2, zorder=3, label=st["label"])
        cxs = [p[1] for p in pts[s]]
        axes[0, j].set_xlim(min(cxs) / 1.3, max(cxs) * 1.3)
        axes[0, j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
        axes[1, j].set_xscale("symlog", linthresh=1)
        axes[1, j].set_xlim(-0.4, max(p[2] for p in pts[s]) * 2.5)
        axes[0, j].set_title(SYSTEM_LABEL[s], loc="left")
        axes[0, j].set_xlabel("mean CNOT count (Qiskit CX basis)")
        axes[1, j].set_xlabel("quantum evals / geometry (symlog)")
    for ax in axes.flat:
        ax.axhspan(ERR_FLOOR / 2, CHEM_ACC, color=BAND, lw=0, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(ERR_FLOOR / 2, None)
    for i in range(2):
        axes[i, 0].set_ylabel("median test |E − E$_{FCI}$| (mHa)")
    seen = {}
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=min(len(seen), 8), bbox_to_anchor=(0.5, -0.04))
    fig.text(0.01, -0.08, "Top: circuit cost (CX count, horizontally dodged where methods share a circuit size). "
             "Bottom: quantum evaluations = energy + pool-gradient evaluations per test geometry. "
             f"Shaded: below 1.6 mHa; medians < {ERR_FLOOR:g} mHa drawn at {ERR_FLOOR:g}.", fontsize=7.5, color=INK2)
    fig.suptitle("Accuracy vs cost on test geometries", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_accuracy_cost")


# ------------------------------------------------------------------ figure 4: training curves

def fig_training(logs, out_dir):
    series = [(s, fam, log) for s, by in logs.items() for fam, log in by.items() if log.get("evals")]
    if not series:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for s, fam, log in series:
        ev = log["evals"]
        ep = [r["epoch"] for r in ev]
        mk = SYSTEM_MARKER.get(s, "o")
        c = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"][SYSTEMS.index(s)] if s in SYSTEMS else INK2
        lab = f"{SYSTEM_LABEL.get(s, s)} {fam}"
        axes[0].plot(ep, [r["train"][0] for r in ev], color=c, label=lab, marker=mk, ms=4)
        axes[1].plot(ep, [r["val_mean_abs_mha"] for r in ev], color=c, label=lab, marker=mk, ms=4)
    axes[0].set_ylabel("training loss")
    axes[1].set_ylabel("val mean |err| one-shot (mHa)")
    axes[1].set_yscale("log")
    axes[1].axhline(CHEM_ACC, color=INK2, lw=1)
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend()
    fig.suptitle("Generator training", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_training")


# ------------------------------------------------------------------------ figure 5: break-even

COST_KINDS = {"evals": ("quantum evaluations", _evals), "wall": ("wall-clock (s)", lambda e: e.runtime_s)}


def breakeven(data, logs) -> dict:
    """Cumulative cost C(M) = fixed + M * per_geometry for each amortized method vs fresh ADAPT.

    fixed = sum of teacher (fresh ADAPT) costs over the training geometries, plus logged model training
    time for generator methods (wall-clock only; training uses no quantum evaluations).
    """
    out = {}
    for s in SYSTEMS:
        res = data.get(s, {})
        if "adapt" not in res:
            continue
        train = [e for e in res["adapt"] if e.split == "train"]
        out[s] = {}
        for kind, (_, f) in COST_KINDS.items():
            c_fresh = float(np.mean([f(e) for e in res["adapt"] if _is_test(e)]))
            teacher = float(sum(f(e) for e in train))
            rows = {"adapt": {"fixed": 0.0, "per_geometry": c_fresh, "break_even_M": 0.0}}
            for m in [m for m in ("adapt_warm", "nn_refine") if m in res] + sorted(m for m in res if is_generator(m)):
                t = [e for e in res[m] if _is_test(e)]
                if not t:
                    continue
                c = float(np.mean([f(e) for e in t]))
                fixed, note = teacher, "teacher ADAPT on training geometries"
                if is_generator(m):
                    log = logs.get(s, {}).get(m.split("_")[0])
                    if kind == "wall":
                        if log and "train_time_s" in log:
                            fixed += float(log["train_time_s"])
                            note += " + model training time"
                        else:
                            note += "; training time not logged"
                    else:
                        note += " (training uses no quantum evaluations)"
                be = fixed / (c_fresh - c) if c < c_fresh else None  # None: never breaks even
                rows[m] = {"fixed": fixed, "per_geometry": c, "break_even_M": be, "fixed_note": note}
            out[s][kind] = {"n_train": len(train), "methods": rows}
    return out


def fig_breakeven(be, out_dir):
    systems = [s for s in SYSTEMS if s in be]
    if not systems:
        return None
    fig, axes = plt.subplots(2, len(systems), figsize=(3.3 * len(systems), 6.2), squeeze=False)
    for j, s in enumerate(systems):
        for i, kind in enumerate(COST_KINDS):
            ax, rows = axes[i, j], be[s][kind]["methods"]
            finite = [r["break_even_M"] for m, r in rows.items() if m != "adapt" and r["break_even_M"] is not None]
            mmax = float(np.clip(1.6 * max(finite, default=50), 20, 1e5))
            M = np.linspace(0, mmax, 200)
            for m, r in rows.items():
                st = style(m)
                ax.plot(M, r["fixed"] + M * r["per_geometry"], color=st["color"], label=st["label"])
                b = r["break_even_M"]
                if m != "adapt" and b is not None and b <= mmax:
                    ax.plot(b, r["fixed"] + b * r["per_geometry"], "o", ms=6, color=st["color"], mec=SURFACE, mew=1.2)
                    above = m == "adapt_warm"  # warm label above-left, others below-right: no collisions
                    ax.annotate(f"M*={b:.0f}", (b, r["fixed"] + b * r["per_geometry"]),
                                xytext=(-6, 6) if above else (6, -10), ha="right" if above else "left",
                                textcoords="offset points", fontsize=7, color=INK2)
            ax.set_ylim(0, None)
            ax.set_xlim(0, mmax)
            if i == 0:
                ax.set_title(f"{SYSTEM_LABEL[s]} — {be[s][kind]['n_train']} training geom.", loc="left")
            if i == 1:
                ax.set_xlabel("new geometries M")
            if j == 0:
                ax.set_ylabel(f"cumulative {COST_KINDS[kind][0]}")
    seen = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=len(seen), bbox_to_anchor=(0.5, -0.03))
    gen = any(is_generator(m) for s in be for k in be[s] for m in be[s][k]["methods"])
    note = "Fixed cost = teacher ADAPT on all training geometries" + (
        " (+ logged model training time for generators)." if gen else ". No generator results present.")
    fig.text(0.01, -0.07, note + " Dots: break-even M* vs fresh ADAPT.", fontsize=7.5, color=INK2)
    fig.suptitle("Amortization: cumulative cost vs number of new geometries", x=0.01, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_breakeven")


# ----------------------------------------------------------------------------- figure 6: IonQ

IONQ_ROWS = ("exact", "local_shots", "local_noisy", "ionq_ideal", "ionq_noisy/aria-1", "ionq_noisy/forte-1")
IONQ_LABEL = {"exact": "exact statevector", "local_shots": "local shots (ideal)",
              "local_noisy": "local noisy (our model)", "ionq_ideal": "IonQ sim, ideal",
              "ionq_noisy/aria-1": "IonQ sim, aria-1 noise", "ionq_noisy/forte-1": "IonQ sim, forte-1 noise"}
IONQ_COLOR = {"exact": INK, "local_shots": "#2a78d6", "local_noisy": "#6250d6", "ionq_ideal": "#1baf7a",
              "ionq_noisy/aria-1": "#eb6834", "ionq_noisy/forte-1": "#e34948"}


def fig_ionq(results_dir, out_dir):
    files = [p for s in SYSTEMS for p in sorted((Path(results_dir) / s).glob("ionq_demo_*.json"))]
    if not files:
        return None
    fig, axes = plt.subplots(len(files), 1, figsize=(9, 2.6 * len(files)), squeeze=False)
    for ax, p in zip(axes[:, 0], files):
        d = json.loads(p.read_text())
        runs = {r["meta"].get("label", r["backend"]): r for r in d["runs"]}
        rows = [k for k in IONQ_ROWS if k in runs]
        ax.axvspan(-CHEM_ACC, CHEM_ACC, color=BAND, lw=0, zorder=0)
        ax.axvline(0, color=INK, lw=1, zorder=1)
        for y, k in enumerate(rows):
            r = runs[k]
            ax.errorbar(r["error_mha"], y, xerr=1000 * r["stderr"], fmt="o", ms=7, color=IONQ_COLOR[k],
                        mec=SURFACE, mew=1.2, ecolor=IONQ_COLOR[k], elinewidth=1.5, capsize=3, zorder=3)
            txt = f"{r['error_mha']:+.2f}" + (f" ± {1000 * r['stderr']:.2f}" if r["stderr"] else "") + " mHa"
            if r["total_shots"]:
                txt += f"  ({r['total_shots']} shots)"
            if "probability_rounding_bound_ha" in r["meta"]:
                txt += f"  (prob. rounding bound {1000 * r['meta']['probability_rounding_bound_ha']:.1f} mHa)"
            ax.annotate(txt, (r["error_mha"], y), xytext=(0, 7), textcoords="offset points", fontsize=7,
                        color=INK2, ha="center", va="bottom")
        # linear within ±10 mHa so shot-noise error bars stay symmetric; log beyond for noisy runs
        ax.set_xscale("symlog", linthresh=10, linscale=1.5)
        ax.set_yticks(range(len(rows)), [IONQ_LABEL[k] for k in rows])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        lo = min(runs[k]["error_mha"] - 1000 * runs[k]["stderr"] for k in rows)
        hi = max(runs[k]["error_mha"] + 1000 * runs[k]["stderr"] for k in rows)
        ax.set_xlim(min(lo * 1.6, -10), max(hi * 2.5, 10))
        ax.grid(axis="y", visible=False)
        g, nat = d["qiskit_gate_stats"], d.get("ionq_native_stats", {})
        nat_txt = ", ".join(f"{v['device']}: {v['native_2q_count']} {v['native_2q_gate'].upper()}"
                            for v in nat.values())
        prm = ", ".join(f"{k}={v:g} Å" for k, v in d["geometry_params"].items())
        ax.set_title(f"{SYSTEM_LABEL.get(d['system'], d['system'])}, {prm} ({d['split']}), {d['circuit']['source']} circuit — "
                     f"{g['cnot_count']} CX, depth {g['depth']}; native 2q: {nat_txt}; "
                     f"{d['num_groups']} QWC groups, {d['shots_requested']} shots requested",
                     loc="left", fontsize=8.5)
    axes[-1, 0].set_xlabel("E − E$_{FCI}$ (mHa; linear within ±10, log beyond; shaded ±1.6 mHa, line = FCI)")
    fig.suptitle("Measured circuit energies: local and IonQ cloud simulators", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_ionq")


# ------------------------------------------------------------ figure 7: CX-budget frontier

FRONTIER_STYLE = {  # curve -> (color, marker, linestyle, label); regression comparators share one hue
    "budget": (GEN_MODE_COLOR["oneshot"], "v", "-", "budget-conditioned policy (one model, one-shot)"),
    "budget_sft": (GEN_MODE_COLOR["oneshot"], "v", ":", "budget policy before GRPO (budget SFT only)"),
    "adapt": (BASE_COLOR["adapt"], "o", "-", "ADAPT truncated at k ops (fresh run, re-optimised; runs that stopped earlier drop out)"),
    "regress_adapt": (BASE_COLOR["angle_regression"], "s", "-", "angle KRR on ADAPT k-op prefixes (one-shot)"),
    "regress_pruned": (BASE_COLOR["angle_regression"], "D", "--", "angle KRR on pruned n-op labels (one-shot)"),
    "regress_prefix": (BASE_COLOR["angle_regression"], "^", ":", "angle KRR on canonical n-op prefixes (one-shot)"),
}
FRONTIER_GROUPS = (("test (grid)", lambda s: s.startswith("test_") and s != "test_fresh"),
                   ("test_fresh", lambda s: s == "test_fresh"))


def fig_budget_frontier(results_dir, out_dir):
    """Per system: fraction of held-out geometries within 1.6 mHa, and median |err|, vs mean compiled CX
    for every level of each frontier curve (results/<system>/*_frontier.jsonl, scripts/budget_frontier.py)."""
    rows = {}
    for s in SYSTEMS:
        seen = {}
        for p in sorted((Path(results_dir) / s).glob("*_frontier.jsonl")):
            for r in load_jsonl(p):
                seen[(r["curve"], r["level"], r["geometry_id"])] = r
        if seen:
            rows[s] = list(seen.values())
    if not rows:
        return None
    systems = list(rows)
    fig, axes = plt.subplots(2 * len(systems), 2, figsize=(10, 3.0 * 2 * len(systems)), sharex="row", squeeze=False)
    for i, s in enumerate(systems):
        for j, (gname, pred) in enumerate(FRONTIER_GROUPS):
            ax_f, ax_e = axes[2 * i, j], axes[2 * i + 1, j]
            for curve, (color, marker, ls, label) in FRONTIER_STYLE.items():
                pts = []
                for lv in sorted({r["level"] for r in rows[s] if r["curve"] == curve}):
                    sel = [r for r in rows[s] if r["curve"] == curve and r["level"] == lv and pred(r["split"])]
                    if sel:
                        err = np.abs([r["err_mha"] for r in sel])
                        pts.append((np.mean([r["cnot_count"] for r in sel]), np.mean(err <= CHEM_ACC),
                                    max(float(np.nanmedian(err)), ERR_FLOOR)))
                if not pts:
                    continue
                x, frac, med = np.array(pts).T
                kw = dict(color=color, marker=marker, ls=ls, lw=2, ms=6, mec=SURFACE, mew=1.0, label=label)
                ax_f.plot(x, 100 * frac, **kw)
                ax_e.plot(x, med, **kw)
            sft_dir = Path(results_dir).parent / "results_fresh" if gname == "test_fresh" else Path(results_dir)
            sft = sft_dir / s / "sft_oneshot.jsonl"
            if sft.exists():
                t = [e for e in load_jsonl(sft, EvalRecord) if pred(e.split)]
                if t:
                    err = np.abs([e.error_mha for e in t])
                    x = np.mean([e.cnot_count for e in t])
                    st = style("sft_oneshot")
                    for ax, y in ((ax_f, 100 * np.mean(err <= CHEM_ACC)), (ax_e, max(np.median(err), ERR_FLOOR))):
                        ax.scatter(x, y, s=70, color=st["color"], marker=st["marker"], edgecolors=INK, linewidths=1.0,
                                   zorder=4, label="SFT one-shot (fixed canonical sequence)")
            n = len({r["geometry_id"] for r in rows[s] if pred(r["split"])})
            ax_f.set_title(f"{SYSTEM_LABEL[s]} — {gname}, {n} geometries", loc="left")
            ax_f.set_ylim(-3, 103)
            ax_f.set_ylabel("% within 1.6 mHa")
            ax_e.axhspan(ERR_FLOOR / 2, CHEM_ACC, color=BAND, lw=0, zorder=0)
            ax_e.set_yscale("log")
            ax_e.set_ylim(ERR_FLOOR / 2, None)
            ax_e.set_ylabel("median |E − E$_{FCI}$| (mHa)")
            ax_e.set_xlabel("mean compiled CX count (Qiskit CX basis)")
    seen = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    leg = fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0),
                     title="Each point = one budget level / prefix length (joined in level order), averaged over held-out "
                           "geometries.\nADAPT needs a fresh ADAPT run per geometry; all other curves are one-shot "
                           "(0 energy evaluations).", title_fontsize=7.5)
    leg.get_title().set_color(INK2)
    fig.suptitle("Accuracy vs two-qubit gate budget on held-out geometries", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_budget_frontier")


# ---------------------------------------------------------------------------- summary table

def summary_markdown(data) -> str:
    lines = ["# GenQAI results (test splits)", "",
             "Generated by `python -m scripts.make_figures`. |err| = |E − E_FCI| in mHa over all test geometries "
             "(interp + interval + extrap). CX = Qiskit CX-basis CNOT count. Evals = mean energy evaluations / "
             "pool-gradient evaluations per geometry.", ""]
    for s in SYSTEMS:
        if s not in data:
            continue
        summ = summarize([e for m in data[s] for e in data[s][m]])
        lines += [f"## {SYSTEM_LABEL[s].replace('$', '').replace('_', '')}", "",
                  "| method | n | median abs err | max abs err | < 1.6 mHa | < 5 mHa | < 10 mHa | mean CX "
                  "| energy evals | pool-grad evals |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for m in method_order(summ):
            r = summ[m].get("test")
            if r is None:
                continue
            cx = "-" if math.isnan(r["mean_cnot_count"]) else f"{r['mean_cnot_count']:.0f}"
            med, mx = (f"< {ERR_FLOOR:g}" if v < ERR_FLOOR else f"{v:.3g}"
                       for v in (r["median_abs_err_mha"], r["max_abs_err_mha"]))
            lines.append(f"| {m} | {r['n']} | {med} | {mx} "
                         f"| {100 * r['frac_under_1.6mha']:.0f}% | {100 * r['frac_under_5mha']:.0f}% "
                         f"| {100 * r['frac_under_10mha']:.0f}% | {cx} | {r['mean_energy_evals']:.1f} "
                         f"| {r['mean_pool_gradient_evals']:.1f} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- driver

def make_all(results_dir, out_dir) -> dict[str, str]:
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {s: r for s in SYSTEMS if (r := load_results(results_dir, s))}
    params = {s: geometry_params(s) for s in data}
    logs = {s: lg for s in SYSTEMS if (lg := train_logs(results_dir, s))}
    report = {}

    def run(name, fn, *a, why="no input data"):
        p = fn(*a)
        report[name] = str(p) if p else f"skipped: {why}"

    run("fig1_pes", fig_pes, data, params, out_dir)
    run("fig2_test_errors", fig_test_errors, data, out_dir)
    run("fig3_accuracy_cost", fig_accuracy_cost, data, out_dir)
    run("fig4_training", fig_training, logs, out_dir, why="no results/<system>/*_train_log.json")
    be = breakeven(data, logs)
    (out_dir / "breakeven.json").write_text(json.dumps(be, indent=1))
    run("fig5_breakeven", fig_breakeven, be, out_dir)
    run("fig6_ionq", fig_ionq, results_dir, out_dir, why="no results/<system>/ionq_demo_*.json")
    run("fig7_budget_frontier", fig_budget_frontier, results_dir, out_dir,
        why="no results/<system>/*_frontier.jsonl")
    (out_dir / "summary.md").write_text(summary_markdown(data))
    report["summary"] = str(out_dir / "summary.md")
    return report
