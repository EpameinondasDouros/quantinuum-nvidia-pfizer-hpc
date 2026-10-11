"""Render site/cover.png, site/cover_square.png, site/og.png and site/circuit.svg from real project data.

LiH (4 qubits): FCI potential-energy curve from data/lih4/hamiltonians.jsonl, the transformer's one-shot
energies on test geometries from results/lih4/sft_oneshot.jsonl, and the one-shot generated circuit for a test
geometry (results/lih4/gen_oneshot.jsonl) drawn wire by wire from the operator pool's Pauli support.

    .venv/bin/python -m scripts.make_site_cover
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from genqai.pool import get_pool  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SYSTEM = "lih4"

# the site's plate palette: five values, never a sixth hue (site/styles.css)
PAPER, INK, COBALT, SIGNAL = "#efedea", "#0a0a0a", "#1e2edc", "#e0492a"
BG = PAPER
MUTED = "#0a0a0a99"  # ink at 60%, as the site's meta text
WIRE = "#0a0a0a59"  # ink at 35%
CURVE = COBALT
POINT = SIGNAL
GATE = PAPER
GATE_EDGE = INK
XGATE = SIGNAL

SERIF = ["Georgia", "Times New Roman", "DejaVu Serif"]  # stands in for Instrument Serif (not installed for matplotlib)
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_data():
    hams = _jsonl(ROOT / "data" / SYSTEM / "hamiltonians.jsonl")
    by_id = {h["geometry_id"]: h for h in hams}
    curve = sorted((h["geometry_params"]["r"], h["fci_energy"]) for h in hams)
    gen = [s for s in _jsonl(ROOT / "results" / SYSTEM / "sft_oneshot.jsonl") if s["split"].startswith("test")]
    points = sorted((by_id[s["geometry_id"]]["geometry_params"]["r"], s["energy"]) for s in gen)
    # the transformer's own one-shot output on the held-out geometry closest to equilibrium
    r_min = min(curve, key=lambda c: c[1])[0]
    gens = [g for g in _jsonl(ROOT / "results" / SYSTEM / "gen_oneshot.jsonl") if g["meta"]["split"].startswith("test")]
    circ = min(gens, key=lambda g: abs(by_id[g["geometry_id"]]["geometry_params"]["r"] - r_min))
    eq = by_id[circ["geometry_id"]]
    return curve, points, eq, circ


def draw_circuit(ax, record: dict, circ: dict, *, wire=WIRE, label=MUTED, text=INK, fs=13):
    n = record["num_qubits"]
    pool = get_pool(SYSTEM)
    ops, angles = circ["operators"], circ["angles"]
    ncols = len(ops) + 1
    ax.set_xlim(-0.9, ncols + 0.2)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")
    for q in range(n):
        ax.plot([-0.4, ncols - 0.1], [q, q], color=wire, lw=1.6, zorder=1, solid_capstyle="round")
        ax.text(-0.55, q, f"q{q}", color=label, ha="right", va="center", fontsize=fs, family="monospace")
    # Hartree-Fock reference: X on occupied qubits (qiskit little-endian, as genqai.circuits.build_circuit)
    for q, b in enumerate(reversed(record["hf_state"])):
        if b == "1":
            ax.add_patch(FancyBboxPatch((-0.02 - 0.2, q - 0.24), 0.44, 0.48, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=XGATE, ec="none", zorder=3))
            ax.text(0, q, "X", color=PAPER, ha="center", va="center", fontsize=fs + 1, weight="bold", zorder=4)
    for col, (k, theta) in enumerate(zip(ops, angles), start=1):
        support = sorted({q for p in pool.generators[k].paulis for q in range(n) if str(p)[n - 1 - q] != "I"})
        lo, hi = min(support), max(support)
        ax.add_patch(FancyBboxPatch((col - 0.36, lo - 0.3), 0.72, hi - lo + 0.6,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=GATE, ec=GATE_EDGE, lw=1.3, zorder=3))
        ax.text(col, (lo + hi) / 2, f"G{k}", color=text, ha="center", va="center", fontsize=fs, weight="bold", zorder=4)
        ax.text(col, -0.62, f"{theta:+.3f}", color=label, ha="center", va="center", fontsize=fs - 3,
                family="monospace", zorder=4)


def draw_pes(ax, curve, points, *, fs=13):
    r, e = zip(*curve)
    ax.plot(r, e, color=CURVE, lw=2.4, zorder=2, label="Exact (FCI)")
    pr, pe = zip(*points)
    ax.scatter(pr, pe, s=58, color=POINT, edgecolor=BG, linewidth=1.6, zorder=3,
               label="Transformer, one shot (test geometries)")
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(WIRE)
    ax.tick_params(colors=MUTED, labelsize=fs - 2)
    ax.set_xlabel("Li–H bond length (Å)", color=MUTED, fontsize=fs)
    ax.set_ylabel("Energy (Ha)", color=MUTED, fontsize=fs)
    leg = ax.legend(frameon=False, fontsize=fs - 1, loc="lower right", labelcolor=INK)
    return leg


def title_block(fig, x, y, *, big, small, gap, tagline="A trained model writes the whole circuit in one forward pass: zero quantum evaluations to propose."):
    fig.text(x, y, "GenQAI", color=INK, fontsize=big, family=SERIF, ha="left", va="top")
    fig.text(x, y - gap, "One-shot quantum chemistry circuits", color=INK, fontsize=small, family=SERIF, ha="left", va="top")
    fig.text(x, y - gap * 1.55, tagline,
             color=MUTED, fontsize=small * 0.62, ha="left", va="top")


def cover(curve, points, eq, circ):
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    title_block(fig, 0.05, 0.93, big=92, small=34, gap=0.15)
    ax = fig.add_axes([0.09, 0.17, 0.40, 0.44])
    draw_pes(ax, curve, points, fs=15)
    cax = fig.add_axes([0.53, 0.12, 0.45, 0.50], facecolor=BG)
    draw_circuit(cax, eq, circ, fs=17)
    fig.text(0.545, 0.63, f"Generated LiH circuit, r = {eq['geometry_params']['r']:.2f} Å · 4 qubits · 8 operators",
             color=MUTED, fontsize=16, ha="left", va="bottom")
    fig.text(0.05, 0.035, "Qollab × IonQ Global Quantum Hackathon 2026 · MIT licence", color=MUTED, fontsize=14)
    fig.savefig(SITE / "cover.png", facecolor=BG)
    plt.close(fig)


def cover_square(curve, points, eq, circ):
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=BG)
    title_block(fig, 0.07, 0.95, big=104, small=36, gap=0.135,
                tagline="A trained model writes the whole circuit in one forward pass:\nzero quantum evaluations to propose.")
    ax = fig.add_axes([0.13, 0.40, 0.82, 0.245])
    draw_pes(ax, curve, points, fs=17)
    cax = fig.add_axes([0.05, 0.03, 0.92, 0.27], facecolor=BG)
    draw_circuit(cax, eq, circ, fs=19)
    fig.text(0.07, 0.305, f"Generated LiH circuit, r = {eq['geometry_params']['r']:.2f} Å", color=MUTED, fontsize=17)
    fig.savefig(SITE / "cover_square.png", facecolor=BG)
    plt.close(fig)


def circuit_svg(eq, circ):
    """Paper figure: the same circuit on a transparent background, in the plate palette."""
    fig = plt.figure(figsize=(9, 4.2))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.patch.set_alpha(0)
    draw_circuit(ax, eq, circ, fs=14)
    fig.savefig(SITE / "circuit.svg", transparent=True)
    plt.close(fig)


def main():
    SITE.mkdir(exist_ok=True)
    curve, points, eq, circ = load_data()
    cover(curve, points, eq, circ)
    cover_square(curve, points, eq, circ)
    circuit_svg(eq, circ)
    shutil.copyfile(SITE / "cover.png", SITE / "og.png")
    print(f"drawn circuit: test geometry {eq['geometry_id']} r={eq['geometry_params']['r']}, ops={circ['operators']}")


if __name__ == "__main__":
    main()
