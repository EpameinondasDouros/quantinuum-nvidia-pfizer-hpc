"""Regenerate every paper figure, figures/breakeven.json and figures/summary.md from results/.

    PYTHONPATH=$PWD python -m scripts.make_figures [--results results] [--out figures]
"""
import argparse
from pathlib import Path

from genqai.figures import make_all

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=ROOT / "results", type=Path)
ap.add_argument("--out", default=ROOT / "figures", type=Path)
a = ap.parse_args()
for name, status in make_all(a.results, a.out).items():
    print(f"{name:20s} {status}")
