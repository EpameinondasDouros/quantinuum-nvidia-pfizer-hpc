import json
from pathlib import Path

from genqai.figures import make_all

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_make_all_writes_figures(tmp_path):
    report = make_all(RESULTS, tmp_path)
    for name in ("fig1_pes", "fig2_test_errors", "fig3_accuracy_cost", "fig5_breakeven", "fig6_ionq"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000, report[name]
        assert (tmp_path / f"{name}.pdf").exists()
    assert "| adapt |" in (tmp_path / "summary.md").read_text()
    be = json.loads((tmp_path / "breakeven.json").read_text())
    assert be["lih4"]["evals"]["methods"]["adapt"]["per_geometry"] > 0


def test_generator_methods_are_discovered(tmp_path):
    """Pipeline plumbing for generator results (records here are relabelled ADAPT, test-only)."""
    import shutil

    res = tmp_path / "results"
    shutil.copytree(RESULTS, res)
    lines = (res / "lih4" / "adapt.jsonl").read_text().splitlines()
    (res / "lih4" / "sft_oneshot.jsonl").write_text("\n".join(
        json.dumps({**json.loads(x), "method": "sft_oneshot", "energy_evals": 0, "pool_gradient_evals": 0})
        for x in lines))
    (res / "lih4" / "gen_oneshot.jsonl").write_text(json.dumps({"geometry_id": "x", "operators": []}) + "\n")
    (res / "lih4" / "sft_train_log.json").write_text(json.dumps(
        {"train_time_s": 10.0, "evals": [{"epoch": 1, "train": [1.0, 0.5, 0.5], "val_mean_abs_mha": 3.0}]}))
    report = make_all(res, tmp_path / "fig")
    assert report["fig4_training"].endswith(".png")
    be = json.loads((tmp_path / "fig" / "breakeven.json").read_text())["lih4"]
    assert be["wall"]["methods"]["sft_oneshot"]["fixed_note"].endswith("model training time")
    assert "| sft_oneshot |" in (tmp_path / "fig" / "summary.md").read_text()
