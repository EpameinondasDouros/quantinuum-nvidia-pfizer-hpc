"""Streamlit workbench smoke tests (streamlit.testing.v1.AppTest). Lane-dependent steps skip with a reason."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app import services  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "workbench.py")


def _ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture(scope="module")
def built():
    """App with the h2 preset (default 0.74 A) built: the Molecule tab's preset path."""
    at = _ok(AppTest.from_file(APP, default_timeout=180).run())
    at.button(key="build_btn").click().run()
    return _ok(at)


def test_app_loads_with_environment_panel():
    at = _ok(AppTest.from_file(APP, default_timeout=120).run())
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert "PySCF" in side and "Lane A" in side
    assert [t.label for t in at.tabs][:6] == ["Molecule", "Generate", "Grade", "Execute", "Benchmarks",
                                             "History & export"]


def test_preset_builds_h2(built):
    msg = " ".join(s.value for s in built.success)
    assert "Current Hamiltonian: h2" in msg and "2 qubits" in msg
    caps = next(df.value for df in built.dataframe if "capability" in df.value.columns)
    assert caps.set_index("capability").loc["Hamiltonian built", "state"] == "yes"


def test_generate_and_grade_on_preset(built):
    at = built
    at.button(key="gen_btn").click().run()
    _ok(at)
    statement = " ".join(m.value for m in at.markdown if "to *propose*" in m.value)
    assert "Building the Hamiltonian" in statement and "separate costs" in statement
    assert any("excitation" in df.value.columns for df in at.dataframe)
    at.button(key="grade_btn").click().run()
    _ok(at)
    assert any(m.label.startswith("Error (mHa)") for m in at.metric)


def test_execute_dry_run(built):
    if services.lane("execution") is None:
        pytest.skip("genqai.execution (Lane D) not in this build")
    at = built
    if not at.session_state["circuits"]:
        at.button(key="gen_btn").click().run()
    at.button(key="exec_dry").click().run()
    _ok(at)
    df = next(df.value for df in at.dataframe if "job_id" in df.value.columns)
    assert "dry_run" in set(df["status"])


def test_smiles_input_renders():
    at = _ok(AppTest.from_file(APP, default_timeout=120).run())
    at.radio(key="mol_source").set_value("SMILES").run()
    _ok(at)
    if services.lane("molecule") is None:
        assert any("not available in this build" in i.value for i in at.info)
        pytest.skip("genqai.molecule (Lane A) not in this build: SMILES panel shows 'not available'")
    at.text_input(key="smiles").set_value("[H][H]").run()
    at.button(key="smiles_go").click().run()
    _ok(at)


def test_benchmarks_tab_shows_committed_results():
    at = _ok(AppTest.from_file(APP, default_timeout=120).run())
    assert any("committed results" in c.value for c in at.caption)
    df = next(df.value for df in at.dataframe if "median |err| mHa" in df.value.columns)
    assert "sft_oneshot" in set(df["method"])
