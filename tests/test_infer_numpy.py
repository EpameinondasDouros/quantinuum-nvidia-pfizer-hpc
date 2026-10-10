"""genqai.infer_numpy (torch-free one-shot decoding for the Qollab bundle) == GQEPolicy.sample(greedy=True)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from genqai.data import FRESH_SPLIT, load_hamiltonians  # noqa: E402
from genqai.demo import ROOT  # noqa: E402
from genqai.infer_numpy import PART_BYTES, NumpyPolicy, export_numpy  # noqa: E402
from genqai.model import GQEPolicy  # noqa: E402
from genqai.sim import Evaluator  # noqa: E402

SYSTEMS = [s for s in ("h2", "lih4", "h4", "lih6") if (ROOT / "checkpoints" / s / "sft.pt").exists()]


@pytest.mark.parametrize("system", SYSTEMS)
def test_numpy_oneshot_matches_torch(system, tmp_path):
    ckpt = ROOT / "checkpoints" / system / "sft.pt"
    written = export_numpy(ckpt, tmp_path / "sft.json")
    assert all(p.stat().st_size <= PART_BYTES for p in written)  # Qollab: <= 2 MB per file
    np_policy, torch_policy = NumpyPolicy.load(tmp_path / "sft.json"), GQEPolicy.load(ckpt)
    records = load_hamiltonians(system, "test") + load_hamiltonians(system, FRESH_SPLIT)
    for rec in records:
        ops_t, ang_t, _, valid_t = torch_policy.sample(torch_policy.condition([rec]), greedy=True)[0]
        ops_n, ang_n, valid_n = np_policy.generate(rec)
        assert ops_n == ops_t and valid_n == valid_t, rec.geometry_id
        d = np.angle(np.exp(1j * (np.array(ang_n) - np.array(ang_t))))  # periodic difference
        assert np.abs(d).max(initial=0.0) < 1e-4, rec.geometry_id
        ev = Evaluator(rec)
        assert abs(ev.energy(ops_n, ang_n) - ev.energy(ops_t, ang_t)) * 1000 < 0.01, rec.geometry_id


def test_qollab_bundle_in_sync():
    """qollab/genqai is a verbatim copy (re-run scripts/build_qollab_bundle.py after editing genqai/)."""
    bundle = ROOT / "qollab" / "genqai"
    if not bundle.exists():
        pytest.skip("no qollab/ bundle")
    stale = [p.name for p in bundle.glob("*.py") if p.read_text() != (ROOT / "genqai" / p.name).read_text()]
    assert not stale, f"rebuild the bundle: {stale}"
