# Qollab runtime checklist (for the judge-facing notebook)

`notebooks/genqai_demo.ipynb` must run top to bottom with *Run All* on Qollab. Its setup cell finds or
clones the repo and installs missing packages, and it degrades gracefully when PySCF or PyTorch can't be
installed. A human logged into Qollab still has to check the items below once, before the deadline.

## 1. Things to verify on Qollab

| # | Check | How | Expected / fallback |
|---|---|---|---|
| 1 | Python version | `import sys; sys.version` | >= 3.11 (pyproject). On 3.10 the code may still run, but this is untested |
| 2 | Qiskit version | `import qiskit; qiskit.__version__` | >= 2.0. qiskit-nature 0.8 must import cleanly next to it (`from qiskit_nature.second_q.circuit.library import UCCSD`) |
| 3 | `pip install` allowed in the kernel | `%pip install -q pylatexenc` | If it is blocked, everything in the REQUIRED list of the setup cell must already be installed |
| 4 | Outbound git/HTTPS | setup cell clones `https://github.com/EpameinondasDouros/quantinuum-nvidia-pfizer-hpc` when the notebook is opened on its own | If cloning is blocked, upload the repo folder (or a zip) and open the notebook from `notebooks/` |
| 5 | PyTorch present or installable | `import torch` | Needed for live generation (CPU is enough, < 0.1 s per circuit). **Without it** the notebook replays `results/<system>/gen_oneshot.jsonl` for grid geometries only, labelled "REPLAY of precomputed circuit, not live generation" |
| 6 | PySCF installable | `%pip install pyscf` (wheel is ~50 MB; no compiler needed on Linux x86_64) | Needed for **off-grid** (truly unseen) geometries. **Without it** the notebook loads the stored Hamiltonian from `data/<system>/hamiltonians.jsonl` and snaps the requested geometry to the nearest grid point, saying so in the output. Pick a `test_interval` / `test_extrap` grid point so it's still unseen by training |
| 7 | IonQ credentials | Is an IonQ key pre-provisioned (env var), or do judges paste their own? | The notebook reads `IONQ_API_KEY` (also `QISKIT_IONQ_API_TOKEN`, `IONQ_API_TOKEN`) from the environment or a `.env` in the repo root. To set it inline: `os.environ["IONQ_API_KEY"] = "..."`, then re-run the setup cell. Never commit it |
| 8 | IonQ job/shot limits | Per-job shot cap, jobs per hour, free-tier quota | Defaults: H2, 4000 shots per backend; 3 IonQ jobs (ideal, aria-1, forte-1), each one job that holds all measurement groups. LiH (4q) has 25 groups, so its per-group shots are tiny unless `SHOTS` is raised. Only do that if quota allows |
| 9 | IonQ noise models available | `ionq_noisy` with `aria-1`, `forte-1` | If a name is rejected, the failing backend is reported and skipped (the others still run). Edit `IONQ_NOISE_MODELS` in step 1 |
| 10 | Matplotlib inline + circuit drawing | step 2 draws with `qc.draw("mpl")` | Needs `pylatexenc` (installed by setup). Falls back to the text drawer |
| 11 | Wall-clock | Run All with defaults (H2) | About 1 min locally plus IonQ queue time. `lih4`: about 1 min. `h4` with `RUN_ADAPT = True`: several minutes (set it to False) |

## 2. How to run the notebook on Qollab

1. Upload `notebooks/genqai_demo.ipynb`, or clone the repo in a Qollab terminal and open the notebook from
   `notebooks/`.
2. (Optional) Set the IonQ key: `IONQ_API_KEY` as a Qollab secret/env var, or `os.environ[...]` in a cell.
3. *Kernel → Restart & Run All*.
4. To try another molecule, edit the parameters cell of step 1 (`SYSTEM`, `GEOMETRY`) and re-run from there.
   Geometries: `h2` r ~ 0.5–2.5 Å, `lih4`/`lih6` r ~ 1.0–4.0 Å, `h4` (a, b) ~ 0.8–1.6 / 0.8–2.0 Å.
   Values outside the training range are extrapolation, and the error will grow.

## 3. Terminal equivalent

```bash
PYTHONPATH=$PWD python scripts/demo.py --system h2 --geometry 1.33 --ionq
PYTHONPATH=$PWD python scripts/demo.py --system lih4 --geometry 2.47 --shots 20000
```
