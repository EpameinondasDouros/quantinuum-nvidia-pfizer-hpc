# Qollab runtime facts (verified from qollab.xyz docs, Oct 10 2026)

Source pages: qollab.xyz/learn/docs/{runtime-environment, python-qiskit-projects, compute-backends, run-your-code}.
These replace the earlier guesses. They change the judge-facing design: Qollab does **not** run Jupyter notebooks.

## What the runtime is

| Item | Fact | Consequence for GenQAI |
| --- | --- | --- |
| Execution | Python 3.14 on **Pyodide (WebAssembly) in the browser**. No server, no terminal, no git clone | The demo is a Qollab *Python project*: a top-level `main.py` plus our package files. `notebooks/genqai_demo.ipynb` stays as the local/offline demo only |
| Entry point | `main.py` runs top to bottom. If it defines `def main(...)` on one line with every parameter annotated `int`/`float`/`str`/`bool`, Run shows a form for those parameters and the runtime calls `main()` | Expose `system: str`, `r: float`, `a: float`, `b: float`, `shots: int` (and maybe `method: str`) as form fields. Don't call `main()` ourselves |
| Qiskit | Always installed: `qiskit>=2.5.2,<3` unless requirements.txt pins it | Matches our 2.5.2 dev env |
| Chemistry | "Qiskit Nature (+ PySCF)" library adds `qiskit-nature==0.8.0` and `pyscf>=2.14.0,<3` | Live Hamiltonians for **any** geometry work in the browser. Still to confirm with a real run: PySCF actually loads under Pyodide, and how long it takes |
| Other packages | `qiskit-algorithms==0.4.0` available. NumPy/SciPy/Matplotlib come with Pyodide. Pure-Python wheels install from requirements.txt. No pip options, no URLs | **PyTorch is not available** (no Pyodide build). The transformer needs a NumPy inference path with exported weights. `qiskit-ionq`/`qiskit-aer` aren't needed: the backend is injected |
| Quantum backend | A global `backend` is injected, chosen by whoever presses Run in the *Select QPU* dialog: built-in simulator (Qiskit BasicSimulator), AWS Braket local, IBM noise-model fakes, IonQ remote simulators (Aria 1/2, Forte 1, Forte Enterprise 1 noise models; free), IonQ hardware (credits) | The code must never construct a backend or use an API key. Measurement code must accept whatever `backend` it is given and parse that provider's result |
| IonQ jobs | **One job per run** on IonQ backends, simulators included. A second `backend.run` is an error. A list of circuits in one `backend.run` call is allowed where the provider supports it (IonQ does) | Submit all qubit-wise-commuting measurement groups in a single `backend.run(list, shots=...)`. Ideal vs noisy is a choice in the dialog, so judges run twice |
| Shots | Set in code (`backend.run(..., shots=n)`); a `shots: int` main() parameter becomes a form field. Default 1024 | IonQ noise-model sims cost time per shot. Keep the default modest (h2: 2 groups × 1000) |
| Files | ≤ 100 files per project, ≤ 2 MB each, 200 MB across all projects. Data files load only if the path appears as a **literal string** in Python code (no f-strings, joins, or `+`) | Export SFT weights split or as float16 to stay under 2 MB per file. Reference every data file by a literal path |
| Network | GET only (pyfetch), CORS required | Don't depend on GitHub at run time |
| Output | stdout/stderr in the console; `plt.savefig("x.png")` shows the figure and offers a download; `js.sendFile` for other files | Print a comparison table; save an energy-vs-method figure |
| Persistence | None between runs | Ship precomputed results (teacher labels, baseline tables) as files |
| Browser | Needs WebAssembly JSPI (recent Chrome/Edge) | Note this for judges |

## Still to verify with a real project run (needs a Qollab project in the account)

1. PySCF imports under Pyodide, and `build_record` time for h2 / lih4 / h4.
2. `qiskit_nature` UCCSD pool construction works (`genqai.pool`).
3. NumPy transformer inference matches the PyTorch checkpoint (also checked offline in tests).
4. `backend.run(list_of_circuits, shots=n)` on IonQ Aria 1 / Forte 1 returns one result with per-circuit counts, and our parser handles it. Also the built-in BasicSimulator.
5. End-to-end wall-clock from pressing Run.
