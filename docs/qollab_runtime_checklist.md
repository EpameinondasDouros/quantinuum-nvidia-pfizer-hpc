# Qollab runtime facts (verified from qollab.xyz docs, Oct 10 2026)

Source pages: qollab.xyz/learn/docs/{runtime-environment, python-qiskit-projects, compute-backends, run-your-code}.
These replace the earlier guesses. They change the judge-facing design: Qollab does **not** run Jupyter notebooks.

## What the runtime is

| Item | Fact | Consequence for GenQAI |
| --- | --- | --- |
| Execution | Python 3.14 on **Pyodide (WebAssembly) in the browser**. No server, no terminal, no git clone | The demo is a Qollab *Python project*: a top-level `main.py` plus our package files. `notebooks/genqai_demo.ipynb` stays as the local/offline demo only |
| Entry point | `main.py` runs top to bottom. If it defines `def main(...)` on one line with every parameter annotated `int`/`float`/`str`/`bool`, Run shows a form for those parameters and the runtime calls `main()` | Expose `system: str`, `r: float`, `a: float`, `b: float`, `shots: int` (and maybe `method: str`) as form fields. Don't call `main()` ourselves |
| Qiskit | Always installed: `qiskit>=2.5.2,<3` unless requirements.txt pins it | Matches our 2.5.2 dev env |
| Chemistry | "Qiskit Nature (+ PySCF)" library adds `qiskit-nature==0.8.0` and `pyscf>=2.14.0,<3` | Live Hamiltonians for **any** geometry work in the browser. **Confirmed:** Qollab's own `showcase-h2-ground-state` project runs `PySCFDriver(...).run()` in the browser (WebAssembly PySCF). Load time still unmeasured |
| Other packages | `qiskit-algorithms==0.4.0` available. NumPy/SciPy/Matplotlib come with Pyodide. Pure-Python wheels install from requirements.txt. No pip options, no URLs | **PyTorch is not available** (no Pyodide build). The transformer needs a NumPy inference path with exported weights. `qiskit-ionq`/`qiskit-aer` aren't needed: the backend is injected |
| Quantum backend | A global `backend` is injected, chosen by whoever presses Run in the *Select QPU* dialog: built-in simulator (Qiskit BasicSimulator), AWS Braket local, IBM noise-model fakes, IonQ remote simulators (Aria 1/2, Forte 1, Forte Enterprise 1 noise models; free), IonQ hardware (credits) | The code must never construct a backend or use an API key. Measurement code must accept whatever `backend` it is given and parse that provider's result |
| IonQ jobs | **One job per run** on IonQ backends, simulators included. A second `backend.run` is an error. A list of circuits in one `backend.run` call is allowed where the provider supports it (IonQ does) | Submit all qubit-wise-commuting measurement groups in a single `backend.run(list, shots=...)`. Ideal vs noisy is a choice in the dialog, so judges run twice |
| Shots | Set in code (`backend.run(..., shots=n)`); a `shots: int` main() parameter becomes a form field. Default 1024 | IonQ noise-model sims cost time per shot. Keep the default modest (h2: 2 groups × 1000) |
| Files | ≤ 100 files per project, ≤ 2 MB each, 200 MB across all projects. Data files load only if the path appears as a **literal string** in Python code (no f-strings, joins, or `+`) | Export SFT weights split or as float16 to stay under 2 MB per file. Reference every data file by a literal path |
| Network | GET only (pyfetch), CORS required | Don't depend on GitHub at run time |
| Output | stdout/stderr in the console; `plt.savefig("x.png")` shows the figure and offers a download; `js.sendFile` for other files | Print a comparison table; save an energy-vs-method figure |
| Persistence | None between runs | Ship precomputed results (teacher labels, baseline tables) as files |
| Browser | Needs WebAssembly JSPI (recent Chrome/Edge) | Note this for judges |

## Verified by running our project on Qollab (Oct 10, 2026)

Project (draft, unpublished): https://qollab.xyz/u/udsy/genqai-one-shot-quantum-chemis. It was created by uploading three .zip archives; the browser upload tool caps a call at 10 MB, so weights went in separate archives with "Keep and add".

| Check | Result |
| --- | --- |
| Runtime | Python 3.14.2 (Pyodide), numpy 2.4.6, scipy 1.18.0, pyscf 2.14.0, qiskit-nature 0.8.0, scikit-learn 1.8.0 (installed from requirements.txt) |
| PySCF in WebAssembly | `gto`, `scf`, `ao2mo` work (LiH RHF in 0.06 s, matches native to 1e-14). **`fci`, `mcscf`, `cc` fail** (`OSError: no file with expected extension`: no libfci/libmcscf/libcc). `mol.intor("int2e")` (s1) **crashes the runtime**. We therefore build the active-space Hamiltonian and FCI from RHF + `ao2mo.incore` (`genqai/chemistry.py`), which is identical to CASCI to 1e-14 natively. CCSD shows "n/a" in the browser |
| IonQ gateway | **Accepts only single-circuit jobs** (`Only job type ionq.circuit.v1 is supported`; a multi-circuit list gets HTTP 400), and one job per run. We pack all QWC groups into ONE circuit as copies of the state on disjoint qubits (`MeasurementPlan.packed_circuit`). h2 = 2 groups × 2 qubits = 4 qubits. Systems whose groups × qubits > 29 can't be measured on IonQ in one run; the code says so |
| IonQ remote simulators | Noise models only (Aria 1/2, Forte 1, Forte Enterprise 1), free. There is no ideal IonQ sim in the dialog. Hardware entries say "Credits apply" |
| Run form | `def main(system: str, r: float, a: float, b: float, shots: int, live_adapt: bool)` shows as a form |
| End to end, h2 r = 1.33 Å, built-in simulator | 6.3 s total. Live PySCF 0.3 s, NumPy transformer 20 ms (error 0.006 mHa), measured +6.6 ± 5.8 mHa |
| End to end, h2 r = 1.33 Å, IonQ Forte 1 sim | 18.2 s total (IonQ 11.8 s). Measured **+7.8 ± 5.8 mHa** vs FCI, job `01a12826-5478-72af-b517-c3e60f663d79`. Qollab also shows the compiled native circuit and the probabilities |

## Context from Qollab's own chemistry article (qollab.xyz/learn/quantum-chemistry)

- Other community projects do textbook VQE: qOrbital (UCCSD-VQE with an orbital visualizer) and Quantum Advantage Lab (a hardware-efficient RXX ansatz). qOrbital's LiH UCCSD-VQE needed 1,035 evaluations and about 1.5 h on a laptop. **Our pitch on Qollab: the circuit and its angles come from one model call, so no optimizer loop runs in the browser.**
- Their teaching listings send one IonQ job per measurement basis, so a run needs several presses. We send all qubit-wise-commuting groups in one multi-circuit job, which IonQProvider supports. That is worth pointing out to judges.
- Their H₂ noise table: 3 gates at 0.1% / 1% depolarizing already costs +9 mHa. This matches our finding that gate count is the hardware lever.
