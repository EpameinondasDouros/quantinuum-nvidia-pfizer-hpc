# GenQAI on Qollab

A transformer conditioned on a molecule's qubit Hamiltonian writes the **whole VQE circuit at once**: which
fermionic excitations to use, in what order, and every rotation angle. It's a reproduction of the
Quantinuum / NVIDIA / Pfizer ADAPT-GQE idea. For a geometry the model never saw, the circuit comes from one
forward pass with **zero** energy evaluations. ADAPT-VQE needs hundreds of quantum measurements per geometry
to build a circuit of the same quality. The model runs in NumPy, so no PyTorch is needed in the browser.

## Run it
1. Upload this whole folder as a Python project (entry `main.py`, 48 files, all under 2 MB).
2. Project libraries: enable **"Qiskit Nature (+ PySCF)"**. `requirements.txt` adds scikit-learn.
3. Press Run and pick a backend in *Select QPU*. The built-in simulator is fastest. **IonQ Aria / Forte
   simulators** work too: every measurement group goes into **one** IonQ job.
4. Fill in the form. `system` is `h2` (2 qubits), `lih4` (4), `lih6` (6) or `h4` (8, uses `a`, `b`).
   `r` is the bond length in Å. `shots` is the shot count per measurement group. Set `live_adapt` to
   compare against a fresh ADAPT-VQE run.

Expected runtime (native timings, so expect several times longer in the browser): h2 and lih4 take
about 2 s plus the backend queue. The IonQ simulator adds roughly 5–30 s. lih6 and h4 have 46–48 groups;
use them with the built-in simulator, not with IonQ noise models.

## What the output means
1. **Hamiltonian**: built live with PySCF for your geometry. It prints HF / CCSD / FCI, says whether the
   geometry is inside the training range, and gives the distance to the nearest training geometry.
   Without PySCF, the run falls back to the nearest stored geometry and says so.
2. **One-shot generation**: the operator sequence and angles, CX count and depth, and the model time in ms.
3. **Exact scoring**: statevector energies vs FCI. The transformer is `sft_oneshot`. The baselines, which
   use training labels only, are `nn_reuse_gauge` (copy the nearest training circuit), `angle_regression`
   (kernel ridge) and `nn_refine_canon` (copy + optimise). `adapt` is fresh ADAPT-VQE; its
   `pool grads` + `energy evals` columns are the quantum cost the transformer skips. A summary of the
   precomputed held-out results follows.
4. **Quantum execution**: the generated circuit is measured on the selected backend in qubit-wise-commuting
   groups, all in one `backend.run`. It prints the energy ± standard error, the error vs FCI, the backend
   name and the job id. On IonQ, circuits are compiled to native gates (GPI/GPI2 + MS/ZZ).
5. **Figure** `genqai_energies.png`: E − E_FCI per method (symlog scale), with the chemical-accuracy band
   and the measured energy's error bar.

Rebuild from the repo with `PYTHONPATH=$PWD python scripts/build_qollab_bundle.py`. Only `main.py` and
this README are hand-written.
