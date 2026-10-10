# GenQAI: Hardware-Aware Generative Quantum Circuit Synthesis

Qollab × IonQ Global Quantum Hackathon (Oct 10–11, 2026). A hackathon-scale reproduction of the
Quantinuum–NVIDIA–Pfizer ADAPT-GQE architecture: a Hamiltonian-conditioned transformer that emits
a complete operator-and-angle circuit for an unseen molecular geometry, trained by supervised
imitation of ADAPT-VQE, then GRPO and self-distillation, executed with Qiskit and IonQ's simulator.

## Setup

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

## Pipeline (so far)

| Stage | Command | Output |
| --- | --- | --- |
| Hamiltonians + HF/CCSD/FCI | `python scripts/make_hamiltonians.py` | `data/<system>/hamiltonians.jsonl` |
| ADAPT-VQE teacher | `python scripts/make_teacher.py` | `data/<system>/teacher.jsonl` |
| Canonical, gauge-fixed, smoothed SFT labels | `python scripts/make_canonical_teacher.py --system <s>` | `data/<system>/teacher_canonical.jsonl` |
| SFT of the generative transformer | `python scripts/train_sft.py --system <s>` | `checkpoints/<system>/sft.pt`, `results/<system>/sft_train_log.json` |
| One-shot / refine / best-of-K generation | `python scripts/generate.py --system <s>` | `results/<system>/gen_<mode>.jsonl`, `results/<system>/sft_<mode>.jsonl` |
| Measured energy (local + IonQ sim) | `python scripts/run_ionq_demo.py --system h2 --shots 4000` | `results/<system>/ionq_demo_<id>.json` |

Systems (`genqai/chemistry.py`): `h2` (2 qubits), `lih4` (4), `lih6` (6), `h4` (8, linear chain a-b-a).

## Integration contracts

See `genqai/schema.py`. Qiskit qubit order (qubit 0 rightmost), total energies in Hartree,
operator IDs index `genqai.pool.get_pool(system)`, circuit = HF then `exp(-i θ_t G_{op_t})` in order.

## IonQ

`genqai/measure.py` groups Pauli terms qubit-wise, measures each group in its product basis and
reports energy ± stderr. IonQ backends need `IONQ_API_KEY` (environment or a `.env` in the repo
root) and `qiskit-ionq`; without it the demo runs only the local backends.
