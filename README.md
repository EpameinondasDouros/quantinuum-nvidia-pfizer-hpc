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

Systems (`genqai/chemistry.py`): `h2` (2 qubits), `lih4` (4), `lih6` (6), `h4` (8, linear chain a-b-a).

## Integration contracts

See `genqai/schema.py`. Qiskit qubit order (qubit 0 rightmost), total energies in Hartree,
operator IDs index `genqai.pool.get_pool(system)`, circuit = HF then `exp(-i θ_t G_{op_t})` in order.
