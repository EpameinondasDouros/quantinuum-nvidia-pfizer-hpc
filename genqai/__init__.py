"""GenQAI: molecular quantum circuit generation and grading workbench."""
import importlib.util

# Load PyTorch's OpenMP runtime before any other native library initialises its own (PySCF, Qiskit
# simulators). The reverse order segfaults on macOS the first time a torch model is used after a
# PySCF/Qiskit computation (reproduced: execute(..., "statevector") then generate(..., "transformer_sft")).
# No-op where torch isn't installed (Qollab/Pyodide), so the browser bundle is unaffected.
if importlib.util.find_spec("torch") is not None:
    import torch  # noqa: F401
