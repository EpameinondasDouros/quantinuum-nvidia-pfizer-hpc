"""Build ONE 12-qubit Hamiltonian from the paper's imipramine geometry and draw the molecule."""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hamgen.hamiltonian import build_hamiltonian, DATA

atoms = read(DATA / "imipramine.xyz")
r = build_hamiltonian(atoms, 12)
terms = sorted(r["terms"].items(), key=lambda kv: -abs(kv[1]))
def pstr(t): 
    d = dict(t); return "".join(d.get(q, "I") for q in range(12))
out = {"n_qubits": 12, "n_terms": len(terms), "e_hf_Ha": r["e_hf"], "e_casci_Ha": r["e_casci"],
       "correlation_energy_mHa": 1000 * (r["e_casci"] - r["e_hf"]),
       "terms": [{"pauli": pstr(t), "coeff": c} for t, c in terms]}
json.dump(out, open(DATA / "example_hamiltonian_12q.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "terms"})
print("largest 8 terms:"); [print(f"  {x['pauli']}  {x['coeff']:+.6f}") for x in out["terms"][:8]]

# --- 3D picture (CPK colours, bonds where atoms are close)
col = {"C": "#404040", "H": "#e8e8e8", "N": "#3050f8"}; size = {"C": 260, "H": 90, "N": 280}
P, S = atoms.positions, atoms.get_chemical_symbols()
fig = plt.figure(figsize=(12, 5))
for k, (el, az) in enumerate([(20, 30), (80, 30)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if np.linalg.norm(P[i] - P[j]) < (1.25 if "H" in (S[i], S[j]) else 1.75):
                ax.plot(*zip(P[i], P[j]), color="#999", lw=2, zorder=1)
    for e in col:
        m = [i for i, s in enumerate(S) if s == e]
        ax.scatter(*P[m].T, s=size[e], c=col[e], edgecolors="k", depthshade=True, label=e)
    ax.set_axis_off(); ax.view_init(el, az); ax.set_box_aspect(np.ptp(P, axis=0))
ax.legend(loc="lower right")
fig.suptitle("Imipramine (C19H24N2), geometry from the paper: C grey, H white, N blue")
plt.tight_layout(); plt.savefig(DATA / "imipramine_3d.png", dpi=130)
