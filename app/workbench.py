"""GenQAI molecular circuit workbench.  Run:  streamlit run app/workbench.py

Molecule -> Hamiltonian -> routing -> circuit generation -> grading -> optimization -> execution -> export.
Every panel says where its numbers come from: a live computation in this session, a committed results
file, or "not available in this build" (a lane module that is not installed). Nothing is mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from genqai.jobs import JobRunner  # noqa: E402  (stdlib only; must not import qiskit)


def _warm_up(ctx) -> None:
    """Import the compiled scientific stack on the runner's long-lived worker thread.

    Qiskit's Rust extension segfaults later if it is first imported on a Streamlit script thread, which
    dies at the end of each run. Importing it here first, on a thread that lives as long as the server,
    avoids that.
    """
    from qiskit import QuantumCircuit

    import genqai.circuits  # noqa: F401
    import genqai.export  # noqa: F401

    QuantumCircuit(1)


@st.cache_resource
def get_runner() -> JobRunner:
    runner = JobRunner(max_workers=1)
    job = runner.submit("warm-up", _warm_up).wait(300)
    if job.status != "done":
        raise RuntimeError(f"workbench warm-up failed: {job.error}")
    return runner


get_runner()

from app import services, theme  # noqa: E402
from genqai import export  # noqa: E402
from genqai.export import sha256_of, to_jsonable  # noqa: E402
from genqai.store import ExperimentStore  # noqa: E402

LIVE = "Source: live computation in this session."
COMMITTED = "Source: committed results files in this repository (not recomputed here)."


# ------------------------------------------------------------------ plumbing


def ss():
    s = st.session_state
    for k, v in (("circuits", {}), ("grades", {}), ("executions", {}), ("optimizations", {}), ("pending", {}),
                 ("errors", []), ("exports", {}), ("conformers", None), ("record", None), ("problem", None),
                 ("decision", None), ("counter", 0)):
        s.setdefault(k, v)
    return s


def get(obj, name, default=None):
    if obj is None:
        return default
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def unavailable(lane_key: str, what: str = "This panel") -> None:
    st.info(f"{what} is not available in this build: it needs {services.LANES[lane_key][1]} "
            f"(`{services.LANES[lane_key][0]}`), which is not installed.")


def submit(kind: str, payload, fn, *args, meta: dict | None = None, inline_s: float = 15.0) -> None:
    """Run fn through the shared JobRunner (deduped by content key), wait briefly inline, then hand over to
    the job monitor. Results are applied in collect_jobs() on the next run."""
    key = f"{kind}:{sha256_of(payload)[:16]}"
    s = ss()
    job = get_runner().submit(key, fn, *args)
    s.pending[key] = {"kind": kind, **(meta or {})}
    job.wait(inline_s)
    if not job.active:
        st.rerun()


def busy(kind: str) -> bool:
    runner = get_runner()
    return any(m["kind"] == kind and runner.get(k) is not None and runner.get(k).active
               for k, m in ss().pending.items())


def collect_jobs() -> None:
    s, runner = ss(), get_runner()
    for key, meta in list(s.pending.items()):
        job = runner.get(key)
        if job is None:
            s.pending.pop(key)
            continue
        if job.active:
            continue
        s.pending.pop(key)
        if job.status != "done":
            s.errors.append(f"{meta['kind']} job {job.status}: {job.error or ''}")
            continue
        apply_result(meta, job.result)


def apply_result(meta: dict, result) -> None:
    s = ss()
    kind = meta["kind"]
    if kind == "build":
        rec, how, decision, decision_error = result
        s.record, s.record_how, s.decision, s.decision_error = rec, how, decision, decision_error
        s.last_config = meta["config"]
    elif kind == "generate":
        circ, log = result
        s.counter += 1
        method = log.get("method") or meta["method"]  # "auto" resolves to the recommended method
        name = f"{method} #{s.counter}"
        s.circuits[name] = {"circuit": circ, "log": log, "method": method, "geometry_id": circ.geometry_id}
        s.gen_view = name
    elif kind == "grade":
        s.grades[meta["name"]] = result
    elif kind == "optimize":
        s.optimizations[meta["name"]] = result
        opt = get(result, "optimized")
        if opt is not None:
            s.circuits[f"{meta['name']} -> opt:{meta['objective']}"] = {
                "circuit": opt, "method": f"optimize:{meta['objective']}", "geometry_id": opt.geometry_id,
                "log": {"method": f"optimize:{meta['objective']}", "parent": meta["name"],
                        "evals_used": get(result, "evals_used")}}
            after = get(result, "after")
            if after is not None:
                s.grades[f"{meta['name']} -> opt:{meta['objective']}"] = after
    elif kind == "execute":
        s.executions.setdefault(meta["name"], []).append(result)


@st.fragment(run_every=1.0)
def job_monitor() -> None:
    s, runner = ss(), get_runner()
    if not s.pending:
        return
    finished = False
    for key, meta in list(s.pending.items()):
        job = runner.get(key)
        if job is None or not job.active:
            finished = True
            continue
        with st.status(f"{meta['kind']}: {job.status} ({(job.elapsed or 0):.0f} s)", expanded=True):
            for m in job.messages[-5:]:
                st.write(m)
            if st.button("Cancel", key=f"cancel-{key}"):
                runner.cancel(key)
    if finished:
        st.rerun()


def current_circuits() -> dict:
    s = ss()
    if s.record is None:
        return {}
    return {k: v for k, v in s.circuits.items() if v["geometry_id"] == s.record.geometry_id}


def show_errors() -> None:
    s = ss()
    for e in s.errors:
        st.error(e)
    if s.errors and st.button("Dismiss messages"):
        s.errors = []
        st.rerun()


# ------------------------------------------------------------------ sidebar

def sidebar() -> None:
    env = services.environment()
    lanes = services.lane_status()
    st.sidebar.header("Environment")
    for name, (ok, detail) in env.items():
        st.sidebar.markdown(f"{':green[yes]' if ok else ':red[no]'} &nbsp; {detail}")
    st.sidebar.header("Workbench modules")
    for name, (ok, desc) in lanes.items():
        st.sidebar.markdown(f"{':green[installed]' if ok else ':gray[missing]'} &nbsp; {desc}")
    st.sidebar.header("Features")
    a = lanes["molecule"][0]
    feats = {
        "Preset molecules": True,
        "SMILES / XYZ / atom input": a,
        "SMILES conformers": a and env["rdkit"][0],
        "Live Hamiltonians": env["pyscf"][0],
        "Routing and model registry": lanes["registry"][0],
        "Transformer inference": True,  # torch or the NumPy exports; checkpoints decide per system
        "Full grading / optimization": lanes["grading"][0] and lanes["optimize"][0],
        "Execution on simulators": lanes["execution"][0] and lanes["backends"][0],
        "IonQ cloud simulators": lanes["execution"][0] and env["ionq_key"][0] and env["qiskit_ionq"][0],
    }
    for f, ok in feats.items():
        st.sidebar.markdown(f"{':green[on]' if ok else ':gray[off]'} &nbsp; {f}")
    st.sidebar.caption("Paid QPU backends are never offered. The Qollab browser bundle (qollab/main.py) is a "
                       "separate, fixed-flow entry point.")


# ------------------------------------------------------------------ Molecule

def molecule_inputs(molmod):
    """-> (molecule spec or None, atoms for preview, preset tuple or None, error string or None)."""
    source = st.radio("Input", ["Preset", "SMILES", "XYZ", "Atoms"], horizontal=True, key="mol_source")
    if source == "Preset":
        c1, c2 = st.columns([1, 2])
        system = c1.selectbox("System", list(services.PRESET_RANGES), key="preset_system",
                              format_func=lambda s: {"h2": "H2 (2 qubits)", "lih4": "LiH (4 qubits)",
                                                     "lih6": "LiH (6 qubits)", "h4": "H4 chain (8 qubits)"}[s])
        params = tuple(c2.slider(f"{p} (Angstrom)", lo, hi, default, 0.01, key=f"preset_{system}_{p}")
                       for p, (lo, hi, default) in services.PRESET_RANGES[system].items())
        atoms = services.preset_atoms(system, params)
        mol = molmod.from_preset(system, params) if molmod is not None else None
        return mol, atoms, (system, params), None
    if molmod is None:
        unavailable("molecule", f"{source} input")
        return None, None, None, None
    c1, c2, c3 = st.columns(3)
    charge = c1.number_input("Charge", -4, 4, 0, key="mol_charge")
    mult = c2.number_input("Multiplicity (2S+1)", 1, 5, 1, key="mol_mult")
    basis = c3.selectbox("Basis", ["sto-3g", "6-31g", "cc-pvdz"], key="mol_basis")
    try:
        if source == "SMILES":
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            smiles = c1.text_input("SMILES", "[H][H]", key="smiles")
            seed = c2.number_input("Conformer seed", 0, 2**31 - 1, 20261010, key="smiles_seed")
            opt = c3.selectbox("Force-field optimizer", ["none", "MMFF", "UFF"], key="smiles_opt")
            n_conf = c4.number_input("Conformers", 1, molmod.LIMITS["max_conformers"], 1, key="smiles_nconf")
            if st.checkbox("Charge and multiplicity from the SMILES (formal charges, radicals)", True,
                           key="smiles_auto_charge"):
                charge, mult = None, None
            req = (smiles, seed, opt, n_conf, charge, mult, basis)
            s = ss()
            if st.button("Embed conformers", key="smiles_go") or (s.conformers and s.conformers[0] != req):
                s.conformers = (req, molmod.from_smiles(smiles, charge=charge, multiplicity=mult, basis=basis,
                                                        seed=int(seed), optimizer=None if opt == "none" else opt,
                                                        num_conformers=int(n_conf)))
            if not s.conformers:
                st.caption("Press 'Embed conformers' to generate 3D geometries (RDKit ETKDGv3).")
                return None, None, None, None
            confs = s.conformers[1]
            i = st.selectbox("Conformer", range(len(confs)), key="conf_idx",
                             format_func=lambda k: f"#{k}  energy={get(confs[k].conformer, 'energy')}")
            mol = confs[i]
        elif source == "XYZ":
            up = st.file_uploader("XYZ file", type=["xyz", "txt"], key="xyz_file")
            text = st.text_area("...or paste XYZ", "", height=120, key="xyz_text")
            if up is not None:
                if up.size > molmod.LIMITS["max_xyz_bytes"]:
                    return None, None, None, f"XYZ file too large ({up.size} bytes > {molmod.LIMITS['max_xyz_bytes']})"
                text = up.getvalue().decode("utf-8", errors="replace")
            if not text.strip():
                return None, None, None, None
            mol = molmod.from_xyz(text, charge=charge, multiplicity=mult, basis=basis)
        else:
            df = st.data_editor(pd.DataFrame({"symbol": ["H", "H"], "x": [0.0, 0.0], "y": [0.0, 0.0],
                                              "z": [0.0, 0.74]}), num_rows="dynamic", key="atom_editor")
            df = df.dropna()
            atoms = [(str(r.symbol).strip(), (float(r.x), float(r.y), float(r.z))) for r in df.itertuples()]
            if not atoms:
                return None, None, None, "add at least one atom"
            mol = molmod.from_atoms(atoms, charge=charge, multiplicity=mult, basis=basis)
    except ValueError as exc:  # MoleculeError is a ValueError with a user-facing message
        return None, None, None, str(exc)
    return mol, [(s_, tuple(xyz)) for s_, xyz in mol.atoms], None, None


def problem_config(molmod, mol):
    with st.expander("Active space and mapping", expanded=False):
        use_default = st.checkbox("Use defaults (presets: exactly the trained system's settings)", True,
                                  key="cfg_default")
        if use_default:
            return molmod.default_config(mol)
        c1, c2, c3, c4 = st.columns(4)
        ne = c1.number_input("Active electrons (0 = auto)", 0, 20, 0, key="cfg_ne")
        no = c2.number_input("Active orbitals (0 = auto)", 0, 20, 0, key="cfg_no")
        mapping = c3.selectbox("Mapping", ["parity", "jordan_wigner"], key="cfg_map")
        frozen = c4.checkbox("Frozen core", True, key="cfg_frozen")
        return molmod.ProblemConfig(molecule=mol, active_electrons=int(ne) or None, active_orbitals=int(no) or None,
                                    mapping=mapping, frozen_core=frozen)


def capability_states(molmod, mol, preset, err, current: bool) -> None:
    """current: the session's Hamiltonian was built from exactly this input."""
    s = ss()
    lanes = services.lane_status()
    rows = []
    accepted = not err and (mol is not None or (molmod is None and preset is not None))
    rows.append(("Input accepted", "yes" if accepted else "no",
                 err or ("validated by genqai.molecule" if molmod else "preset (legacy path, no Lane A)")
                 if accepted or err else "no molecule entered yet"))
    rec = s.record if current else None
    rows.append(("Hamiltonian built", "yes" if rec else "no",
                 f"{rec.num_qubits} qubits, {len(rec.pauli_terms)} Pauli terms; {s.get('record_how', '')}" if rec
                 else "press 'Build Hamiltonian'"))
    d = s.decision if current else None
    if not lanes["registry"][0]:
        legacy = services.legacy_methods(rec.system) if rec else []
        rows.append(("Trained-model inference", "legacy" if legacy else "unknown",
                     f"routing needs {services.LANES['registry'][1]}; legacy checkpoints: {legacy or 'none'}"))
        rows.append(("Fallback (conventional solver)", "n/a", "needs Lane B generate (ADAPT / UCCSD-VQE)"))
    elif d is None:
        rows.append(("Trained-model inference", "pending", s.get("decision_error") or "build the Hamiltonian first"))
        rows.append(("Fallback (conventional solver)", "pending", ""))
    else:
        path = get(d, "path")
        names = {"A": "A: validated", "B": "B: compatible, out of domain (experimental)",
                 "C": "C: no compatible generator", "D": "D: exceeds resources"}
        rows.append(("Trained-model inference", "yes" if path in ("A", "B") else "no",
                     f"path {names.get(path, path)}; " + "; ".join(get(d, "reasons", []))))
        conv = [m for m in get(d, "available_methods", []) if m.split("+")[0] in ("adapt", "uccsd_vqe")]
        rows.append(("Fallback (conventional solver)", "yes" if conv else "no", ", ".join(conv) or "none offered"))
    be = services.lane("backends")
    if be is None:
        rows.append(("Backend execution", "n/a", f"needs {services.LANES['backends'][1]}"))
    elif rec is None:
        rows.append(("Backend execution", "pending", "build the Hamiltonian first"))
    else:
        ok = [b.name for b in be.list_backends() if b.available and not b.paid
              and (b.max_qubits is None or b.max_qubits >= rec.num_qubits)]
        rows.append(("Backend execution", "yes" if ok else "no",
                     f"backends wide enough for {rec.num_qubits} qubits (unpacked): {', '.join(ok) or 'none'}"))
    st.dataframe(pd.DataFrame(rows, columns=["capability", "state", "detail"]), hide_index=True,
                 width="stretch")


def build_job(ctx, molmod_present: bool, config, preset, report):
    """Hamiltonian + routing in one background job."""
    if molmod_present:
        rec, how = services.build_problem_job(ctx, config)
    else:
        rec, how = services.build_preset_legacy(ctx, *preset)
    decision, decision_error = None, None
    reg = services.lane("registry")
    if reg is not None:
        ctx.progress("routing to a compatible generator")
        try:
            decision = reg.route(config, rec, report)
        except Exception as exc:
            decision_error = f"routing failed: {type(exc).__name__}: {exc}"
    return rec, how, decision, decision_error


def molecule_tab() -> None:
    molmod = services.lane("molecule")
    if molmod is None:
        st.caption("genqai.molecule (Lane A) is not installed: preset molecules use the legacy path "
                   "(stored grid geometry or a live PySCF build via genqai.demo.get_record).")
    mol, atoms, preset, err = molecule_inputs(molmod)
    config, report = None, None
    if mol is not None and molmod is not None:
        try:
            warnings = molmod.validate_molecule(mol)
            for w in warnings:
                st.warning(w)
            config = problem_config(molmod, mol)
            rec = ss().record
            built = rec if rec is not None and getattr(rec, "problem_key", "") == config.key else None
            report = molmod.assess(config, built)  # with the built record: IonQ packed width too
        except ValueError as exc:
            err = str(exc)
    if err:
        st.error(err)
    left, right = st.columns([3, 2])
    with left:
        if atoms:
            html = services.viewer_html(atoms)
            if html:
                st.iframe(html, height=340)  # trusted: py3Dmol output for validated atoms
            else:
                st.caption("py3Dmol not installed: no 3D preview.")
    with right:
        if report is not None:
            st.caption(LIVE + " (cheap capability assessment, no SCF unless needed)")
            c1, c2 = st.columns(2)
            c1.metric("Formula", report.formula)
            c2.metric("Electrons", report.n_electrons)
            c1.metric("Qubits (estimated)", report.num_qubits)
            c2.metric("Active space", f"{report.active_electrons} e / {report.active_orbitals} o")
            c1.metric("Exact reference feasible", "yes" if report.exact_feasible else "no")
            fit = report.fits_ionq
            c2.metric("IonQ packed fit", "after build" if fit is None
                      else f"{'yes' if fit else 'no'} ({report.ionq_packed_qubits} q)",
                      help="Qubits for all measurement groups packed into one circuit (Qollab's IonQ gateway "
                           "takes one single-circuit job per run); IonQ simulators allow 29.")
            for e in report.errors:
                st.error(e)
            for w in report.warnings:
                st.warning(w)
        elif preset is not None:
            from genqai.chemistry import SYSTEMS

            spec = SYSTEMS[preset[0]]
            st.metric("Molecule", spec.molecule)
            st.metric("Active space", f"{spec.active_electrons} e / {spec.num_active_orbitals} o ({spec.mapping})")
    can_build = (config is not None and not (report is not None and report.errors)) or \
                (molmod is None and preset is not None)
    if can_build:
        payload = {"config": config.key if config is not None else None, "preset": preset}
        if st.button("Build Hamiltonian", type="primary", disabled=busy("build"), key="build_btn"):
            submit("build", payload, build_job, molmod is not None, config, preset, report,
                   meta={"config": to_jsonable(config) if config is not None else {"preset": preset}})
    s = ss()
    r = s.record
    current = r is not None and (
        getattr(r, "problem_key", "") == config.key if config is not None else
        preset is not None and r.system == preset[0] and
        tuple(r.geometry_params.values()) == tuple(float(p) for p in preset[1]))
    if r is not None:
        st.success(f"Current Hamiltonian: {r.system} {r.geometry_params or ''} | {r.num_qubits} qubits | "
                   f"{r.reference_method} reference {r.reference_energy:.6f} Ha | HF {r.hf_energy:.6f} Ha | "
                   f"{s.get('record_how', '')}")
        if not current:
            st.info("The current Hamiltonian (used by Generate, Grade and Execute) was built from a different "
                    "input. Press 'Build Hamiltonian' to switch to this one.")
    capability_states(molmod, mol, preset, err, current)


# ------------------------------------------------------------------ Generate

def operator_table(circ) -> pd.DataFrame:
    from genqai.pool import get_pool

    pool = get_pool(circ.system)
    return pd.DataFrame({"step": range(len(circ.operators)), "operator": circ.operators,
                         "excitation": [pool.describe(k) for k in circ.operators],
                         "angle (rad)": [float(a) for a in circ.angles]})


def circuit_figure(record, circ):
    from genqai.circuits import build_circuit

    qc = build_circuit(record, circ.operators, circ.angles)
    fig = qc.draw("mpl", fold=30)
    return fig


def show_log(log: dict) -> None:
    st.markdown(services.ZERO_EVAL_STATEMENT.format(
        q=log.get("quantum_evals_to_propose", "?"), f=log.get("forward_passes", "?"),
        r=log.get("refine_energy_evals", 0), p=log.get("adapt_pool_gradient_evals", 0)))
    if log.get("experimental"):
        st.warning("Experimental: outside the validated domain of the model.")
    with st.expander("Generation log"):
        st.json(log)


def generate_tab() -> None:
    s = ss()
    rec = s.record
    if rec is None:
        st.info("Build a Hamiltonian on the Molecule tab first.")
        return
    gen = services.lane("generate")
    d = s.decision
    if gen is not None and d is not None:
        avail = list(get(d, "available_methods", []))
        methods = list(dict.fromkeys(["auto"] + [m.split("+")[0] for m in avail]))
        st.markdown(f"**Registry recommendation:** `{get(d, 'recommended')}` (path {get(d, 'path')})")
        for reason in get(d, "reasons", []):
            st.caption(f"why: {reason}")
        if get(d, "ood"):
            with st.expander("Domain check"):
                st.json(to_jsonable(get(d, "ood")))
    else:
        if gen is None:
            unavailable("generate", "Routed generation (all methods, refine, best-of-K)")
        elif d is None:
            st.warning(s.get("decision_error") or "No routing decision for this Hamiltonian.")
        methods = services.legacy_methods(rec.system)
        if not methods:
            return
        st.caption("Legacy preset path: one-shot generation from checkpoints/<system>/ via genqai.demo.")
    c1, c2, c3 = st.columns([2, 1, 1])
    base = c1.selectbox("Method", methods, key="gen_method",
                        format_func=lambda m: services.METHOD_LABELS.get(m, m))
    allowed = set(avail) if gen is not None and d is not None else set()
    refine = c2.checkbox("+ refine angles", False, disabled=f"{base}+refine" not in allowed, key="gen_refine",
                         help="BFGS re-optimisation of the angles; its energy evaluations are counted separately")
    bestofk = c3.checkbox("best-of-K", False, disabled=f"{base}+bestofk" not in allowed, key="gen_bok",
                          help="sample K candidates and keep the lowest-energy one (K energy evaluations)")
    refine, bestofk = refine and f"{base}+refine" in allowed, bestofk and f"{base}+bestofk" in allowed
    opts = {}
    if bestofk:
        opts["k"] = st.slider("K (candidates)", 2, 64, 16, key="gen_k")
    if base == "transformer_budget":
        opts["cnot_budget"] = st.slider("CX budget", 2, 400, 60, key="gen_budget")
    method = base + ("+bestofk" if bestofk else "") + ("+refine" if refine else "")
    payload = {"g": rec.geometry_id, "sys": rec.system, "method": method, "opts": opts}
    if st.button("Generate circuit", type="primary", disabled=busy("generate"), key="gen_btn"):
        submit("generate", payload, services.generate_job, rec, method, d, opts, meta={"method": method})
    circs = current_circuits()
    if not circs:
        return
    st.divider()
    st.caption(LIVE)
    name = st.selectbox("Circuit", list(circs)[::-1], key="gen_view")
    entry = circs[name]
    circ = entry["circuit"]
    if not circ.valid:
        st.error("The generator did not produce a valid circuit (no end-of-sequence token).")
    show_log(entry["log"])
    st.dataframe(operator_table(circ), hide_index=True, width="stretch")
    if st.checkbox("Show circuit diagram", key="gen_draw"):
        fig = circuit_figure(rec, circ)
        st.pyplot(fig)
        plt.close(fig)


# ------------------------------------------------------------------ Grade

def show_report(rep, title: str) -> None:
    r = to_jsonable(rep)
    st.markdown(f"**{title}** status: `{r.get('status')}`")
    if r.get("basic"):
        st.caption("Basic scoring only (genqai.evaluate.score_circuit); full grading needs "
                   f"{services.LANES['grading'][1]}.")
    e = r.get("energy") or {}
    c = st.columns(4)
    c[0].metric("Energy (Ha)", f"{e['energy']:.6f}" if isinstance(e.get("energy"), float) else "-")
    c[1].metric("Error (mHa)", f"{e['error_mha']:.4f}" if isinstance(e.get("error_mha"), float) else "-")
    c[2].metric(f"Reference ({e.get('reference_kind', '-')})",
                f"{e['reference']:.6f}" if isinstance(e.get("reference"), float) else "-")
    res = r.get("resources") or {}
    c[3].metric("CX (Qiskit basis)", res.get("cx_qiskit", "-"))
    tabs = st.tabs(["checks", "resources", "cost", "noise"])
    tabs[0].json(r.get("checks") or {"checks": "not computed"})
    tabs[1].json(res)
    tabs[2].json(r.get("cost") or {})
    tabs[3].json(r.get("noise") or {"noise": "not requested"})
    if r.get("weighted_score") is not None:
        st.caption(f"weighted score {r['weighted_score']:.4g} (convenience only; weights {r.get('weights')})")


def pareto_plot(reports: dict) -> None:
    rows = []
    for name, rep in reports.items():
        r = to_jsonable(rep)
        err = (r.get("energy") or {}).get("error_mha")
        cx = (r.get("resources") or {}).get("cx_qiskit")
        if isinstance(err, (int, float)) and err == err and cx is not None:
            rows.append({"circuit": name, "|error| mHa": abs(err), "CX": cx, "status": r.get("status")})
    if not rows:
        return
    front = services.pareto(reports)
    df = pd.DataFrame(rows)
    df["Pareto front"] = df["circuit"].isin(front or [])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    other, best = df[~df["Pareto front"]], df[df["Pareto front"]]
    ax.scatter(other["CX"], other["|error| mHa"].clip(lower=1e-6), s=48, color="#8a8984", label="graded circuit",
               edgecolors="white", linewidths=2, zorder=2)
    if len(best):
        ax.scatter(best["CX"], best["|error| mHa"].clip(lower=1e-6), s=64, color="#2a78d6", label="Pareto front",
                   edgecolors="white", linewidths=2, zorder=3)
    for label, cx, err in zip(df["circuit"], df["CX"], df["|error| mHa"]):
        ax.annotate(label, (cx, max(err, 1e-6)), fontsize=7, color="#52514e", xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(1.6, color="#52514e", lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 1.6, " 1.6 mHa", va="center", ha="left", fontsize=7, color="#52514e")
    ax.set_yscale("log")
    ax.set_xlabel("CX count (Qiskit basis)")
    ax.set_ylabel("|E - E_ref| (mHa, log)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if front is None:
        st.caption("Pareto front highlighting needs Lane C (genqai.grading.pareto_front).")
    st.dataframe(df, hide_index=True, width="stretch")


def grade_tab() -> None:
    s = ss()
    rec, circs = s.record, current_circuits()
    if rec is None or not circs:
        st.info("Generate a circuit first.")
        return
    full = services.lane("grading") is not None
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    name = c1.selectbox("Circuit", list(circs)[::-1], key="grade_circ")
    target = c2.number_input("Target (mHa)", 0.01, 100.0, 1.6, key="grade_target", disabled=not full)
    shots = c3.number_input("Shots (0 = exact only)", 0, 1_000_000, 0, step=1000, key="grade_shots", disabled=not full)
    noise = c4.selectbox("Noise model", ["none", "local"], key="grade_noise", disabled=not full,
                         help="local: finite shots + GenQAI's depolarizing approximation (not IonQ's model). IonQ noise: Execute tab.")
    entry = circs[name]
    payload = {"g": rec.geometry_id, "c": to_jsonable(entry["circuit"]), "t": target, "s": shots, "n": noise}
    if st.button("Grade", type="primary", disabled=busy("grade"), key="grade_btn"):
        submit("grade", payload, services.grade_job, rec, entry["circuit"], entry["log"], target,
               shots or None, None if noise == "none" else noise, meta={"name": name})
    if not full:
        unavailable("grading", "Full grading (checks, resources, cost, noise)")
    if name in s.grades:
        st.caption(LIVE)
        show_report(s.grades[name], name)
    graded = {k: v for k, v in s.grades.items() if k in circs}
    if graded:
        st.subheader("Accuracy vs CX: all circuits graded this session for this Hamiltonian")
        pareto_plot(graded)
    st.subheader("Optimize")
    if services.lane("optimize") is None:
        unavailable("optimize", "Circuit optimization")
        return
    c1, c2 = st.columns(2)
    objective = c1.selectbox("Objective", ["accuracy", "min_2q", "min_depth", "min_cost", "noise", "pareto"],
                             key="opt_obj")
    max_evals = c2.number_input("Max energy evaluations", 10, 5000, 500, key="opt_evals")
    payload = {"g": rec.geometry_id, "c": to_jsonable(entry["circuit"]), "o": objective, "m": max_evals, "t": target}
    if st.button("Optimize", disabled=busy("optimize"), key="opt_btn"):
        submit("optimize", payload, services.optimize_job, rec, entry["circuit"], objective, target, int(max_evals),
               meta={"name": name, "objective": objective})
    res = s.optimizations.get(name)
    if res is not None:
        st.caption(LIVE + f" Energy evaluations used by the optimizer: {get(res, 'evals_used')}; same Hamiltonian "
                   f"check: {get(res, 'same_hamiltonian_check')}")
        b, a = st.columns(2)
        with b:
            show_report(get(res, "before"), "before")
        with a:
            show_report(get(res, "after"), "after")
        steps = get(res, "steps") or []
        if steps:
            st.dataframe(pd.DataFrame(to_jsonable(steps)), width="stretch")


# ------------------------------------------------------------------ Execute

def execute_tab() -> None:
    s = ss()
    rec, circs = s.record, current_circuits()
    be, ex = services.lane("backends"), services.lane("execution")
    if be is None or ex is None:
        unavailable("execution" if ex is None else "backends", "Execution on simulators and IonQ")
        return
    infos = be.list_backends()
    st.dataframe(pd.DataFrame([{k: get(b, k) for k in ("name", "available", "reason", "max_qubits", "noise_model",
                                                       "needs_key", "paid", "description")} for b in infos]),
                 hide_index=True, width="stretch")
    st.caption("Unavailable and paid backends cannot be selected; IonQ backends need an API key and are free "
               "cloud simulators. The width check runs before any submission.")
    if rec is None or not circs:
        st.info("Generate a circuit first.")
        return
    usable = [b.name for b in infos if b.available and not b.paid]
    if not usable:
        st.warning("No backend is available in this environment.")
        return
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
    name = c1.selectbox("Circuit", list(circs)[::-1], key="exec_circ")
    backend = c2.selectbox("Backend (available, free only)", usable, key="exec_backend")
    shots = c3.number_input("Shots", 100, 100_000, 2000, step=100, key="exec_shots")
    seed = c4.number_input("Seed", 0, 2**31 - 1, 0, key="exec_seed")
    packed = {"backend default": None, "packed": True, "unpacked": False}[
        c5.selectbox("Packing", ["backend default", "packed", "unpacked"], key="exec_packed")]
    circ = circs[name]["circuit"]
    base = {"g": rec.geometry_id, "c": to_jsonable(circ), "b": backend, "s": shots, "seed": seed, "p": packed}
    b1, b2 = st.columns(2)
    if b1.button("Dry run (width check, no submission)", disabled=busy("execute"), key="exec_dry"):
        submit("execute", {**base, "dry": True}, services.execute_job, rec, circ, backend, int(shots), int(seed),
               packed, True, meta={"name": name})
    if b2.button("Execute", type="primary", disabled=busy("execute"), key="exec_go"):
        submit("execute", {**base, "dry": False}, services.execute_job, rec, circ, backend, int(shots), int(seed),
               packed, False, meta={"name": name}, inline_s=5)
    runs = s.executions.get(name, [])
    if runs:
        st.caption(LIVE)
        last = to_jsonable(runs[-1])
        if last.get("status") not in ("ok", "dry_run"):
            st.error(f"last execution: {last.get('status')}: {last.get('error') or ''}")
        for note in last.get("notes") or []:
            st.caption(note)
        if last.get("planned"):
            with st.expander("Plan of the last execution (circuits, widths, shots)"):
                st.json(last["planned"])
        cols = ["execution_id", "backend", "noise_model", "shots", "packed", "num_groups", "width", "energy", "stderr",
                "error_mha", "job_id", "status", "runtime_s"]
        rows = [{k: to_jsonable(r).get(k) for k in cols} for r in runs]
        g = s.grades.get(name)
        if g is not None:
            e = to_jsonable(g).get("energy") or {}
            rows.insert(0, {"execution_id": "-", "backend": "exact statevector (Grade tab)",
                            "energy": e.get("energy"), "stderr": 0.0, "error_mha": e.get("error_mha"),
                            "status": "computed"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"{rec.reference_method} reference energy {rec.reference_energy:.6f} Ha"
                   f"{'' if rec.reference_method == 'FCI' else ' (not exact)'}. Ideal vs noisy: compare rows by "
                   "noise_model.")


# ------------------------------------------------------------------ Benchmarks

@st.cache_data(show_spinner=False)
def committed_summary(system: str) -> pd.DataFrame:
    return pd.DataFrame(services.committed_summary(system))


def benchmarks_tab() -> None:
    st.caption(COMMITTED + " Tables are recomputed from results_fresh/<system>/*.jsonl (held-out test_fresh "
               "geometries) with genqai.evaluate.summarize.")
    system = st.selectbox("System", ["h2", "lih4", "lih6", "h4"], key="bench_system")
    st.dataframe(committed_summary(system), hide_index=True, width="stretch")
    for path, caption in services.committed_figures():
        with st.expander(f"{path.stem}: {caption}"):
            st.image(str(path), caption=f"committed figure {path.name}: {caption}")


# ------------------------------------------------------------------ History & export

def experiment_config():
    return ss().get("last_config")


def build_exports(rec, name, entry) -> dict[str, str | Exception]:
    s = ss()
    circ = entry["circuit"]
    out: dict[str, str | Exception] = {}
    for fname, fn in (("circuit.py", lambda: export.qiskit_code(rec, circ)),
                      ("circuit.qasm2", lambda: export.qasm(rec, circ, 2)),
                      ("circuit.qasm3", lambda: export.qasm(rec, circ, 3)),
                      ("hamiltonian.json", lambda: export.hamiltonian_json(rec)),
                      ("metadata.json", lambda: export.metadata_json(
                          {"config": experiment_config(), "method": entry["method"], "circuit_name": name,
                           "generation_log": entry["log"], "record": rec, "circuit": circ,
                           "routing": to_jsonable(s.decision)}))):
        try:
            out[fname] = fn()
        except Exception as exc:
            out[fname] = exc
    graded = {k: v for k, v in s.grades.items() if k in current_circuits()}
    if graded:
        out["benchmark.csv"] = export.benchmark_csv(graded)
    for i, run in enumerate(s.executions.get(name, [])):
        out[f"measurements_{i}.csv"] = export.measurements_csv(run)
    return out


def history_tab() -> None:
    s = ss()
    store = ExperimentStore()
    rec, circs = s.record, current_circuits()
    st.subheader("Export this session")
    if rec is None or not circs:
        st.info("Generate a circuit first.")
    else:
        name = st.selectbox("Circuit", list(circs)[::-1], key="exp_circ")
        entry = circs[name]
        c1, c2 = st.columns(2)
        if c1.button("Prepare exports", key="exp_prepare"):
            s.exports[name] = build_exports(rec, name, entry)
        if c2.button("Save experiment to runs/experiments", key="exp_save", disabled=name not in s.exports):
            eid = store.create(experiment_config(), entry["method"], record=rec,
                               meta={"circuit": name, "system": rec.system, "geometry_id": rec.geometry_id})
            for k, v in circs.items():
                store.add_circuit(eid, k, v["circuit"], gen_log=v["log"])
            store.save_grades(eid, {k: v for k, v in s.grades.items() if k in circs})
            for run in s.executions.get(name, []):
                store.add_execution(eid, run)
            for fname, text in s.exports[name].items():
                if isinstance(text, str):
                    store.save_export(eid, fname, text)
            st.success(f"saved experiment {eid}")
        for fname, text in s.exports.get(name, {}).items():
            if isinstance(text, Exception):
                st.warning(f"{fname}: not exportable ({type(text).__name__}: {text})")
            else:
                st.download_button(f"Download {fname}", text, file_name=fname, key=f"dl-{name}-{fname}")
    st.subheader("Saved experiments")
    exps = store.list()
    if not exps:
        st.caption("No saved experiments in runs/experiments/.")
        return
    st.dataframe(pd.DataFrame(exps), hide_index=True, width="stretch")
    eid = st.selectbox("Experiment", [e["experiment_id"] for e in exps], key="hist_eid")
    data = store.load(eid)
    st.json({k: data[k] for k in ("experiment_id", "method", "created_utc", "config", "meta")})
    for fname in data["exports"]:
        st.download_button(f"Download {fname}", store.read_export(eid, fname), file_name=f"{eid}_{fname}",
                           key=f"hist-dl-{eid}-{fname}")
    if st.checkbox("Confirm delete", key="hist_confirm") and st.button("Delete experiment", key="hist_delete"):
        store.delete(eid)
        st.rerun()


# ------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="GenQAI workbench", layout="wide")
    theme.apply()
    ss()
    collect_jobs()
    st.title("GenQAI molecular circuit workbench")
    st.caption("Molecule → Hamiltonian → routing → one-shot circuit generation → grading → execution → export")
    sidebar()
    show_errors()
    job_monitor()
    tabs = st.tabs(["Molecule", "Generate", "Grade", "Execute", "Benchmarks", "History & export"])
    for tab, fn in zip(tabs, (molecule_tab, generate_tab, grade_tab, execute_tab, benchmarks_tab, history_tab)):
        with tab:
            fn()


main()
