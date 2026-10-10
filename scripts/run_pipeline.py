"""Reproduce everything from scratch, in order, with one command.

    PYTHONPATH=$PWD python scripts/run_pipeline.py                 # all systems, full training
    PYTHONPATH=$PWD python scripts/run_pipeline.py --quick         # h2 + lih4, short training
    PYTHONPATH=$PWD python scripts/run_pipeline.py --skip-train    # use committed checkpoints
    PYTHONPATH=$PWD python scripts/run_pipeline.py --systems h2 --ionq --steps demo figures

Steps (each one is the existing script, run as a subprocess so a step behaves exactly as when run
by hand): hamiltonians -> teacher -> canonical -> baselines -> train -> generate -> demo -> figures.
The demo step measures the generated one-shot circuit of the first test geometry on the local
backends (results/<system>/ionq_demo_<id>_sft_oneshot.json); with --ionq and an IONQ_API_KEY it also
runs h2 on the IonQ cloud simulator (ideal + aria-1 + forte-1, 4000 shots per backend).
Outputs overwrite data/, checkpoints/, results/ and figures/ in place (git restores them).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ["h2", "lih4", "lih6", "h4"]
QUICK_SYSTEMS = ["h2", "lih4"]
STEPS = ["hamiltonians", "teacher", "canonical", "baselines", "train", "generate", "demo", "figures"]
TRAIN_SEED = {"h4": 2}  # val-selected seed of the committed h4 checkpoint; 0 elsewhere
QUICK_EPOCHS = 300
IONQ_SYSTEMS = ["h2"]  # free IonQ quota: cloud runs for the 2-qubit system only
IONQ_SHOTS = 4000


def commands(a) -> list[tuple[str, list[str]]]:
    s = a.systems
    cmds = [
        ("hamiltonians", ["scripts/make_hamiltonians.py", *s]),
        ("teacher", ["scripts/make_teacher.py", *s]),
        *[("canonical", ["scripts/make_canonical_teacher.py", "--system", x]) for x in s],
        ("baselines", ["scripts/run_baselines.py", *s]),
        *[("train", ["scripts/train_sft.py", "--system", x, "--seed", str(TRAIN_SEED.get(x, 0)),
                     *(["--epochs", str(QUICK_EPOCHS)] if a.quick else [])]) for x in s],
        *[("generate", ["scripts/generate.py", "--system", x]) for x in s],
    ]
    sys.path.insert(0, str(ROOT))
    from genqai.measure import ionq_available  # after argparse so --help works without the stack

    use_ionq = a.ionq and ionq_available()
    if a.ionq and not use_ionq:
        print("--ionq given but no IONQ_API_KEY (env or .env): demo runs local backends only")
    for x in s:
        cloud = use_ionq and x in IONQ_SYSTEMS
        cmds.append(("demo", [
            "scripts/run_ionq_demo.py", "--system", x, "--circuits", f"results/{x}/gen_oneshot.jsonl",
            "--tag", "_sft_oneshot", "--shots", str(IONQ_SHOTS if cloud else 20_000),
            "--backend", "exact", "exact_probs", "local_shots", "local_noisy",
            *(["ionq_ideal", "ionq_noisy", "--ionq-noise-models", "aria-1", "forte-1"] if cloud else []),
        ]))
    cmds.append(("figures", ["scripts/make_figures.py"]))
    return [(step, c) for step, c in cmds if step in a.steps and not (a.skip_train and step == "train")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=None)
    p.add_argument("--quick", action="store_true", help=f"{'+'.join(QUICK_SYSTEMS)} only, {QUICK_EPOCHS} SFT epochs")
    p.add_argument("--skip-train", action="store_true", help="keep the committed checkpoints/<system>/sft.pt")
    p.add_argument("--ionq", action="store_true", help="IonQ cloud simulator runs in the demo step (needs key)")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS, help="run only these steps")
    a = p.parse_args(argv)
    a.systems = a.systems or (QUICK_SYSTEMS if a.quick else SYSTEMS)

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    timings, t_all = [], time.perf_counter()
    for step, cmd in commands(a):
        print(f"\n### [{step}] python {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)
        timings.append((step, " ".join(cmd[1:]), time.perf_counter() - t0))
    print("\n### timings")
    for step, args, dt in timings:
        print(f"{step:<13}{dt:8.1f}s  {args}")
    print(f"{'total':<13}{time.perf_counter() - t_all:8.1f}s")


if __name__ == "__main__":
    main()
