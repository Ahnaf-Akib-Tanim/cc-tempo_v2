"""Everything the Kaggle notebook does after cloning: restore earlier work, run stages, pack outputs.

    python scripts/kaggle_run.py --stage A+B+C [--set train.epochs=1000] [--hours 10.75]

Kept in the repo (not in the notebook) so that fixes reach Kaggle through `git clone`.
Stages are listed in TODO_PLAN.md and in the notebook's first cell.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.environ.get("CCT_INPUT", "/kaggle/input")  # overridable for local tests

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--stage", required=True, help="stages joined by '+', e.g. D+E")
ap.add_argument("--root", default="/kaggle/working")
ap.add_argument("--set", nargs="*", default=[], help="config overrides passed to every step")
ap.add_argument("--hours", type=float, default=10.75, help="Kaggle stops a version at 12 h")
ap.add_argument("--allow-cpu", action="store_true", help="run GPU stages even without a GPU (very slow)")
args = ap.parse_args()
ROOT = args.root
R = f"--root {ROOT}"
os.environ.update(CCT_ROOT=ROOT, POT_BACKEND_DISABLE_TENSORFLOW="1", POT_BACKEND_DISABLE_JAX="1",
                  PYTHONUNBUFFERED="1")
os.environ.setdefault("CCT_DEADLINE", str(time.time() + args.hours * 3600))
os.chdir(CODE)


LOG = None  # whole console log, copied into work/ so it travels in cct_outputs.zip


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


# ------------------------------------------------------------------ restore
def restore():
    """Copy every `work/` folder found in the attached inputs (any depth) into ROOT/work.

    Kaggle mounts notebook outputs at different depths (/kaggle/input/<slug>/ or
    /kaggle/input/notebooks/<user>/<slug>/...), so the search does not assume one layout."""
    if not os.path.isdir(INPUT):
        say("no /kaggle/input: nothing to restore")
        return
    say("attached inputs:")
    works, zips = [], []
    for base, dirs, files in os.walk(INPUT):
        depth = base[len(INPUT):].count(os.sep)
        if depth <= 3:
            say("   ", base)
        if os.path.basename(base) == "work":
            works.append(base)
            dirs[:] = []
            continue
        if depth >= 6:
            dirs[:] = []
        zips += [os.path.join(base, f) for f in files if f.startswith("cct_outputs") and f.endswith(".zip")]
    for z in zips:
        say("unzipping", z)
        subprocess.run(["unzip", "-o", "-q", z, "-d", ROOT])
    for w in sorted(works, key=os.path.getmtime):  # newest last, so it wins
        say("restoring", w)
        shutil.copytree(w, f"{ROOT}/work", dirs_exist_ok=True)
    if not works and not zips:
        say("!!! nothing restored: no work/ folder in the inputs. Attach the latest version of this "
            "notebook (+ Add Input -> Your Work), or start from a data stage (A, F1, G1).")
    for ds in ("weinreb", "cytokine", "neural_tube", "synthetic"):
        d = f"{ROOT}/work/{ds}"
        if os.path.isdir(d):
            n_runs = len(glob.glob(f"{d}/runs/*/run_summary.json"))
            say(f"  work/{ds}: data.pt {'yes' if os.path.exists(d + '/data.pt') else 'NO'}, finished runs {n_runs}")


# ------------------------------------------------------------------ helpers
def run(cmd):
    """Run one pipeline command and stream its output; on failure print and continue."""
    if args.set and "--set" not in cmd:
        cmd += " --set " + " ".join(args.set)
    say("\n$", cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, errors="replace")
    for line in p.stdout:
        say(line.rstrip("\n"))
    rc = p.wait()
    if rc != 0:
        say(f"!!! FAILED ({rc}): {cmd}")
    return rc


def grid(dataset, tier):
    """Run one tier of configs/experiments.yaml, one shard per GPU in parallel, and wait."""
    import torch
    n = max(1, torch.cuda.device_count())
    procs = []
    for k in range(n):
        dev = f"cuda:{k}" if torch.cuda.is_available() else "cpu"
        c = f"{sys.executable} scripts/06_run_grid.py --dataset {dataset} --tier {tier} --shard {k}/{n} --device {dev} {R}"
        if args.set:
            c += " --set " + " ".join(args.set)
        say("\n$", c)
        procs.append(subprocess.Popen(c + f" > {ROOT}/grid_{dataset}_{tier}_{k}.log 2>&1", shell=True))
    for p in procs:
        p.wait()
    say(subprocess.run(f"tail -n 15 {ROOT}/grid_{dataset}_{tier}_*.log", shell=True, capture_output=True, text=True).stdout)


def need(path, hint):
    if os.path.exists(f"{ROOT}/{path}"):
        return True
    say(f"!!! {path} is missing, so this stage is skipped. {hint}")
    return False


WEINREB_HINT = "Attach the version that ran stage A, or run 'A+' before this stage."


# ------------------------------------------------------------------ stages
def stage(st):
    py = sys.executable
    if st == "A":      # cheap to repeat: step 1 skips itself when its outputs exist
        run(f"{py} scripts/01_prepare_weinreb.py --config weinreb.yaml {R}")
        run(f"{py} scripts/02_liana_ccc.py --config weinreb.yaml {R}")
        run(f"{py} scripts/03_build_dataset.py --config weinreb.yaml {R}")
        shutil.rmtree(f"{ROOT}/raw/weinreb", ignore_errors=True)   # 2 GB download no longer needed
    elif st == "B":
        run(f"{py} scripts/04_synthetic_benchmark.py {R}")
    elif st == "C":
        if need("work/weinreb/data.pt", WEINREB_HINT):
            for preset in ("v2_meanfield", "prescient"):
                run(f"{py} scripts/05_train.py --config weinreb.yaml --preset {preset} --seed 99 --tag timing {R} "
                    "--set train.epochs=50 train.eval_every=25")
            run(f"tail -n 2 {ROOT}/work/weinreb/runs/*timing*/train_log.csv")
    elif st == "D":
        if need("work/weinreb/data.pt", WEINREB_HINT):
            grid("weinreb", "tier1")
    elif st == "E":
        if need("work/weinreb/data.pt", WEINREB_HINT):
            run(f"{py} scripts/07_evaluate.py --config weinreb.yaml {R}")
            run(f"{py} scripts/08_baselines.py --config weinreb.yaml {R} --methods wot cfm")
            run(f"{py} scripts/09_interpret.py --config weinreb.yaml {R} --run v2_meanfield_s0")
            run(f"{py} scripts/13_early_progenitors.py --config weinreb.yaml {R}")
    elif st == "EP":   # early (CD34+) progenitors + held-out-clone upper bound, from saved predictions
        if need("work/weinreb/data.pt", WEINREB_HINT):
            run(f"{py} scripts/13_early_progenitors.py --config weinreb.yaml {R}")
    elif st == "W2":
        if need("work/weinreb/data.pt", WEINREB_HINT):
            grid("weinreb", "tier2")
            run(f"{py} scripts/07_evaluate.py --config weinreb.yaml {R}")
    elif st == "W3":
        if need("work/weinreb/data.pt", WEINREB_HINT):
            grid("weinreb", "tier3")
            run(f"{py} scripts/07_evaluate.py --config weinreb.yaml {R}")
    elif st == "F1":
        run(f"{py} scripts/10_cytokine_experiments.py --config cytokine.yaml --evidence {R}")
        run(f"{py} scripts/01_prepare_weinreb.py --config cytokine.yaml {R}")
        run(f"{py} scripts/02_liana_ccc.py --config cytokine.yaml {R}")
        run(f"{py} scripts/03_build_dataset.py --config cytokine.yaml {R}")
        shutil.rmtree(f"{ROOT}/raw/weinreb_cytokine", ignore_errors=True)
    elif st == "F2":
        if need("work/cytokine/data.pt", "Attach the version that ran stage F1, or run 'F1+F2'."):
            grid("cytokine", "tier1")
            grid("cytokine", "tier2")
            run(f"{py} scripts/07_evaluate.py --config cytokine.yaml {R}")
            run(f"{py} scripts/10_cytokine_experiments.py --config cytokine.yaml --dose-response --run v2_meanfield_s0 {R}")
    elif st == "G1":
        run(f"{py} scripts/11_neural_tube.py --prepare {R}")
        shutil.rmtree(f"{ROOT}/raw/neural_tube", ignore_errors=True)
        run(f"cat {ROOT}/work/neural_tube/inspect.txt")
    elif st == "G2":
        if need("work/neural_tube", "Attach the version that ran stage G1."):
            run(f"{py} scripts/02_liana_ccc.py --config neural_tube.yaml {R}")
            run(f"{py} scripts/03_build_dataset.py --config neural_tube.yaml {R}")
            grid("neural_tube", "tier1")
            run(f"{py} scripts/07_evaluate.py --config neural_tube.yaml {R}")
            run(f"{py} scripts/11_neural_tube.py --analyze --run v2_meanfield_s0 {R}")
    else:
        raise ValueError(f"unknown stage {st}")


STAGES = [x.strip().upper() for x in args.stage.split("+") if x.strip()]
bad = [x for x in STAGES if x not in ("A", "B", "C", "D", "E", "EP", "W2", "W3", "F1", "F2", "G1", "G2")]
if bad:
    sys.exit(f"unknown stage(s) {bad}; valid: A B C D E EP W2 W3 F1 F2 G1 G2")
GPU_STAGES = {"C", "D", "E", "W2", "W3", "F2", "G2"}


def pack():
    """Copy logs into work/ and zip the small outputs; done after every stage so a 12 h kill loses little."""
    os.makedirs(f"{ROOT}/work", exist_ok=True)
    LOG.flush()
    for f in glob.glob(f"{ROOT}/grid_*.log") + glob.glob(f"{ROOT}/kaggle_log_*.txt"):
        shutil.copy(f, f"{ROOT}/work/")           # logs travel with work/ into the next version
    run(f"{sys.executable} scripts/pack_outputs.py {R} --with-models")


os.makedirs(ROOT, exist_ok=True)
LOG = open(f"{ROOT}/kaggle_log_{time.strftime('%Y%m%d_%H%M')}.txt", "w")
import torch  # noqa: E402
if GPU_STAGES & set(STAGES) and not torch.cuda.is_available() and not args.allow_cpu:
    say(f"!!! no GPU, but stage(s) {sorted(GPU_STAGES & set(STAGES))} need one (CPU is ~10x slower and would hit "
        "the 12 h limit). Settings -> Accelerator -> GPU T4 x2, then Save & Run All again. Nothing was run.")
    sys.exit(1)
restore()
run(f"{sys.executable} scripts/00_check_env.py")
for st in STAGES:
    say("=" * 20, "STAGE", st, "=" * 20)
    stage(st)
    pack()

