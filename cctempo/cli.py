"""Command-line interface (installed as `cctempo`).

    cctempo train    --config configs/weinreb.yaml --preset v2_meanfield --seed 0 [--root DIR]
    cctempo evaluate --run work/weinreb/runs/v2_meanfield_s0 [--root DIR]
    cctempo fate     --run work/weinreb/runs/v2_meanfield_s0 --cells 0 5 17 [--ko CSF] [--exo M-csf=1.0]
    cctempo app      [--port 7860]
"""
import argparse
import os

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cctempo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True); t.add_argument("--preset", required=True)
    t.add_argument("--seed", type=int, default=0); t.add_argument("--root", default=".")
    t.add_argument("--device", default="auto"); t.add_argument("--set", nargs="*", default=[])
    t.add_argument("--tag", default=""); t.add_argument("--lr-method", default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True); e.add_argument("--root", default="."); e.add_argument("--device", default="auto")
    f = sub.add_parser("fate")
    f.add_argument("--run", required=True); f.add_argument("--root", default="."); f.add_argument("--device", default="auto")
    f.add_argument("--cells", nargs="+", type=int, required=True, help="indices into the start time point")
    f.add_argument("--ko", nargs="*", default=[], help="ligand families to knock out")
    f.add_argument("--exo", nargs="*", default=[], help="exogenous ligands NAME=relative_dose")
    f.add_argument("--copies", type=int, default=500)
    a = sub.add_parser("app")
    a.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .runner import train_one
        print(train_one(args.config, args.preset, args.seed, args.root, args.device, args.set, args.tag, args.lr_method))
    elif args.cmd == "evaluate":
        from .runner import evaluate_run
        evaluate_run(args.run, args.root, args.device)
    elif args.cmd == "fate":
        import numpy as np
        from .api import CCTempoSession
        s = CCTempoSession(args.run, args.root, args.device)
        exo = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.exo}
        print(s.predict_fate(np.array(args.cells), ko_families=args.ko, exogenous=exo, n_copies=args.copies).to_string())
    elif args.cmd == "app":
        from app.app import build
        build().launch(server_port=args.port)


if __name__ == "__main__":
    main()
