"""Step 5: train ONE model (resumable: rerun the same command after a Kaggle timeout).

    python scripts/05_train.py --config weinreb.yaml --preset v2_meanfield --seed 0 --root /kaggle/working
    python scripts/05_train.py --config weinreb.yaml --preset prescient --seed 0 --set train.holdout_times=[4.0] --tag hold4
    python scripts/05_train.py --config weinreb.yaml --preset v2_meanfield --seed 0 --lr-method cellchat --tag cellchat
"""
import _common

from cctempo.runner import evaluate_run, train_one

p = _common.parser(__doc__, "weinreb.yaml")
p.add_argument("--preset", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--lr-method", default=None, help="use lr_<method>.json instead of the consensus")
p.add_argument("--evaluate", action="store_true", help="evaluate right after training")
args = p.parse_args()
out = train_one(_common.cfg_path(args.config), args.preset, args.seed, args.root, args.device, args.set, args.tag,
                args.lr_method)
print("run dir:", out)
if args.evaluate:
    evaluate_run(out, args.root, args.device)
