"""Zip the small, important outputs of a Kaggle session for download (no raw data, no h5ad).

    python scripts/pack_outputs.py --root /kaggle/working --out /kaggle/working/cct_outputs.zip
    python scripts/pack_outputs.py --root /kaggle/working --with-models   # also model_best.pt (few MB each)

Upload the zip back (or unzip into the project folder) and tell Claude which steps ran.
"""
import _common
import os
import zipfile

p = _common.parser(__doc__)
p.add_argument("--out", default=None)
p.add_argument("--with-models", action="store_true")
args = p.parse_args()
out = args.out or os.path.join(args.root, "cct_outputs.zip")
keep_ext = {".csv", ".json", ".yaml", ".txt", ".npz", ".npy", ".md", ".png", ".pdf"}
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(os.path.join(args.root, "work")):
        for f in files:
            path = os.path.join(base, f)
            ext = os.path.splitext(f)[1]
            big = os.path.getsize(path) > 50e6
            if (ext in keep_ext and not big) or (args.with_models and f == "model_best.pt"):
                z.write(path, os.path.relpath(path, args.root))
                n += 1
            elif f in ("liana_table.csv.gz",) or f.startswith("lr_") and f.endswith(".json"):
                z.write(path, os.path.relpath(path, args.root))
                n += 1
print(f"packed {n} files -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
