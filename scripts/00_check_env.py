"""Step 0: check packages, GPU, and run the CPU smoke tests' core path quickly."""
import _common  # noqa: F401  (sets env + sys.path)
import importlib
import time

import torch

for m in ["numpy", "scipy", "pandas", "sklearn", "torch", "scanpy", "anndata", "liana", "ot", "geomloss",
          "captum", "decoupler", "yaml"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:10s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:10s} MISSING ({type(e).__name__}: {e})")

print("CUDA:", torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

from cctempo.data.synthetic import make_synthetic  # noqa: E402
from cctempo.train import Trainer  # noqa: E402
import tempfile  # noqa: E402

t = time.time()
d = make_synthetic(n_cells=300)
cfg = {"model": {"mode": "meanfield", "hidden": 32, "c_hidden": 16}, "train": {"start_time": 0.0, "epochs": 2,
       "eval_every": 1, "conditions": ["e0.0"], "condition_cytokines": {"e0.0": {}}}}
with tempfile.TemporaryDirectory() as o:
    Trainer(cfg, d, o, device="cuda" if torch.cuda.is_available() else "cpu").fit()
print(f"core training path OK ({time.time() - t:.1f}s)")
