"""Small shared helpers: seeding, devices, configs, json io."""
import copy
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_config(path: str, overrides=None) -> dict:
    """Load a YAML config. Supports `inherit: other.yaml` and CLI overrides `a.b=c`."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(os.path.join(os.path.dirname(path), parent))
        cfg = _deep_update(base, cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def ns(d: dict) -> SimpleNamespace:
    """Recursive dict -> namespace (read-only convenience)."""
    out = SimpleNamespace()
    for k, v in d.items():
        setattr(out, k, ns(v) if isinstance(v, dict) else v)
    return out


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch.is_tensor(o):
            return o.detach().cpu().tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def deepcopy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def to_tensor(a, device=None, dtype=torch.float32):
    if torch.is_tensor(a):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
