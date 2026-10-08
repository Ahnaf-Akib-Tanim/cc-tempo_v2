"""Shared bits for the numbered scripts (importable without installing the package)."""
import argparse
import os
import sys

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)


def parser(desc, config_default=None):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=config_default, help="dataset config yaml")
    p.add_argument("--root", default=os.environ.get("CCT_ROOT", "."),
                   help="where raw/ and work/ live (Kaggle: /kaggle/working)")
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=800")
    return p


def cfg_path(name):
    return name if os.path.exists(name) else os.path.join(PROJECT, "configs", name)
