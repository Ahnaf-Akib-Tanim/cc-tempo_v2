"""Glue used by scripts/, the CLI and the app: paths, presets, train-one-run, evaluate-one-run."""
import copy
import json
import os
import pickle
import shutil

import numpy as np
import torch
import yaml

from .data.dataset import cells, load_data, time_index
from .evaluate.fate import fate_bias_from_counts, fit_classifier, mc_fate_counts, prepare_field
from .evaluate.metrics import (composition_metrics, distribution_metrics, fate_metrics, observed_composition)
from .evaluate.pathint import build_operator, entropy_bits, kl_bits
from .models import load_run_model
from .simulate import simulate
from .train import Trainer
from .utils import get_device, load_config, save_json

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(HERE), "configs")


def presets():
    with open(os.path.join(CONFIG_DIR, "models.yaml")) as f:
        return yaml.safe_load(f)


def P(cfg, key, root="."):
    return os.path.join(root, cfg["paths"][key])


def make_cfg(cfg_path, preset=None, overrides=None):
    cfg = load_config(cfg_path, overrides)
    if preset:
        cfg["model"] = {**cfg["model"], **presets()[preset]}
        cfg["preset"] = preset
    return cfg


def load_dataset(cfg, root=".", lr_method=None):
    data = load_data(P(cfg, "data_pt", root))
    if lr_method and lr_method != data["lr"].get("method"):
        with open(os.path.join(P(cfg, "work", root), f"lr_{lr_method}.json")) as f:
            data["lr"] = json.load(f)
    return data


def run_name(preset, seed, tag=""):
    return f"{preset}{('_' + tag) if tag else ''}_s{seed}"


def _stale_signal(out, cfg):
    """True for an existing mean-field run trained before signals were bounded (no model.sig_clip)."""
    m = cfg["model"]
    used = os.path.join(out, "config_used.yaml")
    if m.get("mode") != "meanfield" or m.get("context") in ("off", False) or not os.path.exists(used):
        return False
    with open(used) as f:
        return "sig_clip" not in (yaml.safe_load(f) or {}).get("model", {})


def train_one(cfg_path, preset, seed, root=".", device="auto", overrides=None, tag="", lr_method=None):
    cfg = make_cfg(cfg_path, preset, overrides)
    data = load_dataset(cfg, root, lr_method)
    out = os.path.join(P(cfg, "runs", root), run_name(preset, seed, tag))
    if _stale_signal(out, cfg):
        stale = os.path.join(P(cfg, "runs", root), "_stale", os.path.basename(out))
        shutil.rmtree(stale, ignore_errors=True)
        shutil.move(out, stale)
        print(f"[stale] {out} used the unbounded mean-field signal; moved to {stale}, retraining")
    if os.path.exists(os.path.join(out, "run_summary.json")):
        print(f"[skip] {out} already finished")
        return out
    cfg["lr_method"] = lr_method or (data["lr"] or {}).get("method")
    tr = Trainer(cfg, data, out, device=str(get_device(device)), seed=seed)
    with open(os.path.join(out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    tr.fit()
    return out


# ---------------------------------------------------------------------------- evaluation
def classifier_for(cfg, data, root="."):
    stem = os.path.splitext(os.path.basename(cfg["paths"]["data_pt"]))[0]
    path = os.path.join(P(cfg, "work", root), f"classifier_{stem}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    clf = fit_classifier(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    return clf


def _exog(model, cfg, cond):
    cc = (cfg["train"].get("condition_cytokines") or {}).get(cond)
    if not model.uses_field or cc is None:
        return None
    e = model.exogenous_field(cc).detach()
    return lambda t: e


def _eval_sets(data, cfg):
    """Yield (condition, prog_idx, target_bias) for the dataset's lineage-tracing ground truth."""
    ev = data.get("eval") or {}
    if "clonal" in ev:
        if "prog_idx" in ev["clonal"]:
            yield None, ev["clonal"]["prog_idx"], ev["clonal"]["bias"]
        else:
            for cond, d in ev["clonal"].items():
                yield cond, d["prog_idx"], d["bias"]
    if "synthetic" in ev:
        for cond, d in ev["synthetic"].items():
            yield cond, d["prog_idx"], d["p_A"]


def evaluate_run(run_dir, root=".", device="auto", which="best", n_copies=None, do_pathint=True,
                 do_dist=True, eval_overrides=None):
    """Evaluate a finished run using the exact config it was trained with (config_used.yaml)."""
    cfg = load_config(os.path.join(run_dir, "config_used.yaml"), eval_overrides)
    lr_method = cfg.get("lr_method")
    dev = str(get_device(device))
    data = load_dataset(cfg, root, lr_method)
    model, ck = load_run_model(run_dir, data, dev, which)
    clf = classifier_for(cfg, data, root)
    ev = cfg["eval"]
    t0 = float(ev.get("start_time", cfg["train"].get("start_time", data["times"][0])))
    t1 = float(ev.get("end_time", data["times"][-1]))
    dt = float(ev.get("dt", 0.1))
    nc = int(n_copies or ev.get("n_copies", 2000))
    pos, neg = ev.get("pos"), ev.get("neg")
    i0 = time_index(data, t0)
    res = {"run": os.path.basename(run_dir), "preset": cfg.get("preset"), "seed": ck.get("seed"),
           "epoch": ck.get("epoch"), "best_val": ck.get("best_val"), "fate": {}, "pathint": {}, "dist": {}}
    preds = {}
    conds = list(ev.get("conditions") or cfg["train"].get("conditions") or [None])
    for cond, prog, target in _eval_sets(data, cfg):
        exog = _exog(model, cfg, cond)
        field = prepare_field(model, data, t0, t1, dt, int(ev.get("n_bg", 5000)), exog_fn=exog, seed=0, device=dev)
        x_prog = data["x"][i0][torch.from_numpy(np.asarray(prog))].to(dev)
        y_prog = data["v1"][i0][torch.from_numpy(np.asarray(prog))].to(dev) if model.mode == "v1" else None
        counts = mc_fate_counts(model, x_prog, t0, t1, dt, clf, nc, int(ev.get("max_particles", 200000)),
                                field_traj=field, exog_fn=exog, y_prog=y_prog, seed=0, device=dev)
        bias, mask = fate_bias_from_counts(counts, clf, pos, neg)
        key = str(cond)
        res["fate"][key] = fate_metrics(bias, target, mask)
        preds[f"mc_{key}"] = bias
        preds[f"target_{key}"] = np.asarray(target)
        if do_pathint:
            pi_cfg = ev.get("pathint", {})
            op, _ = build_operator(model, data, t0, t1, clf, pi_cfg, dt, exog, {i0: np.asarray(prog)}, dev,
                                   n_bg=int(ev.get("n_bg", 5000)))
            Ts = op.transitions(t0, t1, dt, field_traj=field, exog_fn=exog)
            proba = clf.predict_proba(op.X.numpy())
            hp = op.committor(Ts, proba[:, clf.index(pos)])
            hn = op.committor(Ts, proba[:, clf.index(neg)])
            nodes = op.node_index(i0, prog)
            pib = hp[nodes] / np.clip(hp[nodes] + hn[nodes], 1e-12, None)
            res["pathint"][key] = fate_metrics(pib, target)
            preds[f"pi_{key}"] = pib
            # forward KL (bits) to each later observed time
            p0 = op.histogram(data["x"][i0].numpy())
            kl = {}
            for t in data["times"]:
                if t <= t0:
                    continue
                Tt = op.transitions(t0, t, dt, field_traj=field, exog_fn=exog)
                psim = op.evolve(Tt, p0)[-1]
                m = cells(data, t, "all", cond if data.get("condition") is not None and cond else None)
                pobs = op.histogram(data["x"][time_index(data, t)][torch.from_numpy(np.where(m)[0])].numpy())
                kl[str(t)] = {"kl_bits": kl_bits(pobs, psim), "entropy_bits": entropy_bits(pobs)}
            res["pathint"][key]["kl"] = kl
    if do_dist:
        for cond in conds:
            exog = _exog(model, cfg, cond)
            m0 = cells(data, t0, "all", None)
            idx = np.where(m0)[0]
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, min(5000, len(idx)), replace=False)
            x = data["x"][i0][torch.from_numpy(idx)].to(dev)
            g = data["growth"][i0][torch.from_numpy(idx)].to(dev)
            y = data["v1"][i0][torch.from_numpy(idx)].to(dev) if model.mode == "v1" else None
            t_prev, d = t0, {}
            gen = torch.Generator(device=dev).manual_seed(0)
            for t in data["times"]:
                if t <= t0:
                    continue
                out = simulate(model, x, t_prev, t, dt, y0=y, w0=torch.exp(g * (t_prev - t0)), growth=g,
                               exog_fn=exog, generator=gen)
                x, y, t_prev = out["x"].detach(), out["y"], t
                w = torch.exp(g * (t - t0)).cpu().numpy()
                rs = np.random.default_rng(1).choice(len(w), min(3000, len(w)), p=w / w.sum())
                xs = x.cpu().numpy()[rs]
                m = cells(data, t, "all", cond)
                obs = data["x"][time_index(data, t)][torch.from_numpy(np.where(m)[0])].numpy()
                if len(obs) == 0:
                    continue
                dm = distribution_metrics(xs, obs)
                pred_lab = np.array(clf.classes)[clf.predict_idx(torch.from_numpy(xs)).numpy()]
                obs_lab = np.asarray(data["celltype"][time_index(data, t)])[m]
                dm.update(composition_metrics(observed_composition(pred_lab), observed_composition(obs_lab)))
                dm["holdout"] = float(t) in [float(h) for h in cfg["train"].get("holdout_times") or []]
                d[str(t)] = dm
            res["dist"][str(cond)] = d
    save_json(res, os.path.join(run_dir, f"metrics_{which}.json"))
    np.savez_compressed(os.path.join(run_dir, f"preds_{which}.npz"), **preds)
    return res
