"""Training: fit the landscape so that simulated populations match observed snapshots.

Loss = sum over observed later times t_k of the debiased Sinkhorn divergence between the
growth-weighted simulated population (started from the first time point) and observed cells,
plus (v1 only) the same loss on the communication features, plus an L1 penalty on the LR gates.
"""
import csv
import os

import numpy as np
import torch

from .data.dataset import cells, time_index
from .models.cctempo import build_model
from .simulate import simulate
from .sinkhorn import sinkhorn_divergence
from .utils import Timer, save_json, set_seed


def _sample(n_avail, frac, max_n, rng):
    n = int(min(max_n, max(64, round(frac * n_avail))))
    return rng.choice(n_avail, size=n, replace=n > n_avail)


class Trainer:
    def __init__(self, cfg: dict, data: dict, out_dir: str, device="cpu", seed=0):
        self.cfg, self.data, self.out_dir, self.device, self.seed = cfg, data, out_dir, device, seed
        self.t = cfg["train"]
        os.makedirs(out_dir, exist_ok=True)
        set_seed(seed)
        mcfg = dict(cfg["model"])
        mcfg.setdefault("context_seed", seed)
        self.mcfg = mcfg
        self.model = build_model(mcfg, data, device=device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(self.t.get("lr", 0.01)))
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=int(self.t.get("sched_step", 100)),
                                                     gamma=float(self.t.get("sched_gamma", 0.9)))
        self.rng = np.random.default_rng(seed)
        self.t0 = float(self.t.get("start_time", data["times"][0]))
        if self.t0 not in [float(t) for t in data["times"]]:
            print(f"[warn] start_time {self.t0} not in data times {data['times']}; using the first time point")
            self.t0 = float(data["times"][0])
        hold = [float(h) for h in (self.t.get("holdout_times") or [])]
        tt = self.t.get("train_times") or [t for t in data["times"] if float(t) > self.t0]
        self.train_times = sorted(float(t) for t in tt if float(t) not in hold)
        self.holdout_times = hold
        self.dt = float(self.t.get("dt", 0.1))
        self.conditions = self.t.get("conditions") or [None]
        self.cond_cyt = self.t.get("condition_cytokines") or {}
        self.start_condition = self.t.get("start_condition")
        self._prep()

    # ------------------------------------------------------------------
    def _tensor(self, t, mask, key="x"):
        i = time_index(self.data, t)
        arr = self.data[key][i]
        return arr[torch.from_numpy(np.where(mask)[0])].to(self.device)

    def _prep(self):
        d, dev = self.data, self.device
        self.src = {}
        for split in ("train", "val"):
            m = cells(d, self.t0, split, self.start_condition)
            i = time_index(d, self.t0)
            idx = torch.from_numpy(np.where(m)[0])
            self.src[split] = {
                "x": d["x"][i][idx].to(dev),
                "g": d["growth"][i][idx].to(dev),
                "y": d["v1"][i][idx].to(dev) if self.model.mode == "v1" else None,
            }
        self.tgt = {}
        for split in ("train", "val"):
            for cond in self.conditions:
                for t in self.train_times + self.holdout_times:
                    m = cells(d, t, split, cond)
                    if m.sum() == 0:
                        continue
                    self.tgt[(split, cond, t)] = {
                        "x": self._tensor(t, m),
                        "y": self._tensor(t, m, "v1") if self.model.mode == "v1" else None,
                    }

    def _exog_fn(self, cond):
        if not self.model.uses_field or cond is None or cond not in self.cond_cyt:
            return None
        cyt = self.cond_cyt[cond]
        return lambda t: self.model.exogenous_field(cyt, self.device)

    def _rollout_loss(self, split, cond, create_graph, generator=None, times=None):
        src = self.src[split]
        n0 = src["x"].shape[0]
        idx = torch.from_numpy(_sample(n0, float(self.t.get("batch_frac", 0.1)), int(self.t.get("max_batch", 3000)), self.rng)).to(self.device)
        x, g = src["x"][idx], src["g"][idx]
        y = src["y"][idx] if src["y"] is not None else None
        exog_fn = self._exog_fn(cond)
        blur, scaling = float(self.t.get("blur", 0.1)), float(self.t.get("scaling", 0.7))
        losses = {}
        t_prev = self.t0
        for t in sorted(times or self.train_times):
            key = (split, cond, t)
            w_prev = torch.exp(g * (t_prev - self.t0))
            out = simulate(self.model, x, t_prev, t, self.dt, y0=y, w0=w_prev, growth=g,
                           exog_fn=exog_fn, create_graph=create_graph, generator=generator)
            x, y = out["x"], out["y"]
            t_prev = t
            if key not in self.tgt:
                continue
            tgt = self.tgt[key]
            m = tgt["x"].shape[0]
            j = torch.from_numpy(_sample(m, float(self.t.get("batch_frac", 0.1)), int(self.t.get("max_batch", 3000)), self.rng)).to(self.device)
            a = torch.exp(g * (t - self.t0))
            l = sinkhorn_divergence(a / a.sum(), x, None, tgt["x"][j], blur, scaling)
            if self.model.mode == "v1":
                l = l + sinkhorn_divergence(a / a.sum(), y, None, tgt["y"][j], blur, scaling)
            losses[t] = l
        return losses

    # ------------------------------------------------------------------
    def _ckpt(self, name, epoch, best):
        torch.save({"state_dict": self.model.state_dict(), "model_cfg": self.mcfg, "epoch": epoch,
                    "best_val": best, "seed": self.seed}, os.path.join(self.out_dir, name))

    def _save_state(self, epoch, best):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                    "epoch": epoch, "best": best, "rng": self.rng.bit_generator.state},
                   os.path.join(self.out_dir, "state_last.pt"))

    @torch.no_grad()
    def validate(self, n_rep=2):
        tot = {}
        gen = torch.Generator(device=self.device).manual_seed(1234)
        for _ in range(n_rep):
            for cond in self.conditions:
                ls = self._rollout_loss("val", cond, create_graph=False, generator=gen,
                                        times=self.train_times + self.holdout_times)
                for t, l in ls.items():
                    tot.setdefault(t, []).append(float(l))
        return {t: float(np.mean(v)) for t, v in tot.items()}

    def fit(self):
        epochs = int(self.t.get("epochs", 1500))
        start, best = 0, float("inf")
        state_path = os.path.join(self.out_dir, "state_last.pt")
        if self.t.get("resume", True) and os.path.exists(state_path):
            st = torch.load(state_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(st["model"])
            self.opt.load_state_dict(st["opt"])
            self.sched.load_state_dict(st["sched"])
            self.rng.bit_generator.state = st["rng"]
            start, best = st["epoch"], st["best"]
            print(f"[resume] epoch {start}, best val {best:.4f}")
        if start >= epochs:
            return
        timer = Timer()
        max_min = float(self.t.get("max_minutes", 1e9))
        deadline = os.environ.get("CCT_DEADLINE")  # unix time; set by the Kaggle notebook
        if deadline:
            import time
            max_min = min(max_min, (float(deadline) - time.time()) / 60.0)
        log_path = os.path.join(self.out_dir, "train_log.csv")
        new_log = not os.path.exists(log_path) or start == 0
        fh = open(log_path, "w" if new_log else "a", newline="")
        wr = csv.writer(fh)
        if new_log:
            wr.writerow(["epoch", "train_loss", "val_train_times", "val_holdout", "gate_mean", "minutes"])
        eval_every = int(self.t.get("eval_every", 50))
        save_every = int(self.t.get("save_every", 100))
        lam = float(self.mcfg.get("gate_l1", self.t.get("gate_l1", 0.0)))  # a model preset may override
        clip = float(self.t.get("clip", 0.25))
        stopped_early = False
        for epoch in range(start, epochs):
            self.model.train()
            self.opt.zero_grad()
            tot = 0.0
            for cond in self.conditions:
                ls = self._rollout_loss("train", cond, create_graph=True)
                loss = sum(ls.values()) / max(1, len(self.conditions))
                if lam > 0 and self.model.K:
                    loss = loss + lam * torch.sigmoid(self.model.gate_logit).sum() / len(self.conditions)
                loss.backward()
                tot += float(loss.detach())
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.opt.step()
            self.sched.step()
            last = epoch + 1 == epochs
            if (epoch + 1) % eval_every == 0 or last:
                self.model.eval()
                v = self.validate()
                v_train = float(np.mean([v[t] for t in self.train_times if t in v])) if v else float("nan")
                v_hold = float(np.mean([v[t] for t in self.holdout_times if t in v])) if self.holdout_times and v else float("nan")
                gm = float(torch.sigmoid(self.model.gate_logit.detach()).mean()) if self.model.K else float("nan")
                wr.writerow([epoch + 1, tot, v_train, v_hold, gm, round(timer.elapsed() / 60, 2)])
                fh.flush()
                print(f"epoch {epoch+1}/{epochs} train {tot:.4f} val {v_train:.4f} holdout {v_hold:.4f}")
                if v_train < best:
                    best = v_train
                    self._ckpt("model_best.pt", epoch + 1, best)
            if (epoch + 1) % save_every == 0 or last:
                self._ckpt("model_last.pt", epoch + 1, best)
                self._save_state(epoch + 1, best)
            if timer.elapsed() / 60 > max_min:
                print(f"[time budget] stopping at epoch {epoch+1}; resume later with the same command")
                self._ckpt("model_last.pt", epoch + 1, best)
                self._save_state(epoch + 1, best)
                stopped_early = True
                break
        fh.close()
        if stopped_early:  # not finished: no run_summary.json, so the grid resumes this run next session
            return
        if not os.path.exists(os.path.join(self.out_dir, "model_best.pt")):
            self._ckpt("model_best.pt", epochs, best)
        save_json({"seed": self.seed, "best_val": best, "model": self.model.describe(), "config": self.cfg,
                   "minutes": timer.elapsed() / 60}, os.path.join(self.out_dir, "run_summary.json"))
