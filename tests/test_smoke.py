"""CPU smoke tests on a tiny synthetic dataset (runs in ~1-2 min; no GPU needed).

    pytest -q tests/test_smoke.py
"""
import os
import tempfile

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")

import numpy as np
import pytest
import torch

from cctempo.data.synthetic import make_synthetic
from cctempo.evaluate.fate import fate_bias_from_counts, fit_classifier, mc_fate_counts, prepare_field
from cctempo.evaluate.metrics import fate_metrics
from cctempo.evaluate.pathint import PathIntegralOperator, build_operator, kl_bits
from cctempo.interpret.landscape import communication_bifurcation
from cctempo.interpret.perturb import FateRunner, lr_knockout_scan, temporal_block_scan
from cctempo.models import load_run_model
from cctempo.sinkhorn import _ot_eps, sinkhorn_divergence
from cctempo.train import Trainer

COND = {"e0.0": {}, "e0.5": {"EXO": 0.5}, "e1.0": {"EXO": 1.0}}


@pytest.fixture(scope="module")
def data():
    d = make_synthetic(beta=1.5, n_cells=400)
    # fake v1 features so the v1 mode can be exercised
    d["v1"] = [torch.randn(x.shape[0], 6) for x in d["x"]]
    return d


def _cfg(mode, context="lr"):
    return {"model": {"mode": mode, "hidden": 32, "c_hidden": 16, "sigma": 0.5, "context": context},
            "train": {"start_time": 0.0, "dt": 0.1, "epochs": 3, "lr": 0.01, "batch_frac": 0.5, "max_batch": 200,
                      "eval_every": 2, "save_every": 2, "conditions": ["e0.0", "e1.0"], "condition_cytokines": COND}}


def test_sinkhorn_fallback_matches_zero():
    x = torch.randn(50, 3)
    a = torch.full((50,), 1 / 50)
    from cctempo.sinkhorn import _ot_sym
    val = _ot_eps(a, x, a, x, 0.1, 0.7) - _ot_sym(a, x, 0.1, 0.7)
    assert abs(float(val)) < 1e-3
    assert float(sinkhorn_divergence(None, x, None, x + 3.0, backend="torch")) > 1.0


@pytest.mark.parametrize("mode,context", [("prescient", "lr"), ("v1", "lr"), ("local", "lr"),
                                          ("meanfield", "lr"), ("meanfield", "shuffled"), ("meanfield", "off")])
def test_train_and_evaluate(data, mode, context):
    with tempfile.TemporaryDirectory() as out:
        tr = Trainer(_cfg(mode, context), data, out, device="cpu", seed=0)
        tr.fit()
        assert os.path.exists(os.path.join(out, "model_best.pt"))
        model, _ = load_run_model(out, data)
        clf = fit_classifier(data, max_cells=2000)
        gt = data["eval"]["synthetic"]["e1.0"]
        x_prog = data["x"][0][torch.from_numpy(gt["prog_idx"][:20])]
        exog = (lambda t: model.exogenous_field(COND["e1.0"])) if model.uses_field else None
        field = prepare_field(model, data, 0.0, 3.0, 0.1, n_bg=200, exog_fn=exog)
        y = data["v1"][0][torch.from_numpy(gt["prog_idx"][:20])] if mode == "v1" else None
        counts = mc_fate_counts(model, x_prog, 0.0, 3.0, 0.1, clf, n_copies=20, field_traj=field, exog_fn=exog, y_prog=y)
        bias, mask = fate_bias_from_counts(counts, clf, pos="A", neg="B")
        m = fate_metrics(bias, gt["p_A"][:20], mask)
        assert np.isfinite(m["mae"])
        op = PathIntegralOperator(model, data, n_points=300, dims=5, k=10, tau=0.5, kernel="gaussian")
        Ts = op.transitions(0.0, 3.0, 0.1, field_traj=field, exog_fn=exog)
        op2, _ = build_operator(model, data, 0.0, 3.0, clf, {"n_points": 300, "dims": 5, "n_paths": 4}, n_bg=100, exog_fn=exog)
        assert abs(op2.evolve(op2.transitions(0.0, 3.0, 0.1, field_traj=field, exog_fn=exog), np.full(op2.N, 1 / op2.N))[-1].sum() - 1) < 1e-6
        p = op.evolve(Ts, op.histogram(data["x"][0].numpy()))[-1]
        assert abs(p.sum() - 1) < 1e-6
        assert np.isfinite(kl_bits(op.histogram(data["x"][2].numpy()), p))


def test_interpretation(data):
    with tempfile.TemporaryDirectory() as out:
        Trainer(_cfg("meanfield"), data, out, device="cpu", seed=0).fit()
        model, _ = load_run_model(out, data)
        clf = fit_classifier(data, max_cells=2000)
        gt = data["eval"]["synthetic"]["e0.0"]
        runner = FateRunner(model, data, clf, gt["prog_idx"][:10], 0.0, 3.0, n_copies=10, n_bg=200, pos="A", neg="B")
        df = lr_knockout_scan(runner, level="family")
        assert len(df) == model.K
        tb = temporal_block_scan(runner, [0], [(0.0, 1.5), (1.5, 3.0)])
        assert len(tb) == 2
        op, field = build_operator(model, data, 0.0, 3.0, clf, {"n_points": 300, "dims": 5, "n_paths": 4}, n_bg=100)
        bif, _ = communication_bifurcation(model, op, field[-1], [0.0, 1.0, 2.0])
        assert "n_attractors" in bif
