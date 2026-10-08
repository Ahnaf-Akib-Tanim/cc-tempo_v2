"""CC-Tempo web tool (Reviewer 4). Gradio app; deployable for free on Hugging Face Spaces.

Users pick a pretrained model, a set of starting cells, then run in-silico experiments:
  * knock out ligand families (receiver side)
  * add exogenous ligands / cytokines at a chosen dose
  * set lineage genes (z-score) in the starting cells
and get predicted fate distributions, a comparison with the unperturbed run, and a landscape plot.

    python app/app.py            # local
    MODEL_RUNS=work/weinreb/runs CCT_ROOT=. python app/app.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cctempo.api import CCTempoSession  # noqa: E402

ROOT = os.environ.get("CCT_ROOT", ".")
RUNS = sorted(glob.glob(os.path.join(ROOT, os.environ.get("MODEL_RUNS", "work/*/runs"), "*", "model_best.pt")))
_cache = {}


def session(run):
    if run not in _cache:
        _cache[run] = CCTempoSession(run, ROOT, "auto")
    return _cache[run]


def choices(run):
    s = session(run)
    i0 = s.data["times"].index(s.t0)
    types = sorted(set(map(str, s.data["celltype"][i0])))
    return (gr.update(choices=s.families, value=[]), gr.update(choices=s.cytokines, value=None),
            gr.update(choices=types, value=types[0] if types else None))


def run_experiment(run, celltype, n_cells, ko, cyt, dose, genes, gene_value, copies):
    s = session(run)
    cells = s.start_cells(celltype)
    rng = np.random.default_rng(0)
    cells = rng.choice(cells, min(int(n_cells), len(cells)), replace=False)
    gene_list = [g.strip() for g in (genes or "").replace(",", " ").split() if g.strip()]
    base = s.predict_fate(cells, n_copies=int(copies))
    pert = s.predict_fate(cells, ko_families=ko or [], exogenous={cyt: float(dose)} if cyt else None,
                          gene_set=gene_list or None, gene_value=float(gene_value), n_copies=int(copies))
    classes = [c for c in s.clf.classes]
    comp = pd.DataFrame({"cell type": classes, "unperturbed %": (100 * base[classes].mean()).round(2).values,
                         "perturbed %": (100 * pert[classes].mean()).round(2).values})
    comp["change (pp)"] = (comp["perturbed %"] - comp["unperturbed %"]).round(2)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    idx = np.arange(len(classes))
    ax.bar(idx - 0.2, comp["unperturbed %"], 0.4, label="unperturbed")
    ax.bar(idx + 0.2, comp["perturbed %"], 0.4, label="perturbed")
    ax.set_xticks(idx)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("% of simulated descendants")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(ROOT, "cct_app_result.csv")
    pert.to_csv(path, index=False)
    return comp, fig, path


def build():
    with gr.Blocks(title="CC-Tempo") as demo:
        gr.Markdown("# CC-Tempo\nCommunication-aware simulation of cell fate. Choose a trained model, starting "
                    "cells and an in-silico perturbation; predictions are Monte-Carlo simulations of the learned "
                    "dynamics (they are hypotheses, not measurements).")
        with gr.Row():
            run = gr.Dropdown(RUNS, label="model (run directory)", value=RUNS[0] if RUNS else None)
            celltype = gr.Dropdown([], label="starting cell type")
            n_cells = gr.Slider(10, 500, 100, step=10, label="number of starting cells")
            copies = gr.Slider(50, 1000, 200, step=50, label="simulated copies per cell")
        with gr.Row():
            ko = gr.CheckboxGroup([], label="knock out ligand families (receiver side)")
        with gr.Row():
            cyt = gr.Dropdown([], label="add exogenous ligand / cytokine")
            dose = gr.Slider(0, 4, 1, step=0.25, label="relative dose")
            genes = gr.Textbox(label="genes to set in starting cells (space/comma separated)")
            gene_value = gr.Slider(-5, 5, 2.5, step=0.5, label="z-score value")
        btn = gr.Button("Run")
        table = gr.Dataframe(label="predicted composition")
        plot = gr.Plot()
        file = gr.File(label="per-cell predictions (csv)")
        run.change(choices, run, [ko, cyt, celltype])
        demo.load(choices, run, [ko, cyt, celltype]) if RUNS else None
        btn.click(run_experiment, [run, celltype, n_cells, ko, cyt, dose, genes, gene_value, copies], [table, plot, file])
    return demo


if __name__ == "__main__":
    build().launch()
