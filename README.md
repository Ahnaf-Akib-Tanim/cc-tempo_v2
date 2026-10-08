# CC-Tempo v2

**A cell-cell communication aware, mean-field temporal model of cellular dynamics.**

CC-Tempo learns a Waddington landscape from time-course single-cell RNA-seq. In this landscape, a cell's
drift depends on its own gene expression **and** on the ligands produced by the rest of the population
(plus any ligands added to the medium). Ligand-receptor (LR) knowledge comes from
[LIANA](https://github.com/saezlab/liana-py). Preprocessing uses [Scanpy](https://scanpy.readthedocs.io).
No R, Seurat or CellChat is needed.

```
dx_i = -grad_x U(x_i ; s_i(t)) dt + sigma dW
s_ik(t) = gate_k * ( alpha * L_k(population at t) + exogenous_k(t) ) * receptor_k(x_i)
```

What you can do with it:
- **Predict fates** of progenitors (Monte-Carlo, or deterministically with a path-integral operator).
- **Rank communication channels** by their causal effect in the model: receiver-side knockouts,
  sender-lineage silencing, and time-window blocking.
- **Simulate media or cytokine changes**, i.e. exogenous ligand dose-response.
- **Find attractors and basins**, and draw bifurcation diagrams against communication strength.
- **Attribute fate to progenitor genes** by integrated gradients through the SDE.

Code: https://github.com/Ahnaf-Akib-Tanim/cc-tempo_v2

## Install

```bash
git clone https://github.com/Ahnaf-Akib-Tanim/cc-tempo_v2.git && cd cc-tempo_v2
pip install -e ".[sc,dev]"      # scanpy, liana, POT, geomloss, captum, decoupler
pytest -q tests                  # CPU smoke tests, ~30 s
```

On Kaggle use `requirements-kaggle.txt` (see `kaggle/KAGGLE_GUIDE.md`).

## Quick start (your own data)

Input: an AnnData with log-normalised expression, a numeric time column and a cell-type column.

```python
import scanpy as sc
from cctempo.ccc.liana_prior import run_liana, select_pairs
from cctempo.data.preprocess import build_data, select_genes
from cctempo.data.dataset import save_data

adata = sc.read_h5ad("my_timecourse.h5ad")                    # log-normalised
table = run_liana(adata, groupby="celltype", time_key="day")  # LIANA consensus per time point
lr = select_pairs(table, adata.var_names, top_k=64)
genes = select_genes(adata, 2500, force_genes=[g for grp in lr["lig_genes"] + lr["rec_genes"] for g in grp])
data = build_data(adata, genes, "day", "celltype", lr=lr)
save_data(data, "work/mydata/data.pt")
```

Copy `configs/weinreb.yaml` to `configs/mydata.yaml`, edit the paths, times and cell-type names, then:

```bash
cctempo train    --config configs/mydata.yaml --preset v2_meanfield --seed 0
cctempo evaluate --run work/mydata/runs/v2_meanfield_s0
cctempo fate     --run work/mydata/runs/v2_meanfield_s0 --cells 0 1 2 --ko CSF --exo G-csf=1.0
cctempo app                                                     # web interface
```

## Reproducing the paper

Follow `TODO_PLAN.md`. The numbered scripts in `scripts/` run in order:

| step | script | what it does |
|---|---|---|
| 0 | `00_check_env.py` | packages, GPU, one training step |
| 1 | `01_prepare_weinreb.py` | download Weinreb 2020 (in vitro / cytokine), normalise, clonal ground truth |
| 2 | `02_liana_ccc.py` | LIANA consensus + 6 single methods per time point, LR selection |
| 3 | `03_build_dataset.py` | genes, growth, PCA, v1 features → `data.pt` |
| 4 | `04_synthetic_benchmark.py` | ground-truth system with known communication |
| 5–6 | `05_train.py`, `06_run_grid.py` | one run / the experiment plan (`configs/experiments.yaml`) |
| 7 | `07_evaluate.py` | all metrics + statistics |
| 8 | `08_baselines.py` | Waddington-OT, OT-CFM (+ optional official PRESCIENT) |
| 9 | `09_interpret.py` | knockouts, sender ablation, windows, TFs, attributions, bifurcation |
| 10 | `10_cytokine_experiments.py` | clone-matched cytokine evidence, dose-response |
| 11 | `11_neural_tube.py` | neural tube (Cislo et al. PNAS 2025 system) |
| 12 | `12_make_report.py` | one markdown report from all results |

## Models (`configs/models.yaml`)

| preset | description |
|---|---|
| `prescient` | potential on gene state only (re-implementation of PRESCIENT) |
| `v1` | the original CC-Tempo (additive, so communication cannot affect fate; see `docs/ORIGINAL_CODE_AUDIT.md`) |
| `v2_local` | landscape conditioned on the cell's own ligand/receptor expression |
| `v2_meanfield` | **main model**: landscape conditioned on the received population signal |
| `ctrl_off` / `ctrl_shuffled` / `ctrl_random` | same capacity; signal zeroed / receptors re-paired / random genes |

## Repository layout

```
cctempo/            package (models, training, simulation, evaluation, interpretation, baselines, api)
configs/            dataset configs, model presets, experiment plan
scripts/            numbered pipeline
kaggle/             Kaggle notebook + guide
app/                Gradio web tool (Hugging Face Spaces ready)
docs/               derivations, code audit, novelty list, user guide
paper/              response-to-reviewers draft, revised manuscript outline
tests/              CPU smoke tests
work/               (generated) processed data, runs, results
```

## Data

- Weinreb et al., *Science* 2020: https://github.com/AllonKleinLab/paper-data (in vitro + cytokine perturbation)
- Maizels, Snell & Briscoe, *Cell Systems* 2024: GEO GSE236512 (neural tube, used by Cislo et al., *PNAS* 2025)

## Citation

Jony S.S.R., Rahman M.S., Samee M.A.H. *CC-Tempo: a cell-cell communication aware temporal model of
cellular dynamics* (revised, in preparation).
