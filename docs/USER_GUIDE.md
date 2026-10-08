# CC-Tempo user guide (web tool, CLI, Python)

This guide is for biologists and clinicians who want to run in-silico experiments with a trained CC-Tempo model.

## 1. What CC-Tempo predicts
Given starting cells (e.g. Day-2 progenitors), CC-Tempo simulates many possible futures of each cell under
the learned dynamics. It reports the fraction ending in each cell type. You can change:
- **Ligand families (knockout):** the cells no longer respond to that signal (receiver-side block,
  like a receptor-blocking antibody).
- **Exogenous ligands:** add a cytokine to the "medium" at a relative dose (1 = the dose learned from the data).
- **Gene levels:** set the z-scored expression of genes in the starting cells (overexpression / knockdown).

The outputs are model predictions, i.e. hypotheses to test in the lab.

## 2. Web tool
Online: [[Hugging Face Space URL]]. Locally:
```bash
pip install -e ".[app]"
CCT_ROOT=. python app/app.py          # opens http://127.0.0.1:7860
```
Steps: choose a model → choose the starting cell type and number of cells → (optional) tick ligand families,
pick a cytokine and dose, type genes and a value → **Run**. You get a table (unperturbed vs perturbed %),
a bar chart and a CSV of per-cell predictions.

## 3. Command line
```bash
cctempo fate --run work/weinreb/runs/v2_meanfield_s0 --cells 0 1 2 3 --ko CSF --exo G-csf=2.0 --copies 500
```

## 4. Python
```python
from cctempo.api import CCTempoSession
s = CCTempoSession("work/weinreb/runs/v2_meanfield_s0", root=".")
cells = s.start_cells("Undifferentiated")[:200]
base = s.predict_fate(cells)
pert = s.predict_fate(cells, ko_families=["CSF"], exogenous={"G-csf": 2.0}, gene_set=["Irf8"], gene_value=-2.5)
print(pert.mean(numeric_only=True) - base.mean(numeric_only=True))
```

## 5. Your own data
See README "Quick start": provide a log-normalised AnnData with a time column and a cell-type column, run
LIANA and `build_data`, train with `cctempo train`.

## 6. Deploying the web tool on Hugging Face Spaces (free, CPU)
1. Create a Space: SDK **Gradio**, hardware *CPU basic*.
2. Upload: `app/app.py` (as `app.py` at the root, or keep the folder and set `app_file: app/app.py` in the
   Space README header), the `cctempo/` and `configs/` folders, and `app/requirements.txt` as `requirements.txt`.
3. Upload one trained run with its data:
   `work/weinreb/runs/v2_meanfield_s0/{model_best.pt, config_used.yaml}`,
   `work/weinreb/data.pt`, `work/weinreb/classifier_data.pkl`. (`data.pt` is ~30–60 MB; use Git LFS, which
   Spaces support.)
4. Space Settings → Variables: `CCT_ROOT=.`, `MODEL_RUNS=work/*/runs`.
5. The CPU Space is slow for thousands of copies. Defaults (100 cells × 200 copies) take about 10–30 s.

## 7. Limitations to tell users
- Communication is modelled as a population mean (well-mixed culture), not spatial neighbourhoods.
- LR signals are inferred from mRNA of ligands/receptors, not protein activity.
- Gene perturbations act through the PCA subspace. The tool reports how much of the perturbation is retained.
