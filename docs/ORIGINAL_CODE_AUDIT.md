# Audit of the original CC-Tempo code (github.com/SaminRK/CC-Tempo)

Read on 2026-10-07. File references are to that repository. This affects what the revised paper can claim,
so read it before writing the response letter.

## Finding 1 (critical): the communication branch cannot influence fate

`src/models.py`:

```python
def _pot(self, x, y):
    return self.net(x) + self.cc(y)          # additive, separable
def _drift(self, x, y):
    pot = self._pot(x_, y_)
    drift1, drift2 = torch.autograd.grad(pot, (x_, y_), ...)
```

- `drift1 = ∂/∂x [net(x) + cc(y)] = ∇net(x)`. This does **not** depend on y, the communication score.
  Gene-expression trajectories are therefore exactly the trajectories of a PRESCIENT-type model with
  potential `net`.
- Fate is classified from x alone (`evaluate.py`: `lr.predict(x_i)`), so communication cannot change any
  predicted fate, fate bias or PCC.
- `self.cc = Linear(c, k) → Linear(k, 1)` has **no nonlinearity**, so cc(y) is linear and ∇_y cc is a
  constant vector. The y-dynamics are a constant drift, `y(t) = y(0) + c t`.

**Consequences:**
- (a) The 0.50 → 0.52 PCC gain cannot be caused by communication. Plausible sources are the global
  gradient-norm clipping, which couples the parameter updates of `net` and `cc`, random-number consumption,
  and run-to-run variance.
- (b) The pathway-dropping result (top-5/top-10 removal lowering PCC) and the pathway SHAP ranking
  measure something other than a causal influence of communication on fate.
- (c) The pathway perturbation experiment changed the ligand/receptor *gene expression* inside x at the
  same time (Methods: "...as well as the gene expression of the ligand-receptor pairs"). Any fate change
  there came from the gene perturbation, not from the communication channel.

This is exactly the weakness Reviewer 1(a)/(b) sensed. The revision fixes it by design:
- v2-local and v2-meanfield make U depend on x *and* the received signal non-additively (FiLM + tilt).
- Controls with identical capacity separate "communication" from "more parameters": ctrl_off,
  ctrl_shuffled, ctrl_random.
- v1 is kept as a faithful re-implementation (`mode: v1`) so the paper can show the issue transparently.

## Finding 2: other inconsistencies to fix in the text

| Topic | Paper | Code / correct statement |
|---|---|---|
| copies per progenitor | Results: 100; Methods: 2000 | `evaluate.py` NS=100. v2 uses 2000 everywhere (`eval.n_copies`). |
| incoming/outgoing pairing | Results: incoming × ligand | Methods/code: incoming × receptor (biologically right) |
| "without reducing to low dimension" | Results | Methods: PCA(50) as a fixed layer. Say that. |
| AUROC | Methods says computed | never reported. v2 reports AUROC/AUPRC/Spearman/MAE. |
| significance level | 5% genes, 1% pathways | use one level + Holm correction (`evaluate/stats.py`) |
| Cstb vs Ctsb | both appear | check which one SHAP returned |
| "TF" lists | many are markers (S100a8, Ltf, Lcn2...) | v2 uses true TFs (genesets.TFS) and keeps the marker sets labelled as markers |
| 1110002J07Rik etc. | interpreted | v2 validates top genes against **lineage-traced** clone bias (gene_de_validation.csv) |
| README | empty (Reviewer 3) | full README + user guide + tests |
| evaluate.py | imports `prescient.train.AutoGenerator` | v2 evaluates every model with one code path |

## Finding 3: what in v1 is still useful
- The benchmark definition (Day-2 undifferentiated cells in clones with ≥1 Neu/Mono at Day 4/6, bias =
  (n+1)/(n+m+2)). v2 reproduces it exactly (`data/weinreb.py`): 1,461 progenitors.
- The interpretation programme (genes at Day 2/4/6, perturbations). v2 extends it with fate-level
  attribution through the SDE and paired (common-random-number) effect sizes.
