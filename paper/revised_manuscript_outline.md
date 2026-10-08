# Revised manuscript outline

Working title: **CC-Tempo: a communication-aware, mean-field model of cellular dynamics**
(keep the CC-Tempo name for continuity). Target length is about 7,000 words plus 5 main figures.

## Abstract (≈200 words)
Problem (snapshots, autonomy assumption) → model (mean-field landscape driven by LR signals, LIANA prior)
→ evidence (clone-matched cytokine data; synthetic ground truth; controls) → benchmark (4 datasets, 6+
methods, N indexes) → interpretation (causal LR ranking, sender silencing, windows, LR→TF) → tool.

## Introduction (≈900 words)
1. Fate decisions and the landscape picture (Waddington; Cislo et al. 2025 give a rigorous landscape from data).
2. Temporal models (WOT, PRESCIENT, TIGON, flow matching) all treat cells as autonomous.
3. Biology says otherwise. Use the reviewer's erythroblastic-island example and cytokine-driven
   haematopoiesis: the signal comes from *other* cells, so communication is an indirect, population-level effect.
4. Gap: no temporal model in which the landscape of a cell depends on the population's ligands.
5. Contributions (bullets, numbered to match the figures).

## Results (results only; no methods prose)
1. **Ligand signals redirect clonal fates in lineage-traced data** (Fig. 1: schematic of the idea + clone-matched
   cytokine evidence + compositions per medium).
2. **CC-Tempo: a landscape tilted by the population's signals** (Fig. 2: model schematic, one paragraph;
   synthetic ground truth: only coupled models follow the dose; true channel found; β = 0 control silent).
3. **Communication improves fate prediction in haematopoiesis beyond capacity-matched controls**
   (Fig. 3: PCC/AUROC/… vs PRESCIENT, v1, ctrl-off/shuffled/random, WOT, OT-CFM; bootstrap CIs; held-out
   Day 4; robustness across LIANA methods).
4. **Predicting unseen culture conditions** (Fig. 4: leave-one-medium-out on the cytokine data; dose-response;
   autonomous models cannot respond).
5. **Which signals, from whom, when, and through which TFs** (Fig. 5: LR knockout ranking, sender-lineage
   silencing, temporal windows, LR→TF, progenitor genes validated on clone bias).
6. **Landscape geometry and communication bifurcations; neural tube** (Fig. 6 or Suppl.: attractors vs α;
   endogenous Shh hypothesis for the late p3→FP drift; KL in bits).

## Discussion (≈1,000 words)
What communication adds and when it does not (be honest if the in-vitro hematopoiesis gain is small);
relation to PNAS 2025 (morphogen-dose landscapes vs endogenous signal landscapes; MSM operator);
limitations (mean field = well-mixed culture, no space; LR expression ≠ protein activity; PCA subspace
limits gene perturbations; the v1 additive formulation and why it was abandoned); outlook (spatial
neighbourhood field, optimal control of media, reprogramming).

## Methods
- Data and preprocessing (Scanpy); lineage-tracing ground truth; growth rates (Eqs. with the corrected smoothing).
- Communication prior with LIANA (methods, resource, per-time-point, subclusters for single-type time points).
- Model: Langevin → FPE → mean-field SDE (derivation in Suppl. Note 1); landscape parameterisation;
  LR read-out; exogenous ligands.
- Training: growth-weighted Sinkhorn loss, optimisation, model selection on held-out cells.
- Evaluation: fate bias (MC), MSM operator, metrics list, statistics.
- Interpretation and in-silico experiments: knockouts with common random numbers, sender silencing,
  windows, attributions, TF activities, bifurcation.
- Baselines and their settings.
- Software, hardware (Kaggle T4), runtime; code and data availability.

## Supplementary
Notes: derivations; the v1 additive-potential issue; path-integral operator adaptation.
Tables: hyperparameters; per-seed results; LIANA method overlap; gene lists; TF lists.
Figures: per-method robustness; additional knockouts; growth-score robustness; app screenshots + user guide.
