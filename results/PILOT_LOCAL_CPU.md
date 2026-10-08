# Local results so far (no GPU training)

Everything here was run on the laptop **CPU**, as code validation only. None of it is a paper result
except §1, which is pure data analysis and final as is.

## 1. Cytokine perturbation: clone-matched evidence (FINAL, data only)

Source: Weinreb et al. 2020, Experiment 3 (metadata + clone matrix). Script: `10_cytokine_experiments.py --evidence`.
Files: `work/cytokine/results/evidence_clone_matched.csv`, `evidence_composition.csv`.

Same Day-2 progenitor clones, descendants grown in different media. Neutrophil fate bias = (n_Neu+1)/(n_Neu+n_Mo+2):

| medium A | medium B | shared progenitors | bias A | bias B | Wilcoxon p |
|---|---|---|---|---|---|
| M-CSF | G-CSF | 187 | 0.223 | 0.411 | < 1e-4 |
| full cocktail | M-CSF | 171 | 0.334 | 0.208 | < 1e-4 |
| M-CSF | SCF | 60 | 0.263 | 0.467 | < 1e-4 |
| M-CSF | EPO | 45 | 0.253 | 0.431 | < 1e-4 |
| full cocktail | G-CSF | 106 | 0.428 | 0.439 | 0.37 |

Day-6 composition: M-CSF 91% monocytes; G-CSF 46% neutrophils; EPO 57% erythroid + 12% Mk; full
cocktail 44% Mo / 11% Neu / 13% Ba. **Conclusion:** changing the ligands a clone receives changes what it becomes.
This is direct, lineage-traced evidence for Reviewer 1(a).

## 2. Synthetic ground truth: pilot (1 seed, 200 epochs, 1,000 cells; NOT final)

True system: heteroclinic-flip landscape with communication feedback β = 1.5. Trained on doses 0 and 1;
dose 0.5 held out. Fate = P(A) at t = 3.

| model | dose 0 | dose 0.5 (held out) | dose 1 | best val loss |
|---|---|---|---|---|
| truth | 0.455 | 0.126 | 0.032 | – |
| PRESCIENT | 0.333 | 0.333 | 0.333 | 12.7 |
| CC-off (same capacity) | 0.261 | 0.261 | 0.261 | 12.1 |
| **v2 mean-field** | 0.294 | 0.058 | 0.029 | **4.9** |

Autonomous models cannot represent the dose effect. v2 follows it in the right direction but
under-predicts the held-out dose after only 200 epochs. The full run (Phase 2: 400 epochs, 3 seeds, β = 0
control) is needed for the paper.

## 3. Path-integral operator: design decision from the pilot

The analytic short-time kernel of the PNAS paper, applied on the data point cloud, kept 94–97% of the
probability in the progenitor state, while Monte-Carlo simulation of the same model gave 25%. Cause: in
10–50 dimensions, model trajectories leave the data manifold along low-variance PCs, so they cannot be
mapped back faithfully, and one drift step is shorter than the spacing between points. Fix (now the
default, `build_operator`): a Markov state model whose nodes are the model's own simulated states, with
columns of T estimated from short SDE paths. It reproduces Monte-Carlo (A/B/Prog = 0.17/0.55/0.28 vs MC
0.18/0.57/0.25). The paper should state this as an adaptation of the PNAS idea, not as the PNAS kernel.

## 4. Integration test

Fake Weinreb-like data (3,000 cells, real mouse LR genes, clones) through every script:
LIANA (consensus + CellPhoneDB) → dataset → PRESCIENT / v1 / v2 training → evaluation + statistics →
WOT / OT-CFM baselines → interpretation (knockouts, sender ablation, windows, TF perturbation,
attributions, bifurcation) → API → Gradio app → report. All ran without errors after fixes. The fixes
included: LIANA cannot score a single-type time point, so Day-2 progenitors are now split into
subclusters; and YAML parses `off` as False.
