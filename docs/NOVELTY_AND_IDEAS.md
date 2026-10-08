# What is new in CC-Tempo v2, and why it answers the reviews

Each idea lists: what it is, why it matters, where it lives, and its status.

## A. Model ideas

1. **Mean-field (McKean–Vlasov) communication.** A cell's landscape is tilted by the ligand field produced by
   the whole population, L_k(μ_t), times its own receptor level. This makes "communication is an indirect
   effect" (Reviewer 1b) part of the model: senders shape receivers' landscapes. Mathematically, it is an
   interacting-particle Wasserstein gradient flow. → `models/cctempo.py` (mode `meanfield`), `docs/DERIVATIONS.md §1`.
2. **Signal-conditioned landscape (FiLM + linear tilt).** Received signals reshape valleys and tilt the
   landscape, the data-driven analogue of the morphogen parameters (a, b) in Cislo et al. PNAS 2025.
   With s = 0 it reduces to an ordinary landscape. → `models/potentials.py`.
3. **Unified endogenous + exogenous ligand field.** The same LR channel carries ligand made by cells and
   ligand added to the medium (cytokines, SAG), each with one learned dose parameter (a "few knobs" idea
   borrowed from PNAS). This enables zero-shot / leave-one-condition-out prediction of media changes.
4. **LR-specific gates with a LIANA consensus prior + L1.** Answers Reviewer 1(c) ("a combined score cannot
   reflect specific LR functions"): every LR pair has its own channel and gate, and the model must choose few.
5. **Differentiable LR read-out from the state.** Soft-min over complex subunits, decoded from PCA, so the
   signal changes as cells move. In v1 the scores were static per cell.

## B. Evidence / evaluation ideas

6. **Clone-matched cytokine evidence (data only).** The same clones in M-CSF vs G-CSF media differ in fate
   (p < 1e-4). Real, lineage-traced proof that ligands steer fate (Reviewer 1a). DONE (results/PILOT_LOCAL_CPU.md).
7. **Synthetic ground truth with known communication** (heteroclinic flip + ligand feedback), with a β = 0
   control to show the model does not invent communication. → `data/synthetic.py`, script 04.
8. **Capacity-matched negative controls:** signal off, receptors re-paired, random genes. They separate
   "communication" from "more parameters", which v1 could not do.
9. **Robustness to the CCC method** (Reviewer 1c, "inference methods should have different effects"):
   LIANA consensus vs CellPhoneDB, CellChat, NATMI, Connectome, SingleCellSignalR, geometric mean. Report
   top-K overlap and fate accuracy per method.
10. **More indexes** (Reviewer 1): PCC, masked PCC, Spearman, AUROC, AUPRC, MAE, RMSE, fate deviation,
    W2, MMD, energy distance, composition L1/JS, KL in bits, held-out-time and held-out-condition
    generalisation, plus seed-level tests and progenitor-level paired bootstrap with Holm correction.
11. **More datasets and methods:** 4 datasets (Weinreb in vitro, Weinreb cytokine, neural tube, synthetic);
    baselines PRESCIENT, v1, WOT, OT-CFM, OT-CFM-SDE (+ optional official PRESCIENT, CellRank 2).

## C. Interpretation ideas

12. **Causal LR ranking by knockout with common random numbers:** paired effect ± SE per LR pair / family.
    Replaces SHAP on static scores, which in v1 had no causal path to fate.
13. **Sender-lineage silencing:** remove the ligands made by cells currently classified as type T, and
    measure the effect on progenitor fates. This is the in-silico version of the macrophage → erythroblast
    example of Reviewer 1(b).
14. **Temporal blocking windows (window of opportunity / order effects):** block a family in D2–4 vs D4–6.
    Mirrors the SAG up/down/pulse experiments of PNAS 2025 for endogenous signals.
15. **LR → TF → fate chain:** CollecTRI TF activities of simulated cells with a channel on vs knocked out.
    Answers Reviewer 1(a), "dynamics depend on key TFs": communication acts upstream of TF programmes.
16. **Fate attribution through the SDE:** integrated gradients of the end-state fate logit w.r.t.
    progenitor genes, validated against lineage-traced clone bias rather than predicted labels.
17. **Markov-state path-integral operator on the model's own states:** deterministic committors (fate
    probabilities), attractors, basins and an effective potential. It matches Monte-Carlo (validated).
18. **Communication bifurcation diagram:** number of attractors and basin masses vs communication
    strength α. This is an endogenous analogue of the morphogen-dose bifurcation diagram in PNAS 2025.
19. **Neural-tube hypothesis test:** PNAS needed a time-changing landscape to explain the late p3 → floor
    plate drift. We test whether endogenous Shh from floor-plate cells explains it (knock out only the
    endogenous Hedgehog channel and keep SAG). Positive or negative, it is a clean biological statement.

## D. Tool ideas (Reviewer 4)
20. pip package + CLI + Python API + Gradio web app (Hugging Face Spaces) + user guide + tests.

## Further ideas (only if time remains)
- Conformal prediction intervals for fate bias (seed ensemble as a base).
- Spatial extension: replace the population mean with a neighbourhood mean on spatial data (e.g. MOSTA);
  the code already has `field_weight_fn` hooks.
- Learn a sender-specific field (type-resolved L_k) instead of a single population mean.
- Optimal control: find the cheapest ligand schedule e(t) that maximises a target fate (gradient through
  the simulator). This is the "reprogramming factors" promise of the abstract, made computational.
