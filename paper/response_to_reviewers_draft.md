# Response to reviewers: DRAFT (fill every [[...]] from results/RESULTS_REPORT.md)

We thank the reviewers for their careful reading. Their comments led to a substantially revised model,
four datasets instead of one, five additional baselines, new evaluation indexes, a public software
package with a web tool, and a reorganised manuscript. Changes in the manuscript are marked in blue.

---

## Reviewer 1

**(a) Is there strong evidence that cell-cell communication contributes directly to cellular dynamics?**

We agree that this needs direct evidence, and we now provide it at three levels.

1. *Experimental, lineage-traced evidence (new Fig. 2A–C).* Weinreb et al. (2020) split barcoded clones
   across culture media that differ only in the cytokines (ligands) supplied. The *same* progenitor clones
   differ in fate depending on the ligands they receive. Neutrophil fate bias is 0.22 in M-CSF vs 0.41 in
   G-CSF (187 shared progenitors, Wilcoxon p < 1e-4), and 0.21 vs 0.33 for M-CSF vs the full cocktail
   (171 progenitors, p < 1e-4). Day-6 composition shifts from 91% monocytes (M-CSF) to 46% neutrophils
   (G-CSF) and 57% erythroid cells (EPO).
2. *Controlled ground truth (new Fig. 3).* In a simulated system with a known ligand-receptor feedback,
   communication-unaware models (PRESCIENT, and our model with the signal switched off) cannot reproduce
   the dose-dependent fate shift. CC-Tempo recovers it and identifies the true LR channel among decoys
   ([[knockout effect true vs decoy]]). With the feedback removed (β = 0), CC-Tempo reports no
   communication effect ([[...]]).
3. *Model-based, capacity-controlled evidence on real data (new Fig. 4).* CC-Tempo is compared with
   identical-capacity controls in which the signal is set to zero, receptors are re-paired at random, or
   the LR genes are replaced by random genes. [[PCC / AUROC / bootstrap ΔPCC with CI and Holm-adjusted p]].

We also link communication to transcription factors, as the reviewer suggests. Knocking out the top
ligand family changes the activity of [[TFs, e.g. Irf8/Cebpe]] in the simulated descendants (new Fig. 5D),
i.e. communication acts upstream of the TF programmes that drive fate.

**(b) Communication acts indirectly (macrophage → erythroblast example). How can it improve inference of dynamics?**

This example sharpened our formulation. In the revised model (Methods, Eq. 2), a cell's landscape depends
on the ligand field produced by the *rest of the population*, which may be other lineages (e.g.
macrophages), times its own receptor expression. Communication between an erythroblast and its own
descendants plays no special role. Mathematically this is a mean-field (McKean–Vlasov) stochastic
differential equation: the senders change the receivers' landscape. We also added an in-silico experiment
that mirrors the reviewer's example: silencing the ligands produced by one lineage and measuring the change
in the fates of progenitors of other lineages (new Fig. 5B, [[result]]).

We also re-examined the original implementation. In it, the communication potential was added to the
gene-expression potential, so it could not alter gene-expression trajectories. We now state this openly,
keep the original formulation as a baseline ("CC-Tempo v1"), and show that the coupled formulation is
required ([[v1 vs v2 numbers]]).

**(c) A combined score cannot reflect specific LR functions; different CCC methods should have different effects.**

- The model now has one channel per ligand-receptor pair (64 pairs) with its own learned gate, initialised
  from a prior and penalised for sparsity. Effects are reported per pair and per ligand family
  (knockout scan, new Fig. 5A).
- We replaced CellChat with LIANA, which runs [[7]] methods on the same data. CC-Tempo trained with each
  method's LR selection gives [[PCC range]]. The top-ranked families overlap [[Jaccard]] (new Suppl. Table).
  The conclusions do not depend on one inference method.

**More datasets, methods and evaluation indexes.**
- Datasets: (i) mouse hematopoiesis in vitro (Weinreb 2020), (ii) cytokine perturbation with clones split
  across media (Weinreb 2020, Exp. 3), (iii) in vitro neural tube time course (Maizels et al. 2024; the
  system recently modelled by Cislo et al., PNAS 2025), (iv) a synthetic system with known communication.
- Methods: PRESCIENT, CC-Tempo v1, Waddington-OT, OT conditional flow matching (ODE and SDE)
  [[+ official PRESCIENT / CellRank 2 if run]].
- Indexes: Pearson, Spearman, AUROC, AUPRC, MAE, RMSE, fate deviation; Wasserstein-2, MMD, energy distance;
  cell-type composition L1/JS; KL divergence in bits; held-out time point and held-out condition
  generalisation. Statistics: seed-level tests and a progenitor-level paired bootstrap, Holm-corrected.

## Reviewer 2

1. *Results section reads like Methods.* The model description now sits in Methods. Results begin with the
   evidence and benchmark findings, and discussion has been moved to the Discussion.
2. *Derivation of Eqs. (1)–(2).* We added a derivation: overdamped Langevin dynamics → Fokker–Planck →
   Wasserstein gradient flow (Jordan–Kinderlehrer–Otto), and its mean-field extension for communication
   (Methods, "Model"; Supplementary Note 1).
3. *Δt in √(2σ²Δt).* We respectfully keep Δt inside the square root. Brownian increments over Δt have
   variance Δt, so the Euler–Maruyama noise term is σ√Δt·z = √(2DΔt)·z. With Δt outside, the noise variance
   would scale as Δt² and the scheme would not converge to the SDE. We added this explanation.
4. *Mean of z-scores for birth/death.* Z-scoring puts every gene on the same scale, so a few highly
   expressed genes cannot dominate. This is the standard module-score construction (Seurat
   AddModuleScore, scanpy score_genes) used by PRESCIENT. A control-gene-corrected score gives
   [[similar results]] (Suppl.).
5. *Eq. (11).* Corrected: s^(n+1) = (1−a)s^(0) + aAs^(n), with A the row-normalised kNN adjacency. It
   converges geometrically to (1−a)(I−aA)^(−1)s^(0) because the spectral radius of aA is a < 1. We added
   this statement.
6. *R commands.* No R is used any more (Scanpy + LIANA in Python). Code snippets are in the Supplement and
   the repository.

## Reviewer 3
- The repository is complete: README, installation, quick start, tests, numbered reproduction scripts,
  configs and a user guide ([[GitHub URL + Zenodo DOI]]).
- The manuscript was edited by [[a native speaker / professional service]].

## Reviewer 4
- CC-Tempo is now a pip-installable package with a command-line interface, a Python API and a web
  application ([[Hugging Face Space URL]]). Users choose a trained model and starting cells, then knock
  out ligand families, add cytokines or set gene levels, and obtain predicted fate distributions. A user
  guide is provided (Supplementary Note [[x]]; docs/USER_GUIDE.md).
- Researchers can use it to (i) rank signalling channels for a fate of interest, (ii) design media
  (cytokine dose and timing) for directed differentiation, and (iii) generate testable hypotheses
  (e.g. which lineage's ligands support a given progenitor fate).
