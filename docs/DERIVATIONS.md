# Derivations for the revised Methods (answers Reviewer 2, Q2–Q5)

Notation: x ∈ ℝ^d is the gene-expression state (PCA coordinates of scaled log-expression),
μ_t is the population distribution at time t, U is a potential (landscape), σ the noise scale.

## 1. Where Eqs. (1)–(2) come from (Reviewer 2, Q2)

**Single cell: overdamped Langevin dynamics.** We follow the gradient-flow view of Waddington's landscape
(Hashimoto et al. 2016; Schiebinger et al. 2019; Yeo et al. 2021 (PRESCIENT); Cislo et al. PNAS 2025):

  dx_t = −∇U(x_t) dt + σ dW_t                                        (1)

W_t is a standard Wiener process. The diffusion constant is D = σ²/2.

**Population: Fokker–Planck equation.** If every cell follows (1), the density p(x,t) of the population obeys

  ∂_t p = ∇·(p ∇U) + D Δp.                                           (FPE)

**Why fitting snapshots is well-posed: gradient flow in Wasserstein space.** Jordan, Kinderlehrer & Otto (1998)
showed that (FPE) is the gradient flow of the free energy F(p) = ∫U dp + D ∫p log p in the 2-Wasserstein
metric. That is why we fit U by matching simulated and observed snapshots with a Wasserstein-type
(Sinkhorn) loss.

**Adding communication: a McKean–Vlasov (mean-field) SDE.** CC-Tempo v2 lets the landscape of cell i depend
on the signal it receives from the population:

  dx_i = −∇_x U(x_i ; s_i(t)) dt + σ dW_i,
  s_ik(t) = g_k · ( α L_k(μ_t) + e_k(t) − L_k^ref ) · r_k(x_i) / c_k,
  L_k(μ_t) = ∫ l_k(y) w_t(y) μ_t(dy) / ∫ w_t dμ_t                   (2)

- l_k and r_k are the ligand and receptor expression of LR pair k. They are decoded from x, and a complex is the soft-min of its subunits, as in LIANA/CellPhoneDB.
- w_t is the growth weight.
- e_k(t) is an exogenous ligand (cytokine in the medium, SAG).
- g_k ∈ (0,1) is a learned gate initialised from the LIANA consensus prior.
- L_k^ref is the field of the observed start population (reference). Centring is no loss of
  generality, because the term g_k L_k^ref r_k(x) depends on x only and is absorbed by Φ. It
  matters for identifiability: a channel can affect dynamics only through *changes* in the ligand
  field (over time, with population composition, or with the medium), so it cannot serve as a hidden
  description of the cell's own state. A knockout (g_k = 0) then means "clamp channel k at its
  reference level", an input the model has seen. Without centring, the first synthetic benchmark
  showed decoy channels with knockout effects larger than the true channel.
- c_k is a fixed calibration constant: c_k = (max_t |L_k(μ_t^obs) − L_k^ref| ∨ 0.25 L_k^ref) · mean r_k.
  The received signal is softly bounded, s ← 3 tanh(s/3) (identity for |s| ≪ 3). Without the floor and the
  bound, channels whose observed field barely changes got c_k ≈ 10⁻³, and small drifts of the simulated
  population became signals of ~100 that fed back through the field and diverged (Weinreb tier 1, first run).

Because L_k depends on μ_t, (2) is a McKean–Vlasov SDE. As N → ∞ the interacting system becomes a
nonlinear FPE: ∂_t p = ∇·(p ∇_x U(x; s[x, p])) + D Δp.

**Biology of (2) (Reviewer 1(b)).** A cell's dynamics are changed by ligands produced by *other* cells.
In the erythroblastic-island example, macrophage-derived ligand tilts the erythroblast landscape. The
communication between the erythroblast and its own descendants is irrelevant. That is what (2) encodes:
the signal enters the receiver's landscape, and the senders enter only through the population field L_k.

The landscape is U(x; s) = Φ(x) + U_c(x; s) with U_c(x; s) = wᵀ[h(x) ⊙ (1 + tanh(Γs)) + Bs] + sᵀAx.
The term sᵀAx is a signal-dependent linear tilt, exactly the role of the morphogen parameters (a, b) in
Cislo et al. (PNAS 2025, Eq. 7). Γ and B reshape the valleys. With s = 0, U reduces to an ordinary
potential, so the 'CC-off' control has identical capacity.

**Why the original Eq. (2) could not work.** The v1 equations were dX = ∇_X(Φ(X) + Ψ(Y)) dt + noise and
dY = ∇_Y(Φ(X) + Ψ(Y)) dt. Because the potential is a *sum* of a function of X and a function of Y,
∇_X(Φ + Ψ) = ∇Φ(X), so the communication state never changes the gene-expression trajectory. In addition,
Ψ was linear, so ∇Ψ was constant. See ORIGINAL_CODE_AUDIT.md.

## 2. Discretisation, Eq. (5) (Reviewer 2, Q3)

Euler–Maruyama for (1) with step Δt:

  x_{t+Δt} = x_t − ∇U(x_t) Δt + σ √Δt z,   z ~ N(0, I).

The Brownian increment W_{t+Δt} − W_t has variance Δt, so its standard deviation is √Δt. **Δt must stay
inside the square root.** Writing the noise as √(2DΔt) z (D = σ²/2) is the same expression. Moving Δt out
would make the noise variance scale as Δt², and the scheme would converge to a deterministic ODE as
Δt → 0 instead of to (1). The reply to the reviewer should say this politely and add the sentence above
to the Methods.

## 3. Birth/death scores (Reviewer 2, Q4: why the mean of z-scores?)

For a gene set S (KEGG cell cycle / apoptosis) the score of cell i is s_i = (1/|S|) Σ_{g∈S} z_ig, where
z_ig is gene g's expression standardised across cells. We use z-scores rather than raw means because:

1. Each gene contributes on the same scale, so a few highly expressed genes cannot dominate.
2. Genes with low detection still carry signal through their relative variation.
3. This is the standard module-score construction (Seurat AddModuleScore, scanpy score_genes) and the one PRESCIENT uses.

The revision should also report the control-gene-corrected `scanpy.tl.score_genes` variant as a
robustness check. `cctempo/growth.py` implements both.

## 4. Smoothing, Eq. (11) corrected (Reviewer 2, Q5)

Let A be the row-normalised adjacency of the k-nearest-neighbour graph (k = 15, in PCA space) and s⁽⁰⁾ the
raw scores. We iterate

  s⁽ⁿ⁺¹⁾ = (1 − a) s⁽⁰⁾ + a A s⁽ⁿ⁾,   0 < a < 1 (a = 0.5),

which is personalised graph diffusion (label spreading). The spectral radius of aA is a < 1, so the
iteration converges geometrically to s* = (1 − a)(I − aA)⁻¹ s⁽⁰⁾, whatever the starting point. We use
20 iterations, which leaves error ≤ 0.5²⁰ ≈ 10⁻⁶. Birth/death rates are then logistic functions of the
smoothed scores (Eqs. 12–13), and the per-cell growth rate is g = b − d. A source cell carries mass
∝ exp(g Δt) in the unbalanced transport loss.

## 5. Loss (Eq. 7) and the growth-weighted Sinkhorn divergence

For each observed later time t_k, simulate a batch from t_0 with source weights a_i ∝ exp(g_i (t_k − t_0)) and
minimise

  Σ_k S_ε( Σ_i a_i δ_{x̂_i(t_k)} , (1/m) Σ_j δ_{x_j(t_k)} ) + λ Σ_k g_k,

where S_ε is the debiased Sinkhorn divergence (Feydy et al. 2019), with cost |x − y|²/2 and blur ε^{1/2} = 0.1.
The L1 term on the gates makes the model use few LR channels, which keeps them interpretable.

## 6. Path-integral operator (from Cislo et al. PNAS 2025, adapted)

On N sampled cells ("discrete manifold"), the short-time solution of (FPE) gives the transition matrix

  T(x_i ← x_j) ∝ V(x_i) exp( −|x_i − x_j − τ v(x_j)|² / (4Dτ) ),   Σ_i T(x_i ← x_j) = 1.

Here v is CC-Tempo's learned drift (time-dependent through the mean field) and V(x_i) ≈ 1 / local density.
Our adaptations, stated as such in the paper:

1. The kernel is restricted to the k nearest neighbours of the landing point x_j + τv(x_j), with an adaptive bandwidth h_j = max(σ√τ, distance to the (k/2)-th neighbour). An isotropic Gaussian alone is degenerate in 20–50 dimensions.
2. T is time-inhomogeneous because the ligand field changes.

Uses:
- Forward evolution P_{k+1} = T_k P_k gives KL(observed‖simulated) in bits, the PNAS metric.
- The backward Kolmogorov recursion h_k = T_kᵀ h_{k+1} gives fate probabilities (committors) without Monte-Carlo noise.
- The stationary distribution π of a frozen T gives an effective potential −log π, attractors and basins.

## 7. Communication bifurcation analysis

Scale the endogenous field by α (communication strength), freeze it, and compute π(α), attractors and basin
masses. Plotting the number of attractors and basin masses against α gives an endogenous analogue of the
morphogen-dose bifurcation diagram of Cislo et al. (Fig. 3E). Here the "dose" is produced by the cells
themselves.
