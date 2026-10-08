# CC-Tempo: project status (everything in one place)

Last updated: 2026-10-08. Update this file whenever a phase of TODO_PLAN.md finishes.

---

## 1. The project in five lines
- **Paper:** "CC-Tempo: A cell-cell communication aware temporal model of cellular dynamics", by S.S.R. Jony,
  M.S. Rahman (BUET) and M.A.H. Samee (Baylor). Submitted to *Computational and Structural Biotechnology
  Journal*. Four reviews came back; Reviewer 1 said "does not merit publication in current version" and
  Reviewer 4 asked for "a significantly revised version".
- **Idea:** model differentiation as cells rolling down a learned landscape (as in PRESCIENT), plus cell-cell
  communication.
- **Problem found now:** in the original code, communication cannot affect fate (§4). Reviewer 1 sensed this.
- **Revision (v2):** a new mean-field communication model, real evidence (cytokine clone data), synthetic
  ground truth, 4 datasets, 5+ baselines, many metrics, a tool. All in `CC-Tempo-v2/`.
- **Status:** all code is written, tested on CPU and pushed to https://github.com/Ahnaf-Akib-Tanim/cc-tempo_v2.
  GPU training on Kaggle has not started (next: Phase 0.2 / STAGE `A` of TODO_PLAN.md).

## 2. What is in `C:\Buet\CSE 400`
| item | what it is |
|---|---|
| `all_in_one.pdf` | the submitted manuscript (50 pp. incl. figures and supplementary tables) |
| `all in one_Guide.pdf` | a study guide to the paper. Its Part 12 "Reading critically" lists inconsistencies (used in §4) |
| `CLAUDE_GEN__CC-Tempo_Explained.pdf` | plain-language explainer of the paper |
| `CC_Tempo_reviews.pdf` | the four reviews (summarised in §5) |
| `2nd paper/2nd paper.pdf` | Cislo, Delás, Briscoe & Siggia, "Reconstructing Waddington's landscape from data", *PNAS* 122(49) e2521762122, Dec 2025 |
| `2nd paper/…Summary.pdf` | plain-language summary of the PNAS paper |
| `predefense jony vai/` | Jony's MS pre-defence slides (PDF), video and HTML explainer |
| `Links.txt` | GitHub of v1 (github.com/SaminRK/CC-Tempo), PNAS paper + code, datasets |
| **`CC-Tempo-v2/`** | **the new work (this folder)** |

## 3. The original paper (v1) in brief
- **Data:** Weinreb et al. 2020 LARRY in vitro haematopoiesis, 130,887 cells, Days 2/4/6, 5,864 clones.
- **Model:** PRESCIENT-style potential network on 50 PCs + a "communication potential" on 58 static
  features (29 CellChat pathways × sending/receiving; type-level CellChat probability × Seurat module score).
  Training used a Sinkhorn loss, 500 + 2,500 epochs, on a Colab A100.
- **Benchmark:** clonal fate bias, Neutrophil/(Neutrophil + Monocyte) for Day-2 progenitors. PCC 0.52 vs
  PRESCIENT 0.50 over 100 trainings (p = 7e-19). Dropping the top-5 pathways gave 0.512 (n.s.); dropping
  the top-10 gave 0.50.
- **Interpretation:** SHAP genes at Day 2/4/6, "TF" perturbations, pathway SHAP (CD34 and SELL top), pathway
  perturbation, ranking stability.

## 4. Audit of the v1 code (details: `docs/ORIGINAL_CODE_AUDIT.md`)
- U = net(x) + cc(y) is additive, so the gene drift ∇ₓU does not depend on communication, and fate
  (classified from x) cannot depend on it. cc was Linear→Linear, so its gradient was constant.
  ⇒ the 0.50→0.52 gain is not attributable to communication. The pathway perturbation also changed
  ligand/receptor *genes* inside x.
- Text inconsistencies to fix: 100 vs 2000 copies; incoming×ligand vs incoming×receptor; AUROC promised
  but not reported; 5% vs 1% thresholds; Cstb/Ctsb; "TF" lists that are mostly markers; empty README.

## 5. Reviews → what we did (full draft reply: `paper/response_to_reviewers_draft.md`)
| reviewer point | answer in v2 | where |
|---|---|---|
| R1a: evidence that communication matters | clone-matched cytokine data (DONE); synthetic ground truth; capacity-matched controls; LR→TF link | `scripts/10`, `04`, `09`; `results/PILOT_LOCAL_CPU.md` |
| R1b: communication is indirect (macrophage example) | mean-field model: senders shape receivers' landscapes; sender-lineage silencing experiment | `models/cctempo.py`, `interpret/perturb.py` |
| R1c: one combined score; methods differ | 64 LR-pair channels with gates; LIANA consensus + 6 methods; robustness runs | `ccc/liana_prior.py`, experiments tier2 |
| R1: more datasets/methods/indexes | 4 datasets; WOT, OT-CFM(-SDE), v1, PRESCIENT (+ optional official PRESCIENT); ~15 metrics + stats | `evaluate/`, `baselines/` |
| R2.1: Results read like Methods | new outline | `paper/revised_manuscript_outline.md` |
| R2.2–2.5: Eqs. 1, 2, 5, 11; z-scores | derivations; corrected Eq. 11; Δt stays inside the √ (with reason) | `docs/DERIVATIONS.md`, `growth.py` |
| R2.6: R code in Methods | no R any more; snippets go to Supplement | – |
| R3: README empty; language | full README, guides, tests; professional proofreading still TODO | `README.md` |
| R4: tool / webserver | pip package, CLI, API, Gradio app (HF Spaces) + user guide | `cctempo/api.py`, `cli.py`, `app/`, `docs/USER_GUIDE.md` |

## 6. The PNAS paper (Cislo et al. 2025) and how v2 uses it
They fit landscapes U = U_base + U_interp with a few parameters (valley/saddle heights, g, D) to flow-cytometry
and RNA data of mESC→ventral neural tube under SAG doses. They use a point-cloud path-integral transition
matrix (Fokker–Planck on the data), KL-in-bits fit quality, attractors/basins via the backward Kolmogorov
equation, linear interpolation of landscapes between doses, and non-commuting dose protocols. They model
*isolated cells in a dish*; their limitations note that "the embryo has cell-to-cell signalling".

**What v2 takes:**
- the landscape geometry tools (attractors, basins, bifurcation diagrams, KL bits) via a path-integral /
  Markov-state operator;
- the "signal-dependent landscape" idea (our FiLM + tilt);
- the dose/timing experiments (our temporal windows and exogenous dose-response);
- their RNA system (GSE236512) as a dataset;
- their heteroclinic-flip landscape as the synthetic ground truth.

**What v2 adds:** the signal is produced by the cells themselves (mean field), at single-LR resolution,
learned in 30–50 dimensions. We also test their open question: is the late p3→floor-plate drift caused by
endogenous Shh?

**Important adaptation:** their analytic kernel on the data cloud did not reproduce Monte-Carlo in our
dimensions. v2 uses a Markov state model on the model's own simulated states, which matches Monte-Carlo
(`results/PILOT_LOCAL_CPU.md` §3).

## 7. CC-Tempo v2 design (one paragraph)
State x = 50 PCs of scaled log-expression (HVGs + every LR subunit + lineage TFs). Ligand l_k(x) and receptor
r_k(x) are decoded from x for 64 LIANA-selected LR pairs (+ forced cytokine pairs). The population ligand
field is L_k = growth-weighted mean of l_k over the simulated population; the received signal is
s_k = gate_k·(αL_k + exogenous_k)·r_k(x). The landscape is U(x;s) = Φ(x) + U_c(x;s) (FiLM + linear tilt);
dynamics are x ← x − ∇ₓU dt + σ√dt z. Training uses a growth-weighted Sinkhorn loss to Day 4/6 snapshots,
with validation-based checkpointing. Ablation ladder: prescient → v1 → v2_local → v2_meanfield; controls:
ctrl_off, ctrl_shuffled, ctrl_random.

## 8. Datasets
| dataset | role | source | status |
|---|---|---|---|
| Weinreb in vitro (130k cells, 3 times, clones) | main fate benchmark (1,461 progenitors) | kleintools (AllonKleinLab/paper-data) | download code tested on metadata/clones |
| Weinreb cytokine perturbation (65k cells, 5 media) | causal evidence + held-out-medium prediction | same | **evidence analysis DONE locally** |
| Neural tube sci-FATE2 (≈45k cells, D3–D8, SAG 500 nM) | PNAS system, endogenous Shh hypothesis | GEO GSE236512 (splicing h5ad, 1.3 GB) | loader written; columns to confirm (Phase 5.1) |
| Synthetic heteroclinic flip + LR feedback | ground truth, false-positive control | generated | pilot DONE (1 seed) |

## 9. Where everything is (code map)
| path | content |
|---|---|
| `cctempo/models/cctempo.py` | all model modes, LR gates, exogenous doses, knockout masks |
| `cctempo/models/potentials.py` | PRESCIENT MLP, FiLM landscape |
| `cctempo/ccc/lr_encoder.py` | differentiable ligand/receptor read-out |
| `cctempo/ccc/liana_prior.py` | LIANA runs, LR selection, method overlap, v1 features |
| `cctempo/train.py` | trainer (resumable, time-budgeted, val checkpointing) |
| `cctempo/simulate.py` | Euler–Maruyama with mean field, exogenous, α(t), sender weights |
| `cctempo/evaluate/` | MC fate, metrics, statistics, path-integral / MSM operator |
| `cctempo/interpret/` | knockouts, sender silencing, windows, dose-response, attributions, TF activity, bifurcation |
| `cctempo/baselines/` | WOT, OT-CFM, official PRESCIENT adapter |
| `cctempo/data/` | Weinreb, neural tube, synthetic, gene sets, preprocessing |
| `cctempo/runner.py`, `api.py`, `cli.py` | glue, Python API, CLI |
| `scripts/00–12` | the pipeline (see README table) |
| `configs/` | datasets, model presets, **experiments.yaml** (run plan) |
| `kaggle/` | runner notebook (one STAGE per Save & Run All) + guide |
| `app/` | Gradio web tool |
| `docs/` | derivations, code audit, novelty list, user guide |
| `paper/` | reviewer response draft, manuscript outline |
| `results/PILOT_LOCAL_CPU.md` | everything measured so far |
| `work/cytokine/results/` | cytokine clone evidence CSVs (final) |
| `tests/test_smoke.py` | 8 CPU tests (all passing) |

## 10. Done vs left
**Done**
- [x] read the paper, reviews, PNAS paper, v1 code; audit (§4)
- [x] v2 package: models, training, simulation, evaluation, statistics, baselines, interpretation, API, CLI, app
- [x] Scanpy + LIANA pipeline (no Seurat/CellChat/R)
- [x] experiment plan sized for ~20–25 runs per tier (instead of 100 repeats)
- [x] CPU validation: smoke tests, fake-data integration of every script, synthetic pilot, MSM-operator fix
- [x] cytokine clone-matched evidence (final numbers)
- [x] docs: derivations, audit, novelty, user/Kaggle guides, response draft, manuscript outline, TODO plan

- [x] **Stage A on Kaggle (2026-10-08):** Weinreb downloaded and normalised (130,887 cells × 25,289 genes;
      1,461 benchmark progenitors, matching the local check); LIANA ran 7 methods × 3 time points
      (156,702 rows). Findings:
      - with 64 pairs the selection missed known axes (Cxcl5→Cxcr2, Il6, Ccl4→Ccr1, Tnf→Tnfrsf1a), so
        top-K was raised to 128 (68 ligand families); Cd34→Sell (v1's top pathway) ranks 138th;
      - NATMI, Connectome, geometric mean and SingleCellSignalR give identical top-K sets (Jaccard 1.0,
        all use the ligand×receptor expression product). CellChat vs those 0.87; CellPhoneDB 0.62–0.70;
        consensus vs single methods 0.33–0.35. Use this in the Reviewer 1(c) reply: methods differ by
        scoring family, not by name.
      Outputs: `kaggle/cct_outputs_stage A/`.

- [x] **Stages A+B+C on Kaggle (2026-10-08):** data rebuilt with 128 LR pairs (68 families).
      *Timing (T4):* 50 epochs = 0.30 min PRESCIENT / 0.47 min v2 → 2,500 epochs per run is affordable.
      *Synthetic benchmark (3 seeds; P(fate A) at t = 3; truth e0/e0.5/e1 = 0.455/0.127/0.032 at β = 1.5):*
      | model | e0 | e0.5 (held out) | e1 |
      |---|---|---|---|
      | PRESCIENT | 0.137 | 0.137 | 0.137 |
      | CC-off | 0.134 | 0.134 | 0.134 |
      | v2-local | 0.153 | 0.153 | 0.153 |
      | **v2-meanfield** | **0.445** | 0.048 | **0.015** |
      | shuffled-LR control | 0.467 | 0.128 | 0.035 |
      At β = 0 (no true communication) v2-meanfield predicts no dose effect (0.53 at all doses; truth 0.51).
      Problems found: (a) exogenous dose parameters never trained (gradient cut in the simulator: fixed);
      (b) decoy LR channels had larger knockout effects than the true channel, even at β = 0, i.e.
      channels were used as hidden state features → fixed by centring signals on a reference field
      (knockout = clamp at reference); (c) the shuffled control matches v2 here because, in this toy,
      the exogenous input alone identifies the condition; receptor identity is tested on real data.
      Outputs: `kaggle/cct_outputs_stage a+b+c/`.
- [x] **Fix verified on CPU (1 seed, β = 1.5):** dose now trains (1.0 → 2.15); held-out dose predicted
      correctly (P(A) 0.122 vs truth 0.127; before the fix 0.048). Knockouts: with the dose applied the TRUE
      channel dominates (1.56 vs decoys ≤ 0.31); without a dose decoys still outrank it (0.77 vs 0.08).
      Interpretation for the paper: in one unperturbed time course, every ligand whose population level
      changes over time is confounded with the true one; perturbation (exogenous dose) identifies the
      channel. Knockout rankings from Weinreb alone are hypotheses; the cytokine data provide the causal test.
      Added `v2_mf_sparse` (stronger gate L1) and `v2_mf_uncentered` (ablation) to the synthetic benchmark.

**Left (in order; details in TODO_PLAN.md)**
- [x] GitHub repo created and code pushed
- [x] Phase 0–1: Kaggle setup; Weinreb data prep (STAGE `A`)
- [x] STAGE `A+B+C`
- [ ] Next: `STAGE = 'B+D+E'` (synthetic re-run with fixed model, Weinreb tier 1, tables/baselines/interpretation)
- [ ] Phase 2: full synthetic benchmark (β = 1.5 and 0, 3 seeds)
- [ ] Phase 3: Weinreb tier 1 (20 runs) + evaluation + baselines + interpretation
- [ ] Phase 4: cytokine models (joint + leave-one-medium-out) + dose-response; **verify cocktail composition**
- [ ] Phase 5: neural tube (confirm columns first)
- [ ] Phase 6: robustness tier 2 (LIANA methods, random-gene control, held-out Day 4)
- [ ] Phase 7: HF Space, report, rewrite manuscript, fill response letter, proofreading, public repo + DOI

## 11. Risks and how we handle them
| risk | plan |
|---|---|
| v2 does not beat PRESCIENT on the in-vitro benchmark (constant medium) | The paper still stands on cytokine evidence + held-out-medium prediction + synthetic + interpretation. Report honestly: communication matters most when signals vary. |
| Kaggle time | tiers, resumable runs, 2-GPU sharding, epochs 1,500 with best-val selection |
| full-cocktail cytokine list unknown | must verify in Weinreb 2020 Methods before Phase 4 |
| neural-tube obs columns unknown | `--prepare` prints them; fall back to marker annotation |
| MSM operator vs MC disagree on real data | MC is the primary fate metric; the operator is used for landscape geometry, and agreement is reported |
| authorship/ethics of revealing the v1 issue | disclose transparently in the response (drafted); it strengthens the revision |

## 12. Run naming
`work/<dataset>/runs/<preset>[_<tag>]_s<seed>`, e.g. `v2_meanfield_s0`, `prescient_hold4_s1`,
`v2_meanfield_cellchat_s0`, `v2_meanfield_loco_Gcsf_s0`. Each run folder holds `config_used.yaml`,
`train_log.csv`, `model_best.pt`, `metrics_best.json`, `preds_best.npz`, `run_summary.json`.
