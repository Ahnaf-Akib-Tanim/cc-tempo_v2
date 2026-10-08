# TODO plan: what to do, in what order

This is the working checklist for the CC-Tempo revision. Tick boxes as you go.
Training runs on **Kaggle** (GPU T4 ×2). Your laptop is only for reading results and writing.
After each Kaggle stage, download `cct_outputs.zip`, unzip it into this folder (it creates/updates
`work/…/results`), and send me the zip or tell me what ran. I will read the numbers and update the
paper text and the next steps.

Budget assumptions: about 20–25 training runs per week (Kaggle: 30 GPU-h/week, 12 h/session).
"Seeds" below replace the paper's 100 repeated trainings. With 3–5 seeds we use paired bootstrap
over the ~1,461 progenitors plus seed-level tests (see `cctempo/evaluate/stats.py`). That is
statistically sound and is what the reviewers asked for ("more evaluation indexes").

---

## Phase 0: setup (≈30 min, once)
- [x] 0.1 GitHub repo: https://github.com/Ahnaf-Akib-Tanim/cc-tempo_v2 (code pushed; `raw/` and `work/` stay local via `.gitignore`).
- [ ] 0.2 Kaggle: verify your phone. New Notebook → Import `kaggle/cctempo_kaggle_runner.ipynb` →
      Accelerator **GPU T4 x2**, Internet **ON**. Read `kaggle/KAGGLE_GUIDE.md` once.
- [ ] 0.3 How every Kaggle step works: set `STAGE` in cell 2 → **Save Version → Save & Run All (Commit)** →
      wait (browser can be closed) → download `cct_outputs.zip` from the version's Output → send it to me.
      For the next stage, **+ Add Input → Your Work → latest version of this notebook**, change `STAGE`, and run again.
- [ ] 0.4 Locally (optional): `pip install -e .[dev]` then `pytest -q tests` (CPU, about 30 s).
- [ ] 0.5 After you change code locally (or I do): `git add -A && git commit -m "..." && git push`.
      The notebook always pulls the latest code.

## Phase 1: data (STAGE `A`, Accelerator None; ≈1–1.5 h)
- [ ] 1.1 STAGE `A`: download Weinreb, LIANA (7 methods), build `data.pt`.
      Check in the log: **1461** progenitors (I verified this number locally).
- [ ] 1.2 Send me the zip. I check `liana_method_overlap.csv` and the top LR pairs (feeds the Reviewer 1(c) answer).

## Phase 2: synthetic ground truth (STAGE `B`; ≈1–2 h; cheap)
- [ ] 2.1 STAGE `B`: `04_synthetic_benchmark.py` (β = 1.5 and β = 0, 5 models × 3 seeds).
- [ ] 2.2 Expected:
      - for β = 1.5, v2_meanfield has a clearly lower held-out-dose (e0.5) fate error and composition L1 than prescient/ctrl_off;
      - the TRUE pair has the largest knockout effect (synthetic_knockouts.csv);
      - for β = 0, there is no gain and knockout effects are ≈ 0.
      If not, send me the CSVs before Phase 3.

## Phase 3: Weinreb main experiments (Kaggle GPU; the core of the paper)
- [ ] 3.1 STAGE `C` timing run (50 epochs). Minutes/run ≈ (last column) × epochs / 50. If a run would take
      > 60 min, set `train.epochs=1000` in configs/base.yaml (best-val checkpointing protects you).
- [ ] 3.2 STAGE `D`: tier 1 (20 runs: prescient×5, v2_meanfield×5, v1×3, ctrl_off×3, ctrl_shuffled×2,
      v2_local×2), two runs in parallel on the two T4s. If the log shows `[deadline]`, run STAGE `D` again
      (latest version attached): finished runs are skipped and the stopped run resumes.
- [ ] 3.3 STAGE `E`: `07_evaluate.py` → `summary.csv`, `stats_seed.csv`, `stats_bootstrap.csv`.
- [ ] 3.4 (also STAGE `E`) `08_baselines.py` (WOT, OT-CFM, OT-CFM-SDE) → `baselines_long.csv`.
- [ ] 3.5 (also STAGE `E`) `09_interpret.py --run v2_meanfield_s0` (≈1 h): knockouts, sender ablation, temporal windows,
      TF perturbation, gene attribution with lineage-traced validation, signal→TF, bifurcation.
- [ ] 3.6 Pack, download, send me. **Decision point:** if v2_meanfield does not beat prescient and the controls,
      we write the honest version (communication helps on cytokine and synthetic data, and is neutral in
      constant medium). The cytokine evidence carries the paper either way.

## Phase 4: cytokine perturbation (Kaggle; strongest biological evidence)
- [ ] 4.1 `10_cytokine_experiments.py --evidence` (already run locally: results in
      `work/cytokine/results/evidence_*.csv`, M-CSF vs G-CSF on the same 187 clones, p < 1e-4).
- [ ] 4.2 **VERIFY** the full-cocktail cytokine list in Weinreb et al. 2020 Methods and fix
      `configs/cytokine.yaml → train.condition_cytokines["full cocktail"]`.
- [ ] 4.3 STAGE `F1` (prepare → LIANA → build), then STAGE `F2` (tier1 5 runs → tier2 leave-one-medium-out
      6 runs → evaluate → dose-response).
- [ ] 4.4 Expected: v2 predicts the held-out medium's composition shift (M-CSF → more monocytes,
      G-CSF → more neutrophils). PRESCIENT cannot, because it has no input for the medium.

## Phase 5: neural tube, the PNAS system (Kaggle)
- [ ] 5.1 STAGE `G1` first, then send me `work/neural_tube/inspect.txt`. I need to confirm the time and
      cell-type columns (or the marker annotation) before training.
- [ ] 5.2 STAGE `G2`: LIANA → build → tier1 (5 runs) → evaluate → `--analyze`.
- [ ] 5.3 Key outputs:
      - `endo_hh_ko.csv`: does knocking out endogenous Shh signalling remove the late p3→FP drift?
      - `landscape_by_day.csv`: does the FP basin deepen over time?
      - `kl_bits.csv`: fit quality in bits, comparable in spirit to the PNAS paper's 0.3 bits.

## Phase 6: robustness + extras (tier 2/3, as budget allows)
- [ ] 6.1 STAGE `W2`, Weinreb tier2: CCC-method robustness (CellPhoneDB/CellChat/NATMI/Connectome via LIANA), random-gene
      control, held-out Day-4 interpolation.
- [ ] 6.2 Optional: official PRESCIENT package run (`cctempo/baselines/prescient_official.py`).
- [ ] 6.3 Optional: more seeds for the two headline models (tier3).

## Phase 7: tool + paper (laptop)
- [ ] 7.1 Web tool: create a Hugging Face Space (Gradio, CPU). Upload `app/`, `cctempo/`, `configs/`,
      `app/requirements.txt`, one trained run folder plus its `data.pt` and classifier. Steps are in docs/USER_GUIDE.md.
- [ ] 7.2 `python scripts/12_make_report.py --root .` → `results/RESULTS_REPORT.md` (all tables).
- [ ] 7.3 Rewrite the manuscript following `paper/revised_manuscript_outline.md`. Methods go to Methods,
      R/CLI commands go to Supplementary, and the derivations come from docs/DERIVATIONS.md.
- [ ] 7.4 Fill the numbers into `paper/response_to_reviewers_draft.md`.
- [ ] 7.5 Language editing (Reviewer 3): get a native-English proofread or use a professional service before submission.
- [ ] 7.6 Make a GitHub release + Zenodo DOI and cite it in Code Availability.

## What to send me after each phase
The `cct_outputs.zip` (or just `work/*/results/` and any `*.log` with errors). Tell me which cells ran
and whether anything failed. I will update PROJECT_STATUS.md, the tables and the paper text.
