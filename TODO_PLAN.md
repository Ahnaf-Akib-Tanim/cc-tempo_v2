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
- [x] 1.1 STAGE `A`: download Weinreb, LIANA (7 methods), build `data.pt`. 1461 progenitors confirmed.
- [x] 1.2 LR selection reviewed: top-K raised from 64 to 128 (see PROJECT_STATUS.md).
- [x] 1.3 `STAGE = 'A+B+C'` done (128 pairs, data.pt rebuilt, synthetic benchmark, timing).

## Phase 2: synthetic ground truth: done once, re-run after the model fix
- [x] 2.1 First run (STAGE `B`). Communication-coupled model recovers the dose effect; autonomous models
      cannot; no false effect at β = 0. Two problems found and fixed in code: (a) exogenous doses did not
      train (gradient was cut), (b) decoy channels had large knockout effects (channels used as hidden
      state features) → signals are now centred on a reference field (docs/DERIVATIONS.md §1).
- [ ] 2.2 Re-run STAGE `B` with the fixed model (part of the next version, see 3.2).

## Phase 3: Weinreb main experiments (the core of the paper)
- [x] 3.1 Timing (STAGE `C`): 50 epochs = 0.30 min (PRESCIENT), 0.47 min (v2). So runs use 2,500 epochs
      (as in v1): ~15–24 min training + ~10 min evaluation per run.
- [x] 3.2 `B+D+E` (no Weinreb runs: input not restored), then `A+D+E`: 15/20 tier-1 runs finished.
      All 5 v2_meanfield seeds diverged (centred signal scales down to 1e-3). Fixed: scale floor +
      bounded signal + skip of diverged steps (docs/DERIVATIONS.md §1). PRESCIENT 0.472, ctrl_off 0.469,
      v1 0.456, ctrl_shuffled 0.496 (old signal, now marked stale), v2_local 0.431, WOT 0.491 (MC PCC).
- [x] 3.2b `D+E` failed: `work/` was never restored from the attached input (in no version so far).
      Restore now searches every depth of /kaggle/input and lives in scripts/kaggle_run.py.
- [x] 3.2c `A+D+E` again: all 20 tier-1 runs finished (1 diverged step skipped). Monte-Carlo fate-bias PCC
      (mean ± sd over seeds): **v2_meanfield 0.500 ± 0.004** (5/5 seeds above every PRESCIENT seed), WOT 0.491,
      ctrl_shuffled 0.475 (2 seeds), PRESCIENT 0.472 ± 0.014, ctrl_off 0.469, v2_local 0.466, v1 0.456, OT-CFM 0.27.
      Paired bootstrap v2_meanfield − PRESCIENT: +0.034 [0.018, 0.051], Holm p = 0.003. KL bits at Day 4 are
      worse than PRESCIENT (5.6 vs 3.8); W2 and L1 composition are better.
- [x] 3.3 Decision point: v2_meanfield beats PRESCIENT, ctrl_off and v1 on Weinreb. The shuffled control
      has only 2 seeds, so 3 more are added to tier 2 and 07_evaluate now also tests against ctrl_shuffled.
- [x] 3.4 `W2+B` (restore from the attached version worked). Weinreb MC PCC: **ctrl_random 0.505 (2 seeds)**,
      v2_meanfield 0.500, v2 with CellPhoneDB/CellChat/NATMI pairs 0.486–0.489, ctrl_shuffled 0.482 (5 seeds),
      PRESCIENT 0.472. v2 − shuffled: +0.016 [0.005, 0.028], Holm p = 0.08. Held-out Day 4: v2 worse than
      PRESCIENT (KL 16.8 vs 7.1 bits, W2 11.0 vs 9.3).
      Synthetic (bounded signal): mean-field models recover the communication-driven fate (err 0.02–0.03 vs
      PRESCIENT 0.32) but so does ctrl_shuffled (0.05); without a dose, decoy knockouts (~1.0) dwarf the true
      channel (~0.1), also at β = 0; with a dose the true channel wins 3/3 (also for ctrl_shuffled, because the
      dose enters that channel by construction).
      **Reading:** population (mean-field) coupling improves fate prediction, but *which* LR channel carries it
      is not identifiable from unperturbed time courses: channel fields are collinear summaries of population
      composition. Channel identity needs perturbations → the cytokine experiment (Phase 4) is the key evidence.
      To discuss with the co-authors before the rewrite.

## Phase 4: cytokine perturbation (Kaggle; strongest biological evidence)
- [ ] 4.1 `10_cytokine_experiments.py --evidence` (already run locally: results in
      `work/cytokine/results/evidence_*.csv`, M-CSF vs G-CSF on the same 187 clones, p < 1e-4).
- [x] 4.2 Cocktail verified (Supp Methods 2.5, 8.1): IL-3, FLT3-L, IL-11, IL-5, TPO, SCF, IL-6, EPO; arms on day 2:
      cocktail / M-CSF / G-CSF / EPO (SCF arm only in GEO metadata, Day 4 only). Config fixed; IL-11 and IL-5
      channels added, so F1's LIANA/build steps must run again (STAGE `F1+F2`).
- [ ] 4.3 **Next version: `STAGE = 'F1+G1'` (Accelerator None)**. STAGE `F1` (prepare → LIANA → build), then STAGE `F2` (tier1 5 runs → tier2 leave-one-medium-out
      6 runs → evaluate → dose-response).
- [ ] 4.4 Expected: v2 predicts the held-out medium's composition shift (M-CSF → more monocytes,
      G-CSF → more neutrophils). PRESCIENT cannot, because it has no input for the medium.

## Phase 5: neural tube, the PNAS system (Kaggle)
- [x] 5.0 `F1+G1`: cytokine data built (65,075 cells; progenitors: cocktail 662, M-CSF 713, G-CSF 495,
      Epo 106, SCF 145; clone-matched evidence holds). Neural tube: obs has only timepoint/rep/sample (no
      cell types, no SAG-dose column); timepoints D3–D8 plus 05h–20h (= Day 3 + hours, replicate r4 only).
      The prepare step stopped after inspect (no prepare_summary.json; cause not logged). Fixed: hour
      parsing, layer handling, and the runner now saves its full console log into the zip.
      Added scripts/13_early_progenitors.py (stage `EP`): PCC on CD34+ early progenitors (Weinreb Fig 5c,d)
      and the held-out-clone upper bound (Supp Methods 13.5), from saved predictions.
      **Next version: `STAGE = 'G1+F1+EP+F2'`.**
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
