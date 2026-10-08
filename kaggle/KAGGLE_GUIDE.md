# Running CC-Tempo v2 on Kaggle

Code: https://github.com/Ahnaf-Akib-Tanim/cc-tempo_v2 (public; the notebook clones it every run).

## One-time setup
1. Kaggle → Settings → verify your phone number (needed for Internet and GPU).
2. Kaggle → Code → **New Notebook** → File → **Import Notebook** → upload `kaggle/cctempo_kaggle_runner.ipynb`.
   Name it e.g. `cctempo-runner`.
3. Right panel → Session options: Accelerator **GPU T4 x2**, Internet **On**.

## Do I run the whole notebook at once? Yes, with Save & Run All, one STAGE at a time
The notebook runs **one stage per version**. You never run the cells by hand.

1. Edit cell 2: `STAGE = 'A'` (then `B`, `C`, `D`, … following TODO_PLAN.md).
2. Top right: **Save Version** → **Save & Run All (Commit)** → Save.
   Kaggle runs everything in the background for up to 12 h. You can close the browser or turn off your laptop.
3. When it finishes (Kaggle emails you, or check *View Active Events*), open the version → **Output** tab →
   download **`cct_outputs.zip`** and send it to Claude. Also look at the log (the *Logs* tab) if anything says `FAILED`.
4. **Next stage:** in the editor, **+ Add Input → Your Work → `cctempo-runner`** (pick the latest version).
   Its `work/` folder (processed data, finished runs, half-finished runs) is copied back automatically by
   the setup cell. Change `STAGE`, then Save & Run All again.
   Each new version's output contains the full `work/`, so always attach the **latest** version.

Why not run cells interactively? Interactive sessions stop when idle or when you close the tab, and you
would have to watch them. Commit runs are unattended and keep their output.

## Long stages (D, W2, F2, G2)
A stage can need more than 12 h. The notebook sets a deadline of 10.75 h. Training stops cleanly before it,
and no new runs are started. The version then finishes normally and saves everything. **Run the same STAGE
again** (with the latest version attached as input): finished runs are skipped and the half-finished one
resumes from its last checkpoint. Repeat until the log prints no `[deadline]` lines.

## Quota
- 30 GPU-hours per week. Stages A, F1 and G1 need no GPU, so set Accelerator *None* for them.
- Stage C (10–20 min) tells us how long one training run takes. Send the zip before starting D, and I will
  adjust epochs/seeds so tier 1 fits your budget.
- Both T4 GPUs are used in parallel by the grid stages (two runs at a time).

## Output size
`/kaggle/working` (max 20 GB) holds `work/` (processed data ≈ 1–3 GB + runs). Raw downloads are deleted
after each data stage. `cct_outputs.zip` holds only results, metrics, logs and `model_best.pt` files (small).

## Troubleshooting
| symbol | fix |
|---|---|
| `FAILED` in the log | send Claude the log text around it |
| `git clone` fails | Internet is off, or the repo became private |
| CUDA out of memory in evaluation | `EXTRA_SET = ['eval.max_particles=100000']` |
| restored nothing ("restoring" not printed) | you forgot + Add Input → Your Work → latest version |
| version still hit 12 h | lower `SESSION_HOURS` in the setup cell to 10 |
