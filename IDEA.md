# KaggleResearch — Build Specification for Claude Code

> A Colab-native notebook that performs literature-seeded, self-directed autoresearch on arbitrary Kaggle competitions. The agent selects a strategy, exploits it via a ratchet loop, detects when it's stuck, and re-researches to either refine or pivot — branching the codebase rather than discarding prior work.

| Field | Value |
|---|---|
| Version | 2.0 |
| Target Runtime | Google Colab (free tier + A100 session) |
| Primary Inspiration | karpathy/autoresearch (MIT) |
| Core Pattern | Research → Strategise → Exploit → Plateau Detection → Re-research → (Refine or Branch) |
| Novel Contribution | Strategy-aware autoresearch loop with literature-triggered pivots and git branching |

---

## 1. Overview & Goals

KaggleResearch is a Jupyter notebook that runs entirely on Google Colab. It takes a Kaggle competition URL, reads academic literature to select an explicit strategy, then runs an autoresearch-style ratchet loop. When experiment gains plateau below a user-configured threshold, it re-searches for a new angle and either layers new ideas onto the current strategy or branches into a parallel approach — keeping the best prior work safe in a separate git branch.

### 1.1 Core Principles

- **Strategy-first:** Before any experiments run, the agent produces a written `STRATEGY.md` — a short, human-readable document stating what approach it has chosen, why, and what it expects to gain. This is the agent's north star.
- **Exploit before exploring:** The agent stays in the ratchet loop as long as experiments are making progress above the plateau threshold. Re-research is a last resort, not a scheduled step.
- **Branching over discarding:** When a pivot is warranted, the current best state is preserved in a named git branch. The new branch is explored for `BRANCH_COMPARE_N` experiments before the better branch is declared the main line.
- **Colab-resilient:** All state lives in Google Drive. Session disconnects are non-fatal at any phase.
- **Human checkpoints at strategy time, not experiment time:** The user reviews and can edit `STRATEGY.md` and the seeded `IDEAS.md` before the loop begins. After that, the loop runs unattended.

### 1.2 What This Is NOT

- Not a hyperparameter grid search — the agent proposes algorithmic and architectural changes
- Not a distributed system — single Colab GPU, one experiment at a time
- Not AutoKaggle or AIDE — those are one-shot pipelines; this is an iterative self-directed research loop

---

## 2. High-Level Flow

```
[Config Cell]
     │
     ▼
[Bootstrap]
  - Parse competition (problem type, metric, data)
  - Generate baseline train.py from template library
  - Validate baseline runs and produces a real score
     │
     ▼
[Initial Literature Review]
  - Search arXiv + SemanticScholar for SOTA on this problem type
  - LLM synthesises: select a strategy, write STRATEGY.md
  - LLM populates IDEAS.md with ordered experiments for that strategy
     │
     ▼
[Human Checkpoint]  ← user reviews/edits STRATEGY.md and IDEAS.md
     │
     ▼
┌────────────────────────────────────────────────────────────────────────┐
│  EXPLOIT PHASE (ratchet loop)                                          │
│                                                                        │
│  while ideas_remaining and not plateau_triggered and not timeout:      │
│      implement idea → train → eval → keep or revert → log             │
│      update STRATEGY.md learning log with what was learned             │
│      check plateau condition                                           │
│                                                                        │
└──────────────┬─────────────────────────────────────────────────────────┘
               │
        plateau triggered?
               │
               ▼
[Re-research Phase]
  1. Try new angle — search with context of what failed
  2. If new angle yields fresh IDEAS.md entries:
       - Do new ideas fit current strategy?
           YES → append to IDEAS.md, continue exploit phase
           NO  → PIVOT: branch current best, start new strategy branch
  3. If no new angle found:
       - Re-read existing papers with failure context → refine IDEAS.md
       - If still no new ideas → HALT with summary

[Branch Comparison] (only on pivot)
  - Run BRANCH_COMPARE_N experiments on new branch
  - Compare best score of each branch
  - Promote winning branch to main, archive the other
  - Continue exploit phase on winning branch
```

---

## 3. File Layout

```
gdrive/MyDrive/kaggleresearch/<competition_slug>/
  kaggleresearch.ipynb         # The notebook
  config.json                  # All user settings
  STRATEGY.md                  # Agent's current written strategy (human-editable)
  IDEAS.md                     # Ordered experiment agenda (human-editable)
  experiment_log.db            # SQLite: all experiments across all branches
  checkpoint.json              # Current phase, best score, branch, IDEAS pointer
  repo/
    train.py                   # EDITABLE — model/training logic
    metric.py                  # FIXED — competition metric scorer
    data/                      # FIXED — raw competition data (read-only)
    submissions/               # Agent writes submission.csv here
    baseline_score.txt         # Locked reference score
  literature_cache/
    papers.json                # All retrieved papers (reused on re-research)
    search_history.json        # Queries already run (avoid duplicate searches)
    archived_strategies.md     # Old STRATEGY.md files from before pivots
```

Git branches in `repo/`:
```
main                           # Current active line of development
branch/strategy-v1             # Archived branch from before a pivot
branch/strategy-v2             # Current, or also archived if v3 started
```

---

## 4. Configuration

The first notebook cell. User fills this in before running anything else.

```python
# ─── KaggleResearch Config ───────────────────────────────────────────────────
COMPETITION_URL      = 'https://www.kaggle.com/competitions/titanic'
DRIVE_PATH           = '/content/drive/MyDrive/kaggleresearch'

# Experiment timing
TIME_BUDGET_MIN      = 4        # Minutes per experiment (3–5 for Colab)

# Plateau detection — re-research triggers when:
# fewer than PLATEAU_MIN_GAIN_PCT % improvement over last PLATEAU_WINDOW experiments
PLATEAU_WINDOW       = 5        # Number of recent experiments to evaluate
PLATEAU_MIN_GAIN_PCT = 1.0      # Minimum cumulative % gain over that window

# Branch comparison — when a pivot happens, how many experiments to run
# on the new branch before comparing against the old best
BRANCH_COMPARE_N     = 5

# Literature
LITERATURE_DEPTH     = 10       # Papers retrieved per search (5–20)
RUN_LIT_REVIEW       = True     # Set False to skip if STRATEGY.md already exists
# ─────────────────────────────────────────────────────────────────────────────
```

---

## 5. Stage Specifications

### Stage 1 — Bootstrap

Responsibilities:

1. Parse competition metadata via Kaggle API (name, problem type, metric, data files)
2. Classify problem type: `tabular-classification`, `tabular-regression`, `image-classification`, `image-segmentation`, `nlp-classification`, `nlp-regression`, `time-series`, `other`
3. Select and render a baseline `train.py` from the template library (Section 7)
4. Generate `metric.py` — a standalone scorer taking `(y_true_path, y_pred_path) -> float`
5. Download data via Kaggle CLI into `repo/data/`
6. Run baseline once, validate it completes within `TIME_BUDGET_MIN` and produces a real score
7. Write `baseline_score.txt`, initialise `checkpoint.json`

Failure handling:
- Missing Kaggle credentials → print setup instructions, halt
- Unrecognised problem type → default to `tabular-classification`, warn in `STRATEGY.md`
- Baseline crashes → surface error, ask user to fix before proceeding
- Baseline score is NaN/inf → `metric.py` is wrong, halt

---

### Stage 2 — Initial Literature Review

#### Goal

Produce `STRATEGY.md` and a seeded `IDEAS.md`. The literature review has an explicit **strategy selection step** — the agent must commit to one approach before any experiments run.

#### Search

1. Query SemanticScholar with: `[problem_type] + [metric_name] + "SOTA" + current year`
2. Query arXiv with same keywords, filtered to `cs.LG`, `stat.ML`, `cs.CV` as appropriate
3. Retrieve top `LITERATURE_DEPTH` papers by citation count × recency score
4. Cache all results to `literature_cache/papers.json` and queries to `search_history.json`

#### Strategy Selection (LLM step)

After retrieval, ask Claude to produce `STRATEGY.md` using this prompt:

```
You are an ML research strategist. You have:
  - Competition: [name, problem type, metric, dataset description]
  - Baseline score: [score] using [model]
  - The following papers and their key methods: [paper summaries]

Your task: Select ONE primary strategy for this competition and write STRATEGY.md.

STRATEGY.md must contain:
  1. Strategy name (e.g. "Gradient Boosted Trees with Advanced Feature Engineering")
  2. Rationale — why this approach suits this competition (2–3 sentences)
  3. Expected ceiling — what score do you expect this strategy can reach?
  4. Key risks — what could cause this strategy to plateau early?
  5. Pivot signal — what result pattern would tell you this strategy is wrong?
  6. Literature basis — which papers informed this choice (arxiv IDs)

Do not hedge. Pick one strategy and justify it.
```

#### IDEAS.md Population

After `STRATEGY.md` is written, Claude generates `IDEAS.md` containing only experiments consistent with the chosen strategy. See Section 6 for the full schema.

Ordering rules:
- Risk ASC (low risk first)
- Within same risk tier: Estimated Gain DESC
- At least 3 low-risk entries at the top
- Ideas requiring new `pip install` marked `Risk: high`

#### Human Checkpoint

After this stage, the notebook displays `STRATEGY.md` and `IDEAS.md` side-by-side in scrollable widgets and pauses on an **"Approve Strategy & Start Loop"** button. The user may edit either file in Drive before clicking. This is the last human intervention until the summary.

---

### Stage 3 — Exploit Phase (Ratchet Loop)

#### Core Loop

```python
while ideas_remaining() and not session_timeout_imminent():

    # Plateau check before each experiment
    if plateau_triggered(experiment_log, PLATEAU_WINDOW, PLATEAU_MIN_GAIN_PCT):
        trigger_reresearch()
        break  # re-research will restart or branch this loop

    idea = next_untried_idea(IDEAS.md, experiment_log)
    patch = implement_idea(idea, train.py, STRATEGY.md)  # Claude API call
    write_file('repo/train.py', patch)

    score = run_timed_experiment(TIME_BUDGET_MIN)  # float or None

    if score is None:                       # crashed
        git_reset_hard()
        log_experiment(idea, 'crashed')

    elif score > best_score:
        git_commit(f'IMPROVE [{current_branch}]: {idea.title} | '
                   f'{best_score:.4f} -> {score:.4f}')
        best_score = score
        append_learning_log(STRATEGY.md, idea, score)
        update_checkpoint(best_score)
        log_experiment(idea, 'improved', score)

    else:
        git_reset_hard()
        log_experiment(idea, 'no_improvement', score)

    display_live_table()
```

#### Plateau Detection

```python
def plateau_triggered(log, window, min_gain_pct):
    recent = log.last_n_completed(window)  # excludes crashes
    if len(recent) < window:
        return False  # not enough data yet
    best_in_window  = max(e.score for e in recent)
    worst_in_window = min(e.score for e in recent)
    gain_pct = (best_in_window - worst_in_window) / abs(worst_in_window) * 100
    return gain_pct < min_gain_pct
```

A window of all crashes does not trigger re-research — crashes are implementation errors, not strategic plateau.

#### STRATEGY.md Learning Log

After each successful improvement, the agent appends a note to `STRATEGY.md` under a `## Learning Log` section:

```markdown
## Learning Log

- Experiment 4 (IMPROVE): Adding sqrt(feature_x) improved score by 0.8%.
  Suggests feature distributions are skewed — explore more transformations.
- Experiment 7 (IMPROVE): Label smoothing 0.1 helped. Overfitting was a risk.
```

This log is passed as context on every subsequent `implement_idea` call and on re-research, so the agent always knows what it has already learned.

#### Code Agent Prompt

```
You are a precise ML engineer. You will receive:
  1. The current train.py (full file)
  2. The current STRATEGY.md (including the Learning Log)
  3. One IDEA entry from IDEAS.md

Your task: Return ONLY the modified train.py — complete file, no commentary.

Rules:
  - Implement the idea in a way consistent with the current strategy in STRATEGY.md
  - Make the minimum change required
  - Do not modify metric.py or data loading paths
  - Do not add imports requiring pip install (unless in the pre-installed list)
  - If the idea conflicts with the strategy, implement the lowest-risk interpretation
  - If the idea is impossible given the current code, return the file unchanged
```

---

### Stage 4 — Re-research Phase

Triggered when `plateau_triggered()` returns True. The agent has two attempts before halting.

#### Attempt 1: New Angle

Search for papers from a different direction, informed by what has failed:

```python
failure_ctx  = summarise_failures(experiment_log)  # what didn't work
new_query    = build_query(problem_type, metric, exclude=search_history,
                           context=failure_ctx)
new_papers   = search(new_query, LITERATURE_DEPTH)
cache_papers(new_papers)
```

Pass new papers + `STRATEGY.md` learning log to Claude:

```
You are an ML research strategist reviewing a stuck experiment loop.

Current strategy: [STRATEGY.md contents]
What has been tried and failed: [failed IDEAS.md entries + scores]
New papers found: [paper summaries]

Answer:
  1. Do any new papers suggest improvements WITHIN the current strategy?
     If yes: produce new IDEAS.md entries (append, do not replace existing).
  2. Do the new papers strongly suggest the current strategy is wrong for this problem?
     If yes: propose a PIVOT — name the new strategy and justify it in one paragraph.
  3. If neither: respond NO_NEW_IDEAS.
```

#### Decision Tree After Attempt 1

```
new IDEAS within current strategy  →  append to IDEAS.md, resume exploit phase
PIVOT proposed                     →  trigger Branch Comparison (Stage 5)
NO_NEW_IDEAS                       →  Attempt 2: Re-read
```

#### Attempt 2: Re-read Existing Papers

```
You previously read these papers: [literature_cache/papers.json]
Since then, the following experiments have failed: [failure log]
The current STRATEGY.md learning log is: [log]

Re-read the papers with this new context. Are there methods you previously
overlooked that might explain why the current approach is stuck?
Produce new IDEAS.md entries if so, or respond NO_NEW_IDEAS.
```

If this also yields `NO_NEW_IDEAS` → halt loop, render summary, note the strategy is exhausted.

---

### Stage 5 — Branch Comparison (Pivot)

Only triggered when re-research proposes a strategy pivot.

#### Steps

1. **Archive current branch:**
   ```bash
   git checkout -b branch/strategy-v{N}
   # record branch name and best_score in checkpoint.json
   ```

2. **Write new STRATEGY.md** — Claude generates a fresh strategy document. The old one is appended to `literature_cache/archived_strategies.md`.

3. **Reset train.py to baseline** — the new branch starts from the original baseline `train.py`, not the current best. A pivot implies a different approach, not incremental refinement.

4. **Generate new IDEAS.md** — fresh entries for the new strategy. Old `IDEAS.md` is archived.

5. **Run `BRANCH_COMPARE_N` experiments** on the new branch using the normal ratchet loop.

6. **Compare and promote:**
   ```python
   old_best = checkpoint['branches']['strategy-v{N}']['best_score']
   new_best = current_best_score

   if new_best > old_best:
       # new branch wins — continue exploit phase on new branch
   else:
       # old branch wins — restore it
       git_checkout('branch/strategy-v{N}')
       append to STRATEGY.md: "Pivot to [new strategy] underperformed after
         {BRANCH_COMPARE_N} experiments. Returning to prior approach."
   ```

7. The losing branch is kept in git but marked `archived` in `checkpoint.json`. The summary cell shows both branches and their best scores.

> Only one pivot comparison can be active at a time. Nested pivots are not supported in v1.

---

### Stage 6 — Summary Cell

Renders after the loop completes, is interrupted, or halts on `NO_NEW_IDEAS`.

Output:

- **Strategy timeline** — which strategies were tried, how many experiments each ran, best score per strategy
- **Branch comparison table** — if a pivot occurred: old vs new branch best score, winner declared
- **Improvement waterfall** — all winning experiments ranked by score delta, with source paper citations from `IDEAS.md`
- **Learning log** — the full `## Learning Log` from `STRATEGY.md`
- **Git log** — winning commits only, formatted as a table with branch labels
- **Ideas never tried** — remaining `pending` entries in `IDEAS.md` for manual follow-up
- **Final score vs baseline** — absolute delta and estimated leaderboard percentile

---

## 6. IDEAS.md Schema

Each entry follows this exact format — parsed programmatically by the agent:

```
## IDEA: [Short descriptive title]
Source: [Paper title] (arxiv:[ID] | empirical | derived-from-strategy)
Risk: low | medium | high
Estimated gain: small | medium | large
Status: pending | running | improved | no_improvement | crashed | skipped
---
Hypothesis: [One sentence on why this should help, grounded in STRATEGY.md]
Implementation: [Precise description of what to change in train.py —
  function name, before/after pseudocode. Must be unambiguous.]
Validation: [How to know it worked — metric direction and magnitude]
===
```

`Status` is updated by the agent after each experiment. The agent always picks the next `pending` entry. Entries with `Status: improved` are locked and cannot be replaced.

---

## 7. Baseline Template Library

| Problem Type | Default Model | Key Libraries | Notes |
|---|---|---|---|
| `tabular-classification` | LightGBM + OrdinalEncoder | lightgbm, sklearn | Stratified k-fold CV |
| `tabular-regression` | LightGBM Regressor | lightgbm, sklearn | k-fold CV; log-transform target if skewed |
| `image-classification` | timm ResNet18 pretrained | timm, torchvision | Resize 224×224, basic augment |
| `image-segmentation` | SMP Unet (EfficientNet-b0) | segmentation_models_pytorch | Stub OK for v1 |
| `nlp-classification` | DistilBERT fine-tune | transformers, datasets | Max 512 tokens, AdamW |
| `nlp-regression` | DistilBERT + regression head | transformers | MSE loss, clip predictions |
| `time-series` | LightGBM + lag features | lightgbm, pandas | Manual lag/rolling features |
| `other` | LightGBM (generic fallback) | lightgbm, sklearn | Warn in STRATEGY.md |

Each template is a self-contained `train.py` (~100–200 lines) with clearly marked `# EDITABLE` and `# FIXED` sections. On pivot, `train.py` is reset to the template matching the new strategy's approach.

---

## 8. Colab-Specific Constraints

### Session Survival

All state persists to Drive. On reconnect: mount Drive → load `checkpoint.json` → detect current phase → resume from exact position.

`checkpoint.json` structure:
```json
{
  "phase": "exploit | reresearch | branch_compare",
  "current_branch": "main",
  "best_score": 0.8312,
  "ideas_pointer": 7,
  "plateau_window_scores": [0.831, 0.831, 0.830, 0.831, 0.832],
  "reresearch_attempts": 0,
  "branches": {
    "branch/strategy-v1": { "best_score": 0.812, "status": "archived" }
  }
}
```

Wrap the loop in `try/finally` — always flush `checkpoint.json` before exit.

### GPU Scaling

Detect at startup via `torch.cuda.get_device_name(0)`. Scale `TIME_BUDGET_MIN`:
- A100 → use configured value as-is
- T4 → configured value × 1.5
- CPU → warn user, skip GPU-dependent templates

### Pre-installed Libraries (No pip install needed)

`torch`, `torchvision`, `sklearn`, `pandas`, `numpy`, `lightgbm`, `xgboost`, `transformers`, `datasets`, `timm`

### Requires Install (Cell 0)

`kaggle`, `anthropic`, `arxiv`, `segmentation_models_pytorch`

---

## 9. Agent Constraints

### PERMITTED

- Modify any function, class, or constant in `train.py`
- Add new functions inside `train.py`
- Change model architecture, optimizer, scheduler, loss, augmentation
- Change batch size, learning rate, number of epochs (within time budget)
- Use any library already imported at the top of `train.py`
- Append to the `## Learning Log` section of `STRATEGY.md`

### FORBIDDEN

- Modifying `metric.py` — ever, for any reason
- Modifying or deleting anything in `data/`
- Adding `import` statements that require `pip install` (unless pre-installed)
- Changing the output path of `submission.csv`
- Running subprocesses calling external services
- Modifying git history (rebase, amend, force push)
- Overwriting the top sections of `STRATEGY.md` — only the Learning Log may be appended
- Replacing `IDEAS.md` entries with `Status: improved` — these are locked

Constraints enforced both in agent system prompts AND programmatically: a validator runs after every `implement_idea` call and before the experiment starts, checking that `metric.py` and `data/` are unchanged.

---

## 10. External APIs & Authentication

| API | Details |
|---|---|
| **Kaggle API** | Required. User uploads `kaggle.json` in Cell 0. |
| **Anthropic API** | Required. `ANTHROPIC_API_KEY` in Colab Secrets (Settings → Secrets). Model: `claude-sonnet-4-6`. Used for: strategy selection, IDEAS.md generation, idea implementation, re-research decisions, summary. |
| **SemanticScholar** | Free, no auth. Rate limit: 100 req/5min. Implement exponential backoff. Cache all results. |
| **arXiv** | Free, no auth. Use `arxiv` Python package. Cache all results. |
| **Google Drive** | `google.colab.drive.mount('/content/drive')`. No additional auth. |

Never hardcode credentials. Print clear setup instructions if a required key is missing.

---

## 11. Implementation Tasks

### Task 1 — Project Scaffold

- Create `kaggleresearch.ipynb` with cell groups: `[0-INSTALL]`, `[1-CONFIG]`, `[2-BOOTSTRAP]`, `[3-LITERATURE]`, `[4-LOOP]`, `[5-SUMMARY]`
- Create `templates/` with 8 baseline `train.py` files (one per problem type)
- Create `prompts/` with agent system prompts as `.md` files: `bootstrap.md`, `strategy_selection.md`, `ideas_generation.md`, `code_agent.md`, `reresearch.md`, `branch_decision.md`, `summary.md`
- Create `utils/`: `kaggle_api.py`, `literature.py`, `strategy.py`, `experiment_runner.py`, `plateau.py`, `branching.py`, `checkpoint.py`, `display.py`

### Task 2 — Bootstrap (`utils/bootstrap.py`)

- `parse_competition(url) -> CompetitionMeta`
- `classify_problem_type(meta) -> str`
- `select_template(problem_type) -> str`
- `generate_metric_py(meta) -> str`
- `validate_baseline(repo_path, time_budget_min) -> float`

### Task 3 — Literature & Strategy (`utils/literature.py`, `utils/strategy.py`)

- `search_semantic_scholar(query, n, exclude_queries) -> list[Paper]`
- `search_arxiv(query, n, exclude_queries) -> list[Paper]`
- `cache_papers(papers, path)` / `load_cached_papers(path) -> list[Paper]`
- `select_strategy(papers, competition_meta, baseline_score) -> str` — returns `STRATEGY.md` content
- `generate_ideas_md(papers, strategy_md, problem_type, metric) -> str`
- `parse_ideas_md(path) -> list[Idea]`
- `update_idea_status(path, idea_title, status)`
- `append_learning_log(strategy_path, note: str)`
- `Idea` dataclass: `title`, `source`, `risk`, `estimated_gain`, `hypothesis`, `implementation`, `validation`, `status`

### Task 4 — Plateau Detection (`utils/plateau.py`)

- `plateau_triggered(log, window, min_gain_pct) -> bool`
- `summarise_failures(log) -> str` — natural language summary of what failed, used as re-research context

### Task 5 — Re-research (`utils/reresearch.py`)

- `reresearch_new_angle(strategy_md, failure_summary, search_history, literature_depth) -> ReresearchResult`
- `reresearch_reread(cached_papers, strategy_md, failure_summary) -> ReresearchResult`
- `ReresearchResult` dataclass: `outcome` (`new_ideas` | `pivot` | `no_new_ideas`), `new_ideas_md` (optional), `pivot_strategy_md` (optional)

### Task 6 — Branching (`utils/branching.py`)

- `archive_current_branch(repo_path, branch_name)`
- `start_new_branch(repo_path, branch_name, baseline_train_py)`
- `compare_branches(old_best_score, new_best_score) -> str` — returns winning branch name
- `promote_branch(repo_path, branch_name)`

### Task 7 — Experiment Runner (`utils/experiment_runner.py`)

- `implement_idea(idea, train_py_path, strategy_md_path) -> str` — Claude API call, returns new `train.py`
- `validate_patch(repo_path, patched_train_py) -> bool` — checks `metric.py` and `data/` unchanged
- `run_experiment(repo_path, time_budget_min) -> float | None`
- `git_commit(repo_path, message)`
- `git_reset_hard(repo_path)`
- `log_experiment(db_path, experiment)`
- `session_timeout_imminent(time_budget_min) -> bool`

### Task 8 — Checkpoint (`utils/checkpoint.py`)

- `save_checkpoint(path, state: dict)`
- `load_checkpoint(path) -> dict | None`
- `detect_phase(checkpoint) -> str` — returns `exploit | reresearch | branch_compare`

### Task 9 — Display (`utils/display.py`)

- `render_live_table(experiments, current_branch, best_score, baseline_score)`
- `render_strategy_and_ideas_widget(strategy_path, ideas_path)` — side-by-side scrollable
- `render_summary(log_path, checkpoint, baseline_score)`

### Task 10 — Integration & Testing

- Wire all utilities into notebook cells
- Add **"Approve Strategy & Start Loop"** button after Stage 3 (`ipywidgets.Button`)
- **Titanic integration test** end-to-end, including a simulated plateau (set `PLATEAU_WINDOW=3`, `PLATEAU_MIN_GAIN_PCT=99` to force re-research early in testing)
- **Session resume test:** interrupt mid-exploit, restart, verify correct phase detected from `checkpoint.json` and loop resumes at the right idea
- **Branch comparison test:** force a pivot via the simulated plateau, verify both branches are committed and the correct winner is promoted
- **CPU smoke test:** synthetic dataset, no GPU, <5 minutes, validates full pipeline for CI

---

## 12. Explicit Non-Goals (v1)

- Nested pivots — only one branch comparison active at a time
- Multi-GPU or distributed training
- Ensemble search across branches — branches are compared, not combined
- Automatic Kaggle submission — user submits manually from `submissions/submission.csv`
- Reading Kaggle forum discussions — literature source is academic papers only
- Support for video, audio, or 3D data competitions
- A web UI or separate application

---

## 13. Success Criteria

1. User can open `kaggleresearch.ipynb` in Colab, fill in 3 config values, run all cells, and leave it running overnight with no further intervention beyond approving the strategy
2. Titanic integration test passes end-to-end including plateau → re-research → resume exploit
3. Branch comparison test: a forced pivot produces two named git branches, the winner is promoted correctly
4. Session reconnect correctly detects phase from `checkpoint.json` and resumes without repeating work
5. `STRATEGY.md` learning log is updated after every successful experiment
6. All winning experiments are committed to git with messages including branch name and score delta
7. Summary cell shows strategy timeline, branch comparison table (if applicable), and improvement waterfall with arxiv citations
8. CPU smoke test passes in under 5 minutes

---

*KaggleResearch Build Spec v2.0 — pass this document to Claude Code to begin implementation.*