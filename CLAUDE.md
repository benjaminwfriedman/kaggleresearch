# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KaggleResearch is a Colab-native autoresearch system that automatically improves Kaggle competition scores through literature-seeded, self-directed experimentation. It implements a ratchet loop: research → strategize → exploit → plateau detection → re-research → branch or refine.

## Running the System

The primary entrypoint is `kaggleresearch.ipynb` - a Jupyter notebook designed for Google Colab. The notebook cells are organized:
1. Install dependencies and configure credentials
2. Set competition URL and config parameters
3. Bootstrap: parse competition, generate baseline
4. Literature review: search papers, generate STRATEGY.md and IDEAS.md
5. Research loop: implement ideas, evaluate, keep improvements
6. Summary: render results

## Architecture

### Core Loop (`utils/research_loop.py`)
The main research loop orchestrates:
- `do_initial_literature_review()` - searches academic papers and generates strategy
- `run_single_experiment()` - implements one idea from IDEAS.md, runs train.py, evaluates
- `run_reresearch()` - triggers when plateau detected or ideas exhausted

### Key Components

**Idea Tree (`utils/idea_tree.py`)**: Tree structure tracking explored experiments. Each node stores git commit SHA for backtracking. Supports UCB1 selection and depth-first/breadth-first exploration.

**Experiment Runner (`utils/experiment_runner.py`)**:
- `implement_idea()` - calls Claude API to modify train.py based on an IDEA entry
- `validate_patch()` - ensures generated code has required patterns, valid syntax, no forbidden imports
- `run_experiment()` - executes train.py with timeout, parses score from stdout

**Checkpoint (`utils/checkpoint.py`)**: Session persistence to Google Drive. `CheckpointState` dataclass tracks phase, scores, tree position, branch info. All state survives Colab disconnects.

**Strategy (`utils/strategy.py`)**: Parses IDEAS.md entries (specific schema with title, source, risk, hypothesis, implementation, validation). Updates idea statuses and STRATEGY.md learning log.

### File Layout at Runtime

```
project_dir/
├── checkpoint.json      # Session state
├── idea_tree.json       # Exploration tree
├── STRATEGY.md          # Current strategy document
├── IDEAS.md             # Ordered experiment queue
├── experiment_log.db    # SQLite experiment history
├── repo/
│   ├── train.py         # Modified by code agent
│   ├── metric.py        # Never modified
│   └── data/            # Competition data (read-only)
└── literature_cache/
    ├── papers.json
    └── search_history.json
```

### Constraints Enforced by Code Agent

The code agent prompt (`prompts/code_agent.md`) specifies:
- Must return complete train.py with `def main()` and `if __name__ == "__main__":`
- Cannot use `ray` (forbidden import)
- Available libraries: pandas, numpy, sklearn, lightgbm, xgboost, catboost, torch, transformers, timm, optuna, albumentations, etc.

## Key Design Patterns

**Ratchet Loop**: Only keep changes that improve the score. Revert failures via `backup_train_py()`/`restore_train_py()` or git reset.

**Plateau Detection** (`utils/plateau.py`): Triggers re-research when recent experiments show < threshold improvement over a window.

**Strategy Pivots** (`utils/branching.py`): When re-research suggests abandoning current approach, archive to git branch, reset train.py to baseline, start new strategy.

## Config Parameters

Key settings in the notebook config cell:
- `TIME_BUDGET_MIN`: Minutes per experiment (scales by GPU type)
- `PLATEAU_WINDOW` / `PLATEAU_MIN_GAIN_PCT`: Plateau detection thresholds
- `LITERATURE_DEPTH`: Papers to retrieve per search
- `exploration_mode`: "df" (depth-first) or "bf" (breadth-first)

## Testing
**IMPORTANT:**
Never Mock Code from within this project.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=utils --cov-report=term-missing

# Run specific test module
pytest tests/test_plateau.py -v

# Run tests matching a pattern
pytest tests/ -k "plateau" -v
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_plateau.py          # Plateau detection logic
├── test_checkpoint.py       # State management
├── test_idea_tree.py        # Tree operations, UCB1
├── test_strategy.py         # IDEAS.md parsing
└── test_kaggle_api.py       # URL extraction, metric inference
```

### Testing Philosophy

- **Pure functions first**: Most utils have pure functions testable without mocking
- **Fixtures for complex state**: Use `conftest.py` fixtures for IdeaTree, CheckpointState
- **tmp_path for file I/O**: Use pytest's `tmp_path` fixture for checkpoint save/load tests
- **Mock external APIs**: LLM calls and Kaggle API require mocking (see `@patch` usage)

### Integration Testing

Manual integration testing via Titanic competition with forced plateau:
```python
PLATEAU_MIN_GAIN_PCT = 99  # Forces plateau after any run
```

## Common Modifications

When modifying experiment logic, update:
1. `utils/research_loop.py` - main loop control flow
2. `utils/experiment_runner.py` - how ideas become code changes
3. `prompts/code_agent.md` - instructions to Claude for generating code

When changing idea/strategy management:
1. `utils/strategy.py` - IDEAS.md parsing and status updates
2. `utils/reresearch.py` - re-research decision logic


## Commiting, Branching and PRs

- Create a descriptive branch name 
    - feature/{..}
    - bug/{..}
    - refactor/{..}

- We don't credit claude in Commits or PRs
