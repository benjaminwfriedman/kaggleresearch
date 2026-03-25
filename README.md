# KaggleResearch

**Literature-seeded, self-directed autoresearch for Kaggle competitions**

KaggleResearch is a Colab notebook that automatically improves your Kaggle competition score by:

1. Searching academic literature (arXiv, Semantic Scholar) and web content (Tavily) for SOTA approaches
2. Selecting a strategy and generating experiment ideas
3. Running an exploit loop that keeps winning changes
4. Re-researching when progress plateaus
5. Branching the codebase for strategy pivots

## Quick Start

1. **Open in Colab**: Upload `kaggleresearch.ipynb` to Google Colab

2. **Configure**: Edit Cell 1 with your competition URL:
   ```python
   COMPETITION_URL = 'https://www.kaggle.com/competitions/your-competition'
   ```

3. **Set up credentials**:
   - Upload your `kaggle.json` to Colab
   - Add `ANTHROPIC_API_KEY` to Colab secrets (Runtime → Manage secrets)
   - (Optional) Add `TAVILY_API_KEY` for enhanced web search (get one at https://tavily.com)

4. **Run**: Execute cells in order. Approve the strategy, then let it run!

## How It Works

```
[Bootstrap] → Parse competition, create baseline
     ↓
[Literature Review] → Search papers, select strategy, generate IDEAS.md
     ↓
[Human Checkpoint] → Review and approve strategy
     ↓
[Exploit Loop] → Implement ideas, keep improvements, revert failures
     ↓
[Plateau?] → Re-research for new angles or pivot to new strategy
     ↓
[Summary] → Generate report with improvement waterfall
```

## Project Structure

```
kaggleresearch/
├── kaggleresearch.ipynb    # Main Colab notebook
├── templates/              # Baseline train.py for each problem type
│   ├── tabular_classification.py
│   ├── tabular_regression.py
│   ├── image_classification.py
│   ├── image_segmentation.py
│   ├── nlp_classification.py
│   ├── nlp_regression.py
│   ├── time_series.py
│   └── other.py
├── prompts/                # LLM system prompts
│   ├── strategy_selection.md
│   ├── ideas_generation.md
│   ├── code_agent.md
│   └── ...
└── utils/                  # Core utilities
    ├── checkpoint.py       # Session persistence
    ├── kaggle_api.py       # Competition parsing
    ├── literature.py       # Paper search
    ├── strategy.py         # Strategy management
    ├── plateau.py          # Plateau detection
    ├── experiment_runner.py
    └── ...
```

## Supported Problem Types

| Type | Default Model |
|------|---------------|
| tabular-classification | LightGBM |
| tabular-regression | LightGBM |
| image-classification | timm ResNet18 |
| image-segmentation | SMP Unet |
| nlp-classification | DistilBERT |
| nlp-regression | DistilBERT |
| time-series | LightGBM + lag features |

## Configuration Options

```python
COMPETITION_URL      = '...'   # Kaggle competition URL
DRIVE_PATH           = '...'   # Where to save project data
TIME_BUDGET_MIN      = 4       # Minutes per experiment
PLATEAU_WINDOW       = 5       # Experiments before plateau check
PLATEAU_MIN_GAIN_PCT = 1.0     # Min improvement to avoid plateau
BRANCH_COMPARE_N     = 5       # Experiments before branch comparison
LITERATURE_DEPTH     = 10      # Papers to retrieve per search
```

## Requirements

- Google Colab (free tier works, A100 recommended)
- Kaggle API credentials
- Anthropic API key
- (Optional) Tavily API key for enhanced web search (`TAVILY_API_KEY`)

## License

MIT

## Acknowledgments

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
