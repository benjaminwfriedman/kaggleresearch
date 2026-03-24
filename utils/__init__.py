# KaggleResearch Utilities
"""
Utilities for the KaggleResearch autoresearch system.
"""

from .checkpoint import save_checkpoint, load_checkpoint, detect_phase
from .kaggle_api import parse_competition, classify_problem_type, CompetitionMeta
from .literature import (
    search_semantic_scholar,
    search_arxiv,
    cache_papers,
    load_cached_papers,
    Paper,
)
from .strategy import (
    select_strategy,
    generate_ideas_md,
    parse_ideas_md,
    update_idea_status,
    append_learning_log,
    Idea,
)
from .plateau import plateau_triggered, summarise_failures
from .reresearch import (
    reresearch_new_angle,
    reresearch_reread,
    ReresearchResult,
)
from .branching import (
    archive_current_branch,
    start_new_branch,
    compare_branches,
    promote_branch,
)
from .experiment_runner import (
    implement_idea,
    validate_patch,
    run_experiment,
    git_commit,
    git_reset_hard,
    log_experiment,
    session_timeout_imminent,
)
from .display import (
    render_live_table,
    render_strategy_and_ideas_widget,
    render_summary,
)

__all__ = [
    # checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "detect_phase",
    # kaggle_api
    "parse_competition",
    "classify_problem_type",
    "CompetitionMeta",
    # literature
    "search_semantic_scholar",
    "search_arxiv",
    "cache_papers",
    "load_cached_papers",
    "Paper",
    # strategy
    "select_strategy",
    "generate_ideas_md",
    "parse_ideas_md",
    "update_idea_status",
    "append_learning_log",
    "Idea",
    # plateau
    "plateau_triggered",
    "summarise_failures",
    # reresearch
    "reresearch_new_angle",
    "reresearch_reread",
    "ReresearchResult",
    # branching
    "archive_current_branch",
    "start_new_branch",
    "compare_branches",
    "promote_branch",
    # experiment_runner
    "implement_idea",
    "validate_patch",
    "run_experiment",
    "git_commit",
    "git_reset_hard",
    "log_experiment",
    "session_timeout_imminent",
    # display
    "render_live_table",
    "render_strategy_and_ideas_widget",
    "render_summary",
]
