"""
Checkpoint management for session persistence.
All state is saved to Google Drive for Colab resilience.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class BranchInfo:
    """Information about a git branch."""
    name: str
    best_score: float
    status: str  # "active" or "archived"
    strategy_name: str
    experiment_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CheckpointState:
    """Complete checkpoint state for session recovery."""
    phase: str  # "bootstrap", "literature", "exploit", "reresearch", "branch_compare"
    current_branch: str = "main"
    best_score: float = 0.0
    baseline_score: float = 0.0
    ideas_pointer: int = 0  # Index of next idea to try
    plateau_window_scores: List[float] = field(default_factory=list)
    reresearch_attempts: int = 0
    total_experiments: int = 0
    successful_experiments: int = 0
    branches: Dict[str, Dict] = field(default_factory=dict)
    strategy_name: str = ""
    problem_type: str = ""
    competition_slug: str = ""
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        return cls(**data)


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    """
    Save checkpoint state to JSON file.

    Args:
        path: Path to checkpoint.json
        state: CheckpointState object to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update timestamp
    state.last_updated = datetime.now().isoformat()

    # Write atomically by writing to temp file first
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)

    # Rename to actual path (atomic on most filesystems)
    temp_path.rename(path)


def load_checkpoint(path: Path) -> Optional[CheckpointState]:
    """
    Load checkpoint state from JSON file.

    Args:
        path: Path to checkpoint.json

    Returns:
        CheckpointState if file exists, None otherwise
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return CheckpointState.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def detect_phase(checkpoint: Optional[CheckpointState]) -> str:
    """
    Detect the current phase from checkpoint state.

    Args:
        checkpoint: Loaded checkpoint state or None

    Returns:
        Phase string: "bootstrap", "literature", "exploit", "reresearch", "branch_compare"
    """
    if checkpoint is None:
        return "bootstrap"

    return checkpoint.phase


def create_initial_checkpoint(
    competition_slug: str,
    problem_type: str,
    baseline_score: float
) -> CheckpointState:
    """
    Create initial checkpoint after bootstrap phase.

    Args:
        competition_slug: Kaggle competition slug
        problem_type: Detected problem type
        baseline_score: Score from baseline model

    Returns:
        New CheckpointState ready for literature phase
    """
    return CheckpointState(
        phase="literature",
        competition_slug=competition_slug,
        problem_type=problem_type,
        baseline_score=baseline_score,
        best_score=baseline_score,
    )


def update_checkpoint_after_experiment(
    checkpoint: CheckpointState,
    score: Optional[float],
    improved: bool,
    idea_index: int
) -> CheckpointState:
    """
    Update checkpoint after an experiment completes.

    Args:
        checkpoint: Current checkpoint state
        score: Experiment score (None if crashed)
        improved: Whether the experiment improved the best score
        idea_index: Index of the idea that was tried

    Returns:
        Updated checkpoint state
    """
    checkpoint.total_experiments += 1
    checkpoint.ideas_pointer = idea_index + 1

    if score is not None:
        # Add to plateau window
        checkpoint.plateau_window_scores.append(score)
        # Keep only last N scores (will be trimmed by plateau detection)
        checkpoint.plateau_window_scores = checkpoint.plateau_window_scores[-20:]

        if improved:
            checkpoint.successful_experiments += 1
            checkpoint.best_score = score

    return checkpoint


def update_checkpoint_for_reresearch(checkpoint: CheckpointState) -> CheckpointState:
    """
    Update checkpoint when entering re-research phase.

    Args:
        checkpoint: Current checkpoint state

    Returns:
        Updated checkpoint state
    """
    checkpoint.phase = "reresearch"
    checkpoint.reresearch_attempts += 1
    return checkpoint


def update_checkpoint_for_branch(
    checkpoint: CheckpointState,
    new_branch_name: str,
    new_strategy_name: str
) -> CheckpointState:
    """
    Update checkpoint when creating a new branch for pivot.

    Args:
        checkpoint: Current checkpoint state
        new_branch_name: Name of the new branch
        new_strategy_name: Name of the new strategy

    Returns:
        Updated checkpoint state
    """
    # Archive current branch
    old_branch = checkpoint.current_branch
    checkpoint.branches[old_branch] = {
        "best_score": checkpoint.best_score,
        "status": "archived",
        "strategy_name": checkpoint.strategy_name,
        "experiment_count": checkpoint.total_experiments,
    }

    # Set up new branch
    checkpoint.phase = "branch_compare"
    checkpoint.current_branch = new_branch_name
    checkpoint.strategy_name = new_strategy_name
    checkpoint.ideas_pointer = 0
    checkpoint.plateau_window_scores = []

    return checkpoint


def update_checkpoint_after_branch_comparison(
    checkpoint: CheckpointState,
    winner_branch: str,
    loser_branch: str
) -> CheckpointState:
    """
    Update checkpoint after branch comparison completes.

    Args:
        checkpoint: Current checkpoint state
        winner_branch: Name of the winning branch
        loser_branch: Name of the losing branch

    Returns:
        Updated checkpoint state
    """
    # Mark loser as archived
    if loser_branch in checkpoint.branches:
        checkpoint.branches[loser_branch]["status"] = "archived"

    # Continue with winner
    checkpoint.phase = "exploit"
    checkpoint.current_branch = winner_branch
    checkpoint.reresearch_attempts = 0  # Reset for new branch

    return checkpoint
