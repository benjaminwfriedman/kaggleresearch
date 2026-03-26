"""
Checkpoint management for session persistence.
All state is saved to Google Drive for Colab resilience.
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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
    phase: str  # "bootstrap", "research", "halted"
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
    # Tree navigation state
    run_id: str = ""  # Unique ID for this research run (e.g., "run-001")
    current_node_id: str = ""  # Current position in idea tree
    exploration_mode: str = "df"  # "df" (depth-first) or "bf" (breadth-first)
    # Competition metadata (name, metric, metric_direction)
    competition_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary, handling missing fields for backward compatibility."""
        # Add default values for fields that may not exist in old checkpoints
        defaults = {
            'run_id': '',
            'current_node_id': '',
            'exploration_mode': 'df',
            'competition_meta': {},
        }
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
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
        Phase string: "bootstrap", "research", "halted"
    """
    if checkpoint is None:
        return "bootstrap"

    # Map old phase names to new simplified phases for backward compatibility
    phase = checkpoint.phase
    if phase in ('literature', 'exploit', 'reresearch', 'branch_compare'):
        return 'research'

    return phase


def create_initial_checkpoint(
    competition_slug: str,
    problem_type: str,
    baseline_score: float,
    run_id: str = "",
    exploration_mode: str = "df",
    competition_meta: Optional[Dict[str, Any]] = None
) -> CheckpointState:
    """
    Create initial checkpoint after bootstrap/setup phase.

    Args:
        competition_slug: Kaggle competition slug
        problem_type: Detected problem type
        baseline_score: Score from baseline model
        run_id: Unique ID for this research run
        exploration_mode: "df" (depth-first) or "bf" (breadth-first)
        competition_meta: Competition metadata (name, metric, metric_direction)

    Returns:
        New CheckpointState ready for research phase
    """
    return CheckpointState(
        phase="research",
        competition_slug=competition_slug,
        problem_type=problem_type,
        baseline_score=baseline_score,
        best_score=baseline_score,
        run_id=run_id,
        exploration_mode=exploration_mode,
        competition_meta=competition_meta or {},
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
    Update checkpoint when entering re-research (still within research phase).

    Args:
        checkpoint: Current checkpoint state

    Returns:
        Updated checkpoint state
    """
    # Stay in research phase - re-research is internal to the loop
    checkpoint.phase = "research"
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

    # Set up new branch (stay in research phase)
    checkpoint.phase = "research"
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

    # Continue with winner (stay in research phase)
    checkpoint.phase = "research"
    checkpoint.current_branch = winner_branch
    checkpoint.reresearch_attempts = 0  # Reset for new branch

    return checkpoint


def generate_run_id(project_dir: Path) -> str:
    """
    Generate a unique run ID for a new research session.

    Reads the archive manifest to determine the next run number.

    Args:
        project_dir: Path to the project directory

    Returns:
        New run ID (e.g., "run-001", "run-002")
    """
    manifest_path = project_dir / 'archive_manifest.json'

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            run_count = len(manifest) + 1
        except (json.JSONDecodeError, IOError):
            run_count = 1
    else:
        run_count = 1

    return f"run-{run_count:03d}"


def archive_and_reset(project_dir: Path, competition_slug: str) -> str:
    """
    Archive current run and prepare for fresh start.

    Moves all state files to an archive folder, preserving them for reference.
    Git branches are preserved (searchable via branch naming convention).

    Args:
        project_dir: Path to the project directory
        competition_slug: Kaggle competition slug

    Returns:
        New run_id for the fresh session
    """
    # Generate archive name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Find current run_id from checkpoint
    checkpoint_path = project_dir / 'checkpoint.json'
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        old_run_id = checkpoint.run_id if checkpoint and checkpoint.run_id else "unknown"
    else:
        old_run_id = "unknown"

    archive_name = f"archive/{old_run_id}-{timestamp}"
    archive_dir = project_dir / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Files to archive (move, not copy - to indicate fresh start)
    files_to_archive = [
        'checkpoint.json',
        'idea_tree.json',
        'IDEAS.md',
        'STRATEGY.md',
        'experiment_log.db',
    ]

    archived_files = []
    for filename in files_to_archive:
        src = project_dir / filename
        if src.exists():
            dst = archive_dir / filename
            shutil.move(str(src), str(dst))
            archived_files.append(filename)

    # Update archive manifest
    manifest_path = project_dir / 'archive_manifest.json'
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, IOError):
            manifest = []
    else:
        manifest = []

    manifest.append({
        'run_id': old_run_id,
        'archived_at': timestamp,
        'archive_path': str(archive_dir),
        'competition': competition_slug,
        'files': archived_files,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Generate new run_id
    new_run_id = f"run-{len(manifest) + 1:03d}"

    print(f"✓ Archived previous run: {old_run_id} → {archive_name}")
    print(f"  Files archived: {', '.join(archived_files)}")
    print(f"✓ Starting fresh with run_id: {new_run_id}")

    return new_run_id


def get_or_create_run_id(project_dir: Path, checkpoint: Optional[CheckpointState]) -> str:
    """
    Get existing run_id from checkpoint or create a new one.

    Args:
        project_dir: Path to the project directory
        checkpoint: Loaded checkpoint state or None

    Returns:
        Run ID string
    """
    if checkpoint and checkpoint.run_id:
        return checkpoint.run_id

    return generate_run_id(project_dir)
