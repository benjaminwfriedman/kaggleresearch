"""
Plateau detection for the experiment loop.
Triggers re-research when progress stalls.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    idea_title: str
    score: Optional[float]
    status: str  # "improved", "no_improvement", "crashed"
    idea_index: int


def plateau_triggered(
    recent_scores: List[float],
    window: int,
    min_gain_pct: float
) -> bool:
    """
    Check if the experiment loop has plateaued.

    A plateau is detected when the improvement over the last `window` experiments
    is less than `min_gain_pct` percent.

    Args:
        recent_scores: List of recent experiment scores (excluding crashes)
        window: Number of experiments to consider
        min_gain_pct: Minimum cumulative % gain required

    Returns:
        True if plateau detected, False otherwise
    """
    # Filter out None values (crashes)
    valid_scores = [s for s in recent_scores if s is not None]

    # Need at least `window` scores to detect plateau
    if len(valid_scores) < window:
        return False

    # Get the last `window` scores
    window_scores = valid_scores[-window:]

    best_in_window = max(window_scores)
    worst_in_window = min(window_scores)

    # Avoid division by zero
    if worst_in_window == 0:
        worst_in_window = 1e-10

    # Calculate percentage gain
    gain_pct = (best_in_window - worst_in_window) / abs(worst_in_window) * 100

    return gain_pct < min_gain_pct


def plateau_triggered_directional(
    recent_scores: List[float],
    window: int,
    min_gain_pct: float,
    higher_is_better: bool = True
) -> bool:
    """
    Check if plateau triggered, accounting for metric direction.

    Args:
        recent_scores: List of recent experiment scores
        window: Number of experiments to consider
        min_gain_pct: Minimum improvement required
        higher_is_better: True if higher scores are better

    Returns:
        True if plateau detected
    """
    valid_scores = [s for s in recent_scores if s is not None]

    if len(valid_scores) < window:
        return False

    window_scores = valid_scores[-window:]

    if higher_is_better:
        best_in_window = max(window_scores)
        reference = min(window_scores)
    else:
        best_in_window = min(window_scores)  # Lower is better
        reference = max(window_scores)

    if reference == 0:
        reference = 1e-10

    # For "lower is better", improvement is when score decreases
    if higher_is_better:
        gain_pct = (best_in_window - reference) / abs(reference) * 100
    else:
        gain_pct = (reference - best_in_window) / abs(reference) * 100

    return gain_pct < min_gain_pct


def summarise_failures(
    experiments: List[ExperimentResult],
    max_entries: int = 10
) -> str:
    """
    Generate a natural language summary of failed experiments.
    Used as context for re-research.

    Args:
        experiments: List of experiment results
        max_entries: Maximum number of entries to include

    Returns:
        Summary string for re-research prompt
    """
    failed = [e for e in experiments if e.status in ('no_improvement', 'crashed')]

    if not failed:
        return "No failed experiments to summarize."

    lines = ["## Failed Experiments Summary\n"]

    # Group by failure type
    no_improvement = [e for e in failed if e.status == 'no_improvement']
    crashed = [e for e in failed if e.status == 'crashed']

    if no_improvement:
        lines.append(f"### No Improvement ({len(no_improvement)} experiments)")
        for exp in no_improvement[:max_entries]:
            score_str = f" (score: {exp.score:.4f})" if exp.score else ""
            lines.append(f"- {exp.idea_title}{score_str}")
        if len(no_improvement) > max_entries:
            lines.append(f"  ... and {len(no_improvement) - max_entries} more")
        lines.append("")

    if crashed:
        lines.append(f"### Crashed ({len(crashed)} experiments)")
        for exp in crashed[:max_entries]:
            lines.append(f"- {exp.idea_title}")
        if len(crashed) > max_entries:
            lines.append(f"  ... and {len(crashed) - max_entries} more")
        lines.append("")

    # Identify patterns
    patterns = identify_failure_patterns(failed)
    if patterns:
        lines.append("### Observed Patterns")
        for pattern in patterns:
            lines.append(f"- {pattern}")

    return '\n'.join(lines)


def identify_failure_patterns(experiments: List[ExperimentResult]) -> List[str]:
    """
    Identify patterns in failed experiments.

    Args:
        experiments: List of failed experiment results

    Returns:
        List of pattern descriptions
    """
    patterns = []

    titles = [e.idea_title.lower() for e in experiments]

    # Check for common themes
    theme_keywords = {
        'learning rate': ['learning rate', 'lr', 'step size'],
        'regularization': ['regularization', 'dropout', 'weight decay', 'l1', 'l2'],
        'architecture': ['layer', 'hidden', 'depth', 'width', 'architecture'],
        'augmentation': ['augment', 'transform', 'flip', 'rotate'],
        'feature engineering': ['feature', 'encoding', 'transform'],
        'hyperparameter': ['hyperparameter', 'batch size', 'epoch'],
    }

    for theme, keywords in theme_keywords.items():
        matches = sum(1 for t in titles if any(kw in t for kw in keywords))
        if matches >= 2:
            patterns.append(f"Multiple {theme} changes failed ({matches} experiments)")

    # Check for crash rate
    crash_rate = sum(1 for e in experiments if e.status == 'crashed') / len(experiments)
    if crash_rate > 0.3:
        patterns.append(f"High crash rate ({crash_rate:.0%}) suggests implementation issues")

    # Check if all recent experiments failed
    if len(experiments) >= 5:
        recent_all_failed = all(e.status != 'improved' for e in experiments[-5:])
        if recent_all_failed:
            patterns.append("Last 5 experiments all failed - may need strategic pivot")

    return patterns


def calculate_improvement_rate(
    experiments: List[ExperimentResult],
    window: Optional[int] = None
) -> float:
    """
    Calculate the rate of successful experiments.

    Args:
        experiments: List of experiment results
        window: Optional window to limit calculation

    Returns:
        Success rate as a fraction
    """
    if not experiments:
        return 0.0

    if window:
        experiments = experiments[-window:]

    successful = sum(1 for e in experiments if e.status == 'improved')
    return successful / len(experiments)


def should_continue_loop(
    experiments: List[ExperimentResult],
    ideas_remaining: int,
    max_consecutive_failures: int = 10
) -> bool:
    """
    Determine if the exploit loop should continue.

    Args:
        experiments: List of experiment results
        ideas_remaining: Number of ideas left to try
        max_consecutive_failures: Max failures before suggesting stop

    Returns:
        True if loop should continue
    """
    if ideas_remaining <= 0:
        return False

    if not experiments:
        return True

    # Check consecutive failures
    consecutive_failures = 0
    for exp in reversed(experiments):
        if exp.status == 'improved':
            break
        consecutive_failures += 1

    if consecutive_failures >= max_consecutive_failures:
        return False

    return True
