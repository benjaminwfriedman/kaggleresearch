"""Tests for plateau detection logic."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plateau import (
    plateau_triggered,
    plateau_triggered_directional,
    summarise_failures,
    identify_failure_patterns,
    calculate_improvement_rate,
    should_continue_loop,
    ExperimentResult,
)


class TestPlateauTriggered:
    """Tests for plateau_triggered function."""

    def test_no_plateau_with_improvement(self):
        """Scores improving - no plateau."""
        scores = [0.70, 0.72, 0.75, 0.78, 0.82]
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is False

    def test_plateau_detected_flat_scores(self):
        """Flat scores should trigger plateau."""
        scores = [0.80, 0.80, 0.80, 0.80, 0.80]
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is True

    def test_plateau_detected_minimal_improvement(self):
        """Very small improvement below threshold triggers plateau."""
        scores = [0.800, 0.801, 0.802, 0.801, 0.802]
        # Gain is ~0.25%, threshold is 1%
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is True

    def test_no_plateau_insufficient_data(self):
        """Not enough data - no plateau."""
        scores = [0.80, 0.80, 0.80]
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is False

    def test_plateau_uses_last_window_scores(self):
        """Should only consider last N scores."""
        # Early improvement, then flat
        scores = [0.50, 0.60, 0.70, 0.80, 0.80, 0.80, 0.80, 0.80]
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is True

    def test_none_values_filtered_out(self):
        """None values (crashes) should be excluded."""
        scores = [0.75, None, 0.76, None, 0.77, 0.78, 0.79]
        # Valid scores: [0.75, 0.76, 0.77, 0.78, 0.79] - improving
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is False

    def test_all_none_returns_false(self):
        """All crashes should not trigger plateau."""
        scores = [None, None, None, None, None]
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is False

    def test_zero_scores_handled(self):
        """Zero scores should not cause division error."""
        scores = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Should not raise, returns plateau
        result = plateau_triggered(scores, window=5, min_gain_pct=1.0)
        assert result is True

    def test_negative_scores_handled(self):
        """Negative scores (e.g., log loss) should work."""
        scores = [-0.5, -0.4, -0.3, -0.2, -0.1]
        # Improving (less negative)
        result = plateau_triggered(scores, window=5, min_gain_pct=1.0)
        assert isinstance(result, bool)

    def test_single_window_exactly_at_threshold(self):
        """Gain exactly at threshold should not trigger plateau."""
        # 1% gain over worst
        scores = [1.00, 1.005, 1.008, 1.01, 1.01]
        # best=1.01, worst=1.00, gain=1%
        assert plateau_triggered(scores, window=5, min_gain_pct=1.0) is False


class TestPlateauTriggeredDirectional:
    """Tests for direction-aware plateau detection."""

    def test_higher_is_better_improving(self):
        """Higher-is-better with improvement."""
        scores = [0.70, 0.75, 0.80, 0.85, 0.90]
        assert plateau_triggered_directional(
            scores, window=5, min_gain_pct=1.0, higher_is_better=True
        ) is False

    def test_higher_is_better_plateau(self):
        """Higher-is-better with flat scores."""
        scores = [0.85, 0.85, 0.85, 0.85, 0.85]
        assert plateau_triggered_directional(
            scores, window=5, min_gain_pct=1.0, higher_is_better=True
        ) is True

    def test_lower_is_better_improving(self):
        """Lower-is-better (RMSE) with improvement."""
        scores = [0.50, 0.45, 0.40, 0.35, 0.30]  # Decreasing = good
        assert plateau_triggered_directional(
            scores, window=5, min_gain_pct=1.0, higher_is_better=False
        ) is False

    def test_lower_is_better_plateau(self):
        """Lower-is-better with flat scores."""
        scores = [0.30, 0.30, 0.30, 0.30, 0.30]
        assert plateau_triggered_directional(
            scores, window=5, min_gain_pct=1.0, higher_is_better=False
        ) is True

    def test_lower_is_better_getting_worse(self):
        """Lower-is-better with increasing scores (bad)."""
        scores = [0.30, 0.35, 0.40, 0.45, 0.50]  # Getting worse
        assert plateau_triggered_directional(
            scores, window=5, min_gain_pct=1.0, higher_is_better=False
        ) is True


class TestSummariseFailures:
    """Tests for failure summary generation."""

    def test_no_failures(self):
        """No failures returns appropriate message."""
        experiments = [
            ExperimentResult("Idea 1", 0.85, "improved", 0),
            ExperimentResult("Idea 2", 0.86, "improved", 1),
        ]
        summary = summarise_failures(experiments)
        assert "No failed experiments" in summary

    def test_summarizes_no_improvement(self):
        """Summarizes no_improvement experiments."""
        experiments = [
            ExperimentResult("Reduce LR", 0.80, "no_improvement", 0),
            ExperimentResult("Add dropout", 0.79, "no_improvement", 1),
        ]
        summary = summarise_failures(experiments)
        assert "No Improvement" in summary
        assert "Reduce LR" in summary
        assert "Add dropout" in summary

    def test_summarizes_crashed(self):
        """Summarizes crashed experiments."""
        experiments = [
            ExperimentResult("Bad idea", None, "crashed", 0),
        ]
        summary = summarise_failures(experiments)
        assert "Crashed" in summary
        assert "Bad idea" in summary

    def test_mixed_failures(self):
        """Handles mix of failure types."""
        experiments = [
            ExperimentResult("Idea 1", 0.80, "no_improvement", 0),
            ExperimentResult("Idea 2", None, "crashed", 1),
            ExperimentResult("Idea 3", 0.85, "improved", 2),  # Not a failure
        ]
        summary = summarise_failures(experiments)
        assert "No Improvement" in summary
        assert "Crashed" in summary
        assert "Idea 3" not in summary  # Successful shouldn't appear

    def test_respects_max_entries(self):
        """Respects max_entries limit."""
        experiments = [
            ExperimentResult(f"Idea {i}", 0.75, "no_improvement", i)
            for i in range(15)
        ]
        summary = summarise_failures(experiments, max_entries=5)
        assert "... and 10 more" in summary

    def test_includes_scores_in_summary(self):
        """Includes scores for no_improvement experiments."""
        experiments = [
            ExperimentResult("Test idea", 0.7532, "no_improvement", 0),
        ]
        summary = summarise_failures(experiments)
        assert "0.7532" in summary


class TestIdentifyFailurePatterns:
    """Tests for failure pattern identification."""

    def test_detects_learning_rate_pattern(self):
        """Detects multiple learning rate failures."""
        experiments = [
            ExperimentResult("Reduce learning rate", 0.80, "no_improvement", 0),
            ExperimentResult("Increase LR", 0.79, "no_improvement", 1),
        ]
        patterns = identify_failure_patterns(experiments)
        assert any("learning rate" in p.lower() for p in patterns)

    def test_detects_high_crash_rate(self):
        """Detects high crash rate pattern."""
        experiments = [
            ExperimentResult("Idea 1", None, "crashed", 0),
            ExperimentResult("Idea 2", None, "crashed", 1),
            ExperimentResult("Idea 3", None, "crashed", 2),
            ExperimentResult("Idea 4", 0.80, "no_improvement", 3),
        ]
        patterns = identify_failure_patterns(experiments)
        assert any("crash rate" in p.lower() for p in patterns)

    def test_detects_all_recent_failures(self):
        """Detects when last 5 experiments all failed."""
        experiments = [
            ExperimentResult("Idea 1", 0.80, "no_improvement", 0),
            ExperimentResult("Idea 2", 0.79, "no_improvement", 1),
            ExperimentResult("Idea 3", None, "crashed", 2),
            ExperimentResult("Idea 4", 0.78, "no_improvement", 3),
            ExperimentResult("Idea 5", 0.77, "no_improvement", 4),
        ]
        patterns = identify_failure_patterns(experiments)
        assert any("Last 5" in p for p in patterns)

    def test_empty_list_returns_empty(self):
        """Empty experiment list returns no patterns."""
        patterns = identify_failure_patterns([])
        assert patterns == []


class TestCalculateImprovementRate:
    """Tests for improvement rate calculation."""

    def test_all_improved(self):
        """All experiments improved."""
        experiments = [
            ExperimentResult("Idea 1", 0.81, "improved", 0),
            ExperimentResult("Idea 2", 0.82, "improved", 1),
        ]
        assert calculate_improvement_rate(experiments) == 1.0

    def test_none_improved(self):
        """No experiments improved."""
        experiments = [
            ExperimentResult("Idea 1", 0.80, "no_improvement", 0),
            ExperimentResult("Idea 2", None, "crashed", 1),
        ]
        assert calculate_improvement_rate(experiments) == 0.0

    def test_partial_improvement(self):
        """Some experiments improved."""
        experiments = [
            ExperimentResult("Idea 1", 0.81, "improved", 0),
            ExperimentResult("Idea 2", 0.80, "no_improvement", 1),
            ExperimentResult("Idea 3", 0.82, "improved", 2),
            ExperimentResult("Idea 4", None, "crashed", 3),
        ]
        assert calculate_improvement_rate(experiments) == 0.5

    def test_with_window(self):
        """Respects window parameter."""
        experiments = [
            ExperimentResult("Idea 1", 0.81, "improved", 0),  # Outside window
            ExperimentResult("Idea 2", 0.80, "no_improvement", 1),
            ExperimentResult("Idea 3", 0.82, "improved", 2),
        ]
        # Window of 2 should only consider last 2
        rate = calculate_improvement_rate(experiments, window=2)
        assert rate == 0.5

    def test_empty_list(self):
        """Empty list returns 0."""
        assert calculate_improvement_rate([]) == 0.0


class TestShouldContinueLoop:
    """Tests for loop continuation logic."""

    def test_continues_with_ideas_remaining(self):
        """Continues when ideas remain."""
        experiments = []
        assert should_continue_loop(experiments, ideas_remaining=5) is True

    def test_stops_when_no_ideas(self):
        """Stops when no ideas remain."""
        experiments = [
            ExperimentResult("Idea 1", 0.81, "improved", 0),
        ]
        assert should_continue_loop(experiments, ideas_remaining=0) is False

    def test_stops_after_consecutive_failures(self):
        """Stops after too many consecutive failures."""
        experiments = [
            ExperimentResult(f"Idea {i}", 0.80, "no_improvement", i)
            for i in range(10)
        ]
        assert should_continue_loop(
            experiments, ideas_remaining=5, max_consecutive_failures=10
        ) is False

    def test_continues_after_success_resets_count(self):
        """Success resets consecutive failure count."""
        experiments = [
            ExperimentResult("Fail 1", 0.80, "no_improvement", 0),
            ExperimentResult("Fail 2", 0.79, "no_improvement", 1),
            ExperimentResult("Success", 0.85, "improved", 2),  # Resets count
            ExperimentResult("Fail 3", 0.84, "no_improvement", 3),
            ExperimentResult("Fail 4", 0.83, "no_improvement", 4),
        ]
        # Only 2 consecutive failures after success
        assert should_continue_loop(
            experiments, ideas_remaining=5, max_consecutive_failures=5
        ) is True

    def test_empty_experiments_continues(self):
        """Empty experiment list should continue."""
        assert should_continue_loop([], ideas_remaining=5) is True
