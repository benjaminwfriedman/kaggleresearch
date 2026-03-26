"""Tests for checkpoint state management."""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint import (
    CheckpointState,
    BranchInfo,
    save_checkpoint,
    load_checkpoint,
    detect_phase,
    create_initial_checkpoint,
    update_checkpoint_after_experiment,
    update_checkpoint_for_reresearch,
    update_checkpoint_for_branch,
    update_checkpoint_after_branch_comparison,
    generate_run_id,
)


class TestCheckpointState:
    """Tests for CheckpointState dataclass."""

    def test_to_dict(self, sample_checkpoint):
        """Test serialization to dict."""
        data = sample_checkpoint.to_dict()

        assert data["phase"] == "research"
        assert data["best_score"] == 0.85
        assert data["competition_slug"] == "titanic"
        assert "competition_meta" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "phase": "research",
            "current_branch": "main",
            "best_score": 0.90,
            "baseline_score": 0.75,
            "ideas_pointer": 5,
            "plateau_window_scores": [0.85, 0.87, 0.90],
            "reresearch_attempts": 1,
            "total_experiments": 10,
            "successful_experiments": 5,
            "branches": {},
            "strategy_name": "Test Strategy",
            "problem_type": "tabular-classification",
            "competition_slug": "test-comp",
            "last_updated": "2024-01-01T00:00:00",
            "session_start": "2024-01-01T00:00:00",
            "run_id": "run-001",
            "current_node_id": "node123",
            "exploration_mode": "bf",
            "competition_meta": {"metric": "accuracy"},
        }

        checkpoint = CheckpointState.from_dict(data)

        assert checkpoint.phase == "research"
        assert checkpoint.best_score == 0.90
        assert checkpoint.exploration_mode == "bf"
        assert checkpoint.competition_meta["metric"] == "accuracy"

    def test_from_dict_with_missing_fields(self):
        """Test backward compatibility with missing fields."""
        # Older checkpoint format without new fields
        data = {
            "phase": "exploit",
            "current_branch": "main",
            "best_score": 0.80,
            "baseline_score": 0.75,
            "ideas_pointer": 3,
            "plateau_window_scores": [],
            "reresearch_attempts": 0,
            "total_experiments": 3,
            "successful_experiments": 2,
            "branches": {},
            "strategy_name": "Old Strategy",
            "problem_type": "tabular-regression",
            "competition_slug": "old-comp",
            "last_updated": "2023-01-01T00:00:00",
            "session_start": "2023-01-01T00:00:00",
            # Missing: run_id, current_node_id, exploration_mode, competition_meta
        }

        checkpoint = CheckpointState.from_dict(data)

        # Should use defaults
        assert checkpoint.run_id == ""
        assert checkpoint.current_node_id == ""
        assert checkpoint.exploration_mode == "df"
        assert checkpoint.competition_meta == {}


class TestSaveLoadCheckpoint:
    """Tests for checkpoint persistence."""

    def test_save_and_load_round_trip(self, tmp_path, sample_checkpoint):
        """Test save/load preserves data."""
        checkpoint_path = tmp_path / "checkpoint.json"

        save_checkpoint(checkpoint_path, sample_checkpoint)
        loaded = load_checkpoint(checkpoint_path)

        assert loaded is not None
        assert loaded.phase == sample_checkpoint.phase
        assert loaded.best_score == sample_checkpoint.best_score
        assert loaded.competition_slug == sample_checkpoint.competition_slug
        assert loaded.run_id == sample_checkpoint.run_id

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading nonexistent file returns None."""
        checkpoint_path = tmp_path / "nonexistent.json"
        assert load_checkpoint(checkpoint_path) is None

    def test_load_corrupted_returns_none(self, tmp_path):
        """Loading corrupted JSON returns None."""
        checkpoint_path = tmp_path / "corrupted.json"
        checkpoint_path.write_text("{ invalid json }")

        result = load_checkpoint(checkpoint_path)
        assert result is None

    def test_save_creates_parent_dirs(self, tmp_path):
        """Save creates parent directories if needed."""
        checkpoint_path = tmp_path / "deep" / "nested" / "checkpoint.json"
        checkpoint = CheckpointState(phase="bootstrap")

        save_checkpoint(checkpoint_path, checkpoint)

        assert checkpoint_path.exists()

    def test_save_updates_timestamp(self, tmp_path, sample_checkpoint):
        """Save updates last_updated timestamp."""
        checkpoint_path = tmp_path / "checkpoint.json"
        original_timestamp = sample_checkpoint.last_updated

        save_checkpoint(checkpoint_path, sample_checkpoint)

        # Timestamp should be updated
        assert sample_checkpoint.last_updated != original_timestamp


class TestDetectPhase:
    """Tests for phase detection."""

    def test_none_checkpoint_returns_bootstrap(self):
        """None checkpoint means bootstrap phase."""
        assert detect_phase(None) == "bootstrap"

    def test_research_phase_preserved(self, sample_checkpoint):
        """Research phase preserved."""
        sample_checkpoint.phase = "research"
        assert detect_phase(sample_checkpoint) == "research"

    def test_halted_phase_preserved(self, sample_checkpoint):
        """Halted phase preserved."""
        sample_checkpoint.phase = "halted"
        assert detect_phase(sample_checkpoint) == "halted"

    def test_old_exploit_phase_mapped_to_research(self, sample_checkpoint):
        """Old 'exploit' phase mapped to 'research'."""
        sample_checkpoint.phase = "exploit"
        assert detect_phase(sample_checkpoint) == "research"

    def test_old_reresearch_phase_mapped_to_research(self, sample_checkpoint):
        """Old 'reresearch' phase mapped to 'research'."""
        sample_checkpoint.phase = "reresearch"
        assert detect_phase(sample_checkpoint) == "research"

    def test_old_literature_phase_mapped_to_research(self, sample_checkpoint):
        """Old 'literature' phase mapped to 'research'."""
        sample_checkpoint.phase = "literature"
        assert detect_phase(sample_checkpoint) == "research"


class TestCreateInitialCheckpoint:
    """Tests for initial checkpoint creation."""

    def test_creates_valid_checkpoint(self):
        """Creates checkpoint with correct values."""
        checkpoint = create_initial_checkpoint(
            competition_slug="titanic",
            problem_type="tabular-classification",
            baseline_score=0.75,
            run_id="run-001",
            exploration_mode="df",
            competition_meta={"metric": "accuracy"},
        )

        assert checkpoint.phase == "research"
        assert checkpoint.competition_slug == "titanic"
        assert checkpoint.problem_type == "tabular-classification"
        assert checkpoint.baseline_score == 0.75
        assert checkpoint.best_score == 0.75  # Starts at baseline
        assert checkpoint.run_id == "run-001"
        assert checkpoint.exploration_mode == "df"
        assert checkpoint.competition_meta["metric"] == "accuracy"

    def test_defaults_when_no_meta(self):
        """Handles missing competition_meta."""
        checkpoint = create_initial_checkpoint(
            competition_slug="test",
            problem_type="tabular-regression",
            baseline_score=0.50,
        )

        assert checkpoint.competition_meta == {}


class TestUpdateCheckpointAfterExperiment:
    """Tests for post-experiment updates."""

    def test_increments_total_experiments(self, sample_checkpoint):
        """Increments experiment count."""
        original_count = sample_checkpoint.total_experiments

        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=0.86, improved=False, idea_index=5
        )

        assert updated.total_experiments == original_count + 1

    def test_updates_pointer_on_any_result(self, sample_checkpoint):
        """Updates ideas_pointer regardless of outcome."""
        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=0.80, improved=False, idea_index=7
        )

        assert updated.ideas_pointer == 8  # idea_index + 1

    def test_updates_best_score_on_improvement(self, sample_checkpoint):
        """Updates best_score when improved."""
        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=0.90, improved=True, idea_index=5
        )

        assert updated.best_score == 0.90
        assert updated.successful_experiments == sample_checkpoint.successful_experiments

    def test_increments_successful_on_improvement(self, sample_checkpoint):
        """Increments successful_experiments on improvement."""
        original_successful = sample_checkpoint.successful_experiments

        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=0.90, improved=True, idea_index=5
        )

        assert updated.successful_experiments == original_successful + 1

    def test_adds_to_plateau_window(self, sample_checkpoint):
        """Adds score to plateau_window_scores."""
        original_len = len(sample_checkpoint.plateau_window_scores)

        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=0.86, improved=False, idea_index=5
        )

        assert len(updated.plateau_window_scores) == original_len + 1
        assert updated.plateau_window_scores[-1] == 0.86

    def test_none_score_not_added_to_window(self, sample_checkpoint):
        """Crashed experiments (None score) not added to window."""
        original_len = len(sample_checkpoint.plateau_window_scores)

        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=None, improved=False, idea_index=5
        )

        assert len(updated.plateau_window_scores) == original_len

    def test_plateau_window_trimmed_to_20(self, sample_checkpoint):
        """Plateau window doesn't grow beyond 20 entries."""
        sample_checkpoint.plateau_window_scores = list(range(20))

        updated = update_checkpoint_after_experiment(
            sample_checkpoint, score=99.0, improved=False, idea_index=5
        )

        assert len(updated.plateau_window_scores) <= 20
        assert updated.plateau_window_scores[-1] == 99.0


class TestUpdateCheckpointForReresearch:
    """Tests for re-research state updates."""

    def test_increments_reresearch_attempts(self, sample_checkpoint):
        """Increments reresearch_attempts counter."""
        original = sample_checkpoint.reresearch_attempts

        updated = update_checkpoint_for_reresearch(sample_checkpoint)

        assert updated.reresearch_attempts == original + 1

    def test_keeps_research_phase(self, sample_checkpoint):
        """Keeps phase as research."""
        sample_checkpoint.phase = "research"

        updated = update_checkpoint_for_reresearch(sample_checkpoint)

        assert updated.phase == "research"


class TestUpdateCheckpointForBranch:
    """Tests for branch/pivot state updates."""

    def test_archives_old_branch(self, sample_checkpoint):
        """Archives current branch before switching."""
        sample_checkpoint.current_branch = "old-branch"
        sample_checkpoint.best_score = 0.85

        updated = update_checkpoint_for_branch(
            sample_checkpoint, "new-branch", "New Strategy"
        )

        assert "old-branch" in updated.branches
        assert updated.branches["old-branch"]["best_score"] == 0.85
        assert updated.branches["old-branch"]["status"] == "archived"

    def test_sets_new_branch(self, sample_checkpoint):
        """Sets new branch as current."""
        updated = update_checkpoint_for_branch(
            sample_checkpoint, "new-branch", "New Strategy"
        )

        assert updated.current_branch == "new-branch"
        assert updated.strategy_name == "New Strategy"

    def test_resets_ideas_pointer(self, sample_checkpoint):
        """Resets ideas_pointer for new branch."""
        sample_checkpoint.ideas_pointer = 10

        updated = update_checkpoint_for_branch(
            sample_checkpoint, "new-branch", "New Strategy"
        )

        assert updated.ideas_pointer == 0

    def test_clears_plateau_window(self, sample_checkpoint):
        """Clears plateau_window_scores for new branch."""
        sample_checkpoint.plateau_window_scores = [0.8, 0.81, 0.82]

        updated = update_checkpoint_for_branch(
            sample_checkpoint, "new-branch", "New Strategy"
        )

        assert updated.plateau_window_scores == []


class TestUpdateCheckpointAfterBranchComparison:
    """Tests for branch comparison completion."""

    def test_archives_loser_branch(self, sample_checkpoint):
        """Archives losing branch."""
        sample_checkpoint.branches = {
            "branch-a": {"best_score": 0.80, "status": "active"},
            "branch-b": {"best_score": 0.85, "status": "active"},
        }

        updated = update_checkpoint_after_branch_comparison(
            sample_checkpoint, winner_branch="branch-b", loser_branch="branch-a"
        )

        assert updated.branches["branch-a"]["status"] == "archived"

    def test_sets_winner_as_current(self, sample_checkpoint):
        """Sets winner as current branch."""
        updated = update_checkpoint_after_branch_comparison(
            sample_checkpoint, winner_branch="winner", loser_branch="loser"
        )

        assert updated.current_branch == "winner"

    def test_resets_reresearch_attempts(self, sample_checkpoint):
        """Resets reresearch_attempts after comparison."""
        sample_checkpoint.reresearch_attempts = 2

        updated = update_checkpoint_after_branch_comparison(
            sample_checkpoint, winner_branch="winner", loser_branch="loser"
        )

        assert updated.reresearch_attempts == 0


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_first_run_generates_run_001(self, tmp_path):
        """First run generates run-001."""
        run_id = generate_run_id(tmp_path)
        assert run_id == "run-001"

    def test_increments_based_on_manifest(self, tmp_path):
        """Increments based on archive manifest."""
        manifest = [
            {"run_id": "run-001"},
            {"run_id": "run-002"},
        ]
        manifest_path = tmp_path / "archive_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        run_id = generate_run_id(tmp_path)
        assert run_id == "run-003"

    def test_handles_corrupted_manifest(self, tmp_path):
        """Handles corrupted manifest gracefully."""
        manifest_path = tmp_path / "archive_manifest.json"
        manifest_path.write_text("{ invalid }")

        run_id = generate_run_id(tmp_path)
        assert run_id == "run-001"
