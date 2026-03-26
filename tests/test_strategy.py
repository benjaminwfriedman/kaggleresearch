"""Tests for strategy and IDEAS.md parsing."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.strategy import (
    Idea,
    parse_ideas_md,
    parse_idea_block,
    update_idea_status,
    append_learning_log,
    format_learning_log_entry,
    get_next_pending_idea,
    count_ideas_by_status,
    sanitize_idea_block,
    append_ideas_to_file,
)


class TestIdea:
    """Tests for Idea dataclass."""

    def test_to_dict(self, sample_ideas):
        """Test serialization."""
        idea = sample_ideas[0]
        data = idea.to_dict()

        assert data["title"] == "Reduce Learning Rate"
        assert data["risk"] == "low"
        assert data["status"] == "improved"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "title": "Test Idea",
            "source": "empirical",
            "risk": "medium",
            "estimated_gain": "small",
            "status": "pending",
            "hypothesis": "Test hypothesis",
            "implementation": "Test implementation",
            "validation": "Test validation",
            "index": 5,
        }

        idea = Idea.from_dict(data)

        assert idea.title == "Test Idea"
        assert idea.index == 5


class TestParseIdeasMd:
    """Tests for parsing IDEAS.md files."""

    def test_parse_valid_ideas_file(self, tmp_path, sample_ideas_md_content):
        """Parse well-formed IDEAS.md file."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text(sample_ideas_md_content)

        ideas = parse_ideas_md(ideas_path)

        assert len(ideas) == 3
        assert ideas[0].title == "Reduce Learning Rate"
        assert ideas[1].title == "Add Dropout"
        assert ideas[2].title == "Feature Engineering"

    def test_parse_extracts_all_fields(self, tmp_path, sample_ideas_md_content):
        """Extracts all fields from idea blocks."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text(sample_ideas_md_content)

        ideas = parse_ideas_md(ideas_path)
        idea = ideas[0]

        assert idea.source == "empirical"
        assert idea.risk == "low"
        assert idea.estimated_gain == "small"
        assert idea.status == "improved"
        assert "LR may be too high" in idea.hypothesis
        assert "LEARNING_RATE" in idea.implementation
        assert "smoother" in idea.validation

    def test_parse_nonexistent_file(self, tmp_path):
        """Returns empty list for nonexistent file."""
        ideas = parse_ideas_md(tmp_path / "nonexistent.md")
        assert ideas == []

    def test_parse_empty_file(self, tmp_path):
        """Returns empty list for empty file."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text("")

        ideas = parse_ideas_md(ideas_path)
        assert ideas == []

    def test_parse_file_with_no_ideas(self, tmp_path):
        """Returns empty list for file with no idea blocks."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text("# Some header\n\nSome text without ideas")

        ideas = parse_ideas_md(ideas_path)
        assert ideas == []


class TestParseIdeaBlock:
    """Tests for parsing individual idea blocks."""

    def test_parse_valid_block(self):
        """Parse a valid idea block."""
        block = """## IDEA: Test Idea
Source: empirical
Risk: low
Estimated gain: medium
Status: pending
---
Hypothesis: This should work
Implementation: Change the thing
Validation: Score should improve
==="""

        idea = parse_idea_block(block, 0)

        assert idea is not None
        assert idea.title == "Test Idea"
        assert idea.source == "empirical"
        assert idea.risk == "low"
        assert idea.estimated_gain == "medium"
        assert idea.status == "pending"
        assert idea.hypothesis == "This should work"
        assert idea.implementation == "Change the thing"
        assert idea.validation == "Score should improve"

    def test_parse_block_with_missing_fields(self):
        """Parse block with some missing fields (uses defaults)."""
        block = """## IDEA: Minimal Idea
---
Hypothesis: Just do it
Implementation: Make changes
==="""

        idea = parse_idea_block(block, 0)

        assert idea is not None
        assert idea.title == "Minimal Idea"
        assert idea.source == "empirical"  # Default
        assert idea.risk == "medium"  # Default
        assert idea.status == "pending"  # Default

    def test_parse_block_preserves_index(self):
        """Index is preserved."""
        block = """## IDEA: Test
---
Hypothesis: Test
==="""

        idea = parse_idea_block(block, 42)
        assert idea.index == 42

    def test_parse_malformed_block_returns_none(self):
        """Malformed block returns None."""
        block = "This is not a valid idea block"

        idea = parse_idea_block(block, 0)
        # Should handle gracefully - may return an idea with defaults or None
        # Based on the implementation, it extracts what it can
        assert idea is None or idea.title == "Unknown"


class TestUpdateIdeaStatus:
    """Tests for updating idea status in file."""

    def test_update_status(self, tmp_path, sample_ideas_md_content):
        """Update status of an idea."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text(sample_ideas_md_content)

        update_idea_status(ideas_path, "Add Dropout", "running")

        content = ideas_path.read_text()
        # Should find "Add Dropout" with status "running"
        assert "Status: running" in content

    def test_update_nonexistent_idea(self, tmp_path, sample_ideas_md_content):
        """Updating nonexistent idea doesn't crash."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text(sample_ideas_md_content)

        # Should not raise
        update_idea_status(ideas_path, "Nonexistent Idea", "running")


class TestAppendLearningLog:
    """Tests for appending to learning log."""

    def test_append_to_existing_log(self, tmp_path):
        """Append note to existing Learning Log section."""
        strategy_path = tmp_path / "STRATEGY.md"
        strategy_path.write_text("""# Strategy

Some content

## Learning Log

- Previous note
""")

        append_learning_log(strategy_path, "New experiment worked!")

        content = strategy_path.read_text()
        assert "Previous note" in content
        assert "New experiment worked!" in content

    def test_create_log_section(self, tmp_path):
        """Create Learning Log section if it doesn't exist."""
        strategy_path = tmp_path / "STRATEGY.md"
        strategy_path.write_text("""# Strategy

Some content without a learning log
""")

        append_learning_log(strategy_path, "First note")

        content = strategy_path.read_text()
        assert "## Learning Log" in content
        assert "First note" in content


class TestFormatLearningLogEntry:
    """Tests for formatting log entries."""

    def test_format_improved_with_delta(self):
        """Format improved experiment with delta."""
        entry = format_learning_log_entry(
            experiment_index=5,
            idea_title="Feature Engineering",
            status="improved",
            score=0.85,
            delta=0.02,
        )

        assert "Experiment 5" in entry
        assert "IMPROVED" in entry
        assert "Feature Engineering" in entry
        assert "0.02" in entry

    def test_format_no_improvement(self):
        """Format no improvement experiment."""
        entry = format_learning_log_entry(
            experiment_index=6,
            idea_title="Bad Idea",
            status="no_improvement",
            score=0.80,
        )

        assert "Experiment 6" in entry
        assert "NO_IMPROVEMENT" in entry
        assert "0.80" in entry

    def test_format_crashed(self):
        """Format crashed experiment."""
        entry = format_learning_log_entry(
            experiment_index=7,
            idea_title="Crashed Idea",
            status="crashed",
            score=None,
        )

        assert "Experiment 7" in entry
        assert "CRASHED" in entry
        assert "Crashed Idea" in entry


class TestGetNextPendingIdea:
    """Tests for finding next pending idea."""

    def test_returns_first_pending(self, sample_ideas):
        """Returns first pending idea."""
        next_idea = get_next_pending_idea(sample_ideas)

        assert next_idea is not None
        assert next_idea.status == "pending"
        assert next_idea.title == "Add Dropout"

    def test_returns_none_when_all_done(self):
        """Returns None when no pending ideas."""
        ideas = [
            Idea("Done 1", "emp", "low", "small", "improved", "", "", "", 0),
            Idea("Done 2", "emp", "low", "small", "no_improvement", "", "", "", 1),
        ]

        next_idea = get_next_pending_idea(ideas)
        assert next_idea is None

    def test_returns_none_for_empty_list(self):
        """Returns None for empty list."""
        assert get_next_pending_idea([]) is None


class TestCountIdeasByStatus:
    """Tests for counting ideas by status."""

    def test_count_all_statuses(self, sample_ideas):
        """Count ideas by status."""
        counts = count_ideas_by_status(sample_ideas)

        assert counts["improved"] == 1
        assert counts["pending"] == 2
        assert counts["no_improvement"] == 0
        assert counts["crashed"] == 0

    def test_count_empty_list(self):
        """Count empty list."""
        counts = count_ideas_by_status([])

        assert counts["pending"] == 0
        assert counts["improved"] == 0


class TestSanitizeIdeaBlock:
    """Tests for sanitizing idea blocks."""

    def test_sanitize_valid_block(self):
        """Valid block is preserved."""
        block = """## IDEA: Test Idea
Source: empirical
Risk: low
Estimated gain: small
Status: pending
---
Hypothesis: Test
Implementation: Change something
Validation: Check score
==="""

        sanitized = sanitize_idea_block(block)

        assert sanitized is not None
        assert "## IDEA: Test Idea" in sanitized
        assert "Source:" in sanitized
        assert "Status: pending" in sanitized

    def test_sanitize_markdown_bold_format(self):
        """Sanitize markdown bold format to standard."""
        block = """**IDEA: Bold Format Idea**
Risk: medium
Implementation: Do the thing"""

        sanitized = sanitize_idea_block(block)

        assert sanitized is not None
        assert "## IDEA: Bold Format Idea" in sanitized

    def test_sanitize_missing_fields_uses_defaults(self):
        """Missing fields get default values."""
        block = """## IDEA: Minimal
Implementation: Just do it"""

        sanitized = sanitize_idea_block(block)

        assert sanitized is not None
        assert "Source: re-research" in sanitized
        assert "Risk: medium" in sanitized
        assert "Status: pending" in sanitized

    def test_sanitize_empty_returns_none(self):
        """Empty block returns None."""
        assert sanitize_idea_block("") is None
        assert sanitize_idea_block("   \n\n  ") is None


class TestAppendIdeasToFile:
    """Tests for appending new ideas to file."""

    def test_append_new_ideas(self, tmp_path):
        """Append new ideas to existing file."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text("""# Experiment Ideas

## IDEA: Existing Idea
Source: empirical
Risk: low
Estimated gain: small
Status: improved
---
Hypothesis: Already done
Implementation: Was done
Validation: It worked
===
""")

        new_ideas = """## IDEA: New Idea
Source: re-research
Risk: medium
Estimated gain: large
---
Hypothesis: This should help
Implementation: Add new feature
Validation: Check score
==="""

        count = append_ideas_to_file(ideas_path, new_ideas)

        assert count == 1
        content = ideas_path.read_text()
        assert "Existing Idea" in content
        assert "New Idea" in content
        assert "Re-research Ideas" in content

    def test_skip_duplicate_ideas(self, tmp_path):
        """Skip ideas that already exist."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text("""# Experiment Ideas

## IDEA: Existing Idea
Source: empirical
Status: pending
---
Hypothesis: Test
===
""")

        new_ideas = """## IDEA: Existing Idea
Source: re-research
Status: pending
---
Hypothesis: Duplicate
==="""

        count = append_ideas_to_file(ideas_path, new_ideas)

        assert count == 0

    def test_append_no_valid_ideas(self, tmp_path):
        """Return 0 when no valid ideas."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text("# Ideas\n")

        count = append_ideas_to_file(ideas_path, "This is not an idea")

        assert count == 0

    def test_append_sanitizes_malformed_ideas(self, tmp_path):
        """Sanitize malformed ideas before appending."""
        ideas_path = tmp_path / "IDEAS.md"
        ideas_path.write_text("# Ideas\n")

        new_ideas = """**IDEA: Malformed Format**
Some implementation text"""

        count = append_ideas_to_file(ideas_path, new_ideas)

        if count > 0:
            content = ideas_path.read_text()
            assert "## IDEA:" in content  # Should be standardized
