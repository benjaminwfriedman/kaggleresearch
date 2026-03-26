"""Tests for idea tree operations."""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.idea_tree import IdeaTree, IdeaNode


class TestIdeaNode:
    """Tests for IdeaNode dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        node = IdeaNode(
            id="abc123",
            idea_title="Test Idea",
            parent_commit="commit_xyz",
            parent_node_id=None,
            status="improved",
            score=0.85,
            depth=0,
        )

        data = node.to_dict()

        assert data["id"] == "abc123"
        assert data["idea_title"] == "Test Idea"
        assert data["score"] == 0.85

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "id": "def456",
            "idea_title": "Another Idea",
            "parent_commit": "commit_abc",
            "parent_node_id": "parent123",
            "children": ["child1", "child2"],
            "siblings": [],
            "status": "pending",
            "score": None,
            "depth": 1,
            "branch_name": "main",
            "timestamp": "2024-01-01T00:00:00",
            "error_message": None,
            "hypothesis": "Test hypothesis",
            "dimension_answered": "feature_engineering",
            "value_chosen": "polynomial",
            "fixed_context": {"model": "lightgbm"},
            "open_dimensions": ["regularization"],
            "visits": 5,
            "total_reward": 0.3,
            "relationship": "child",
        }

        node = IdeaNode.from_dict(data)

        assert node.id == "def456"
        assert node.parent_node_id == "parent123"
        assert node.children == ["child1", "child2"]
        assert node.visits == 5

    def test_from_dict_backward_compatibility(self):
        """Test deserialization with missing fields (old format)."""
        # Old format without new fields
        data = {
            "id": "old123",
            "idea_title": "Old Idea",
            "parent_commit": "commit_old",
            "parent_node_id": None,
            "children": [],
            "status": "improved",
            "score": 0.80,
            "depth": 0,
            "branch_name": "",
            "timestamp": "",
            "error_message": None,
            # Missing: siblings, hypothesis, dimension_answered, etc.
        }

        node = IdeaNode.from_dict(data)

        # Should use defaults
        assert node.siblings == []
        assert node.hypothesis == ""
        assert node.visits == 0
        assert node.relationship == "child"


class TestIdeaTreeBasicOperations:
    """Tests for basic tree operations."""

    def test_add_root_node(self):
        """Adding first node sets it as root."""
        tree = IdeaTree()
        root = tree.add_node("Baseline", "commit_abc", None)

        assert tree.root_id == root.id
        assert root.depth == 0
        assert root.parent_node_id is None

    def test_add_child_node(self):
        """Adding child node updates parent's children list."""
        tree = IdeaTree()
        root = tree.add_node("Baseline", "commit_abc", None)
        child = tree.add_node("Feature Eng", "commit_def", root.id)

        assert child.parent_node_id == root.id
        assert child.depth == 1
        assert child.id in root.children

    def test_add_grandchild_node(self):
        """Grandchild has correct depth."""
        tree = IdeaTree()
        root = tree.add_node("Baseline", "commit_abc", None)
        child = tree.add_node("Feature Eng", "commit_def", root.id)
        grandchild = tree.add_node("Polynomial Features", "commit_ghi", child.id)

        assert grandchild.depth == 2
        assert grandchild.id in child.children

    def test_get_node(self, sample_tree):
        """Get node by ID."""
        root = sample_tree.get_node(sample_tree.root_id)
        assert root is not None
        assert root.idea_title == "Baseline"

    def test_get_node_nonexistent(self, sample_tree):
        """Get nonexistent node returns None."""
        assert sample_tree.get_node("nonexistent") is None

    def test_update_status(self, sample_tree):
        """Update node status."""
        root_id = sample_tree.root_id
        sample_tree.update_status(root_id, "plateau", score=0.82)

        node = sample_tree.get_node(root_id)
        assert node.status == "plateau"
        assert node.score == 0.82

    def test_update_status_with_error(self, sample_tree):
        """Update status with error message."""
        root_id = sample_tree.root_id
        sample_tree.update_status(root_id, "crashed", error_message="OOM error")

        node = sample_tree.get_node(root_id)
        assert node.status == "crashed"
        assert node.error_message == "OOM error"


class TestIdeaTreeNavigation:
    """Tests for tree navigation."""

    def test_get_current_node(self, sample_tree):
        """Get current node pointer."""
        # Current node should be last added
        current = sample_tree.get_current_node()
        assert current is not None
        assert current.idea_title == "Add Interaction Features"

    def test_set_current_node(self, sample_tree):
        """Set current node pointer."""
        sample_tree.set_current_node(sample_tree.root_id)
        current = sample_tree.get_current_node()
        assert current.id == sample_tree.root_id

    def test_get_parent(self, sample_tree):
        """Get parent node."""
        # Find a child node
        root = sample_tree.get_node(sample_tree.root_id)
        child_id = root.children[0]
        child = sample_tree.get_node(child_id)

        parent = sample_tree.get_parent(child_id)
        assert parent.id == sample_tree.root_id

    def test_get_parent_of_root(self, sample_tree):
        """Root node has no parent."""
        parent = sample_tree.get_parent(sample_tree.root_id)
        assert parent is None

    def test_get_siblings(self, sample_tree):
        """Get sibling nodes."""
        root = sample_tree.get_node(sample_tree.root_id)
        # Root has two children which are siblings
        child1_id = root.children[0]

        siblings = sample_tree.get_siblings(child1_id)

        # Should have 1 sibling (the other child)
        assert len(siblings) == 1
        assert siblings[0].id in root.children
        assert siblings[0].id != child1_id

    def test_get_children(self, sample_tree):
        """Get child nodes."""
        children = sample_tree.get_children(sample_tree.root_id)
        assert len(children) == 2

    def test_get_pending_children(self, sample_tree):
        """Get only pending children."""
        root = sample_tree.get_node(sample_tree.root_id)
        child1 = sample_tree.get_node(root.children[0])

        pending = sample_tree.get_pending_children(child1.id)

        assert len(pending) == 1
        assert pending[0].status == "pending"


class TestIdeaTreeBestNode:
    """Tests for finding best nodes."""

    def test_get_best_node_higher_better(self, sample_tree):
        """Find best node when higher is better."""
        best = sample_tree.get_best_node(metric_direction="higher_better")

        # Feature Engineering has score 0.82, highest in tree
        assert best.score == 0.82
        assert best.idea_title == "Feature Engineering"

    def test_get_best_node_lower_better(self, sample_tree):
        """Find best node when lower is better."""
        best = sample_tree.get_best_node(metric_direction="lower_better")

        # Hyperparameter Tuning has score 0.79, lowest
        assert best.score == 0.79
        assert best.idea_title == "Hyperparameter Tuning"

    def test_get_best_child(self, sample_tree):
        """Find best child of a node."""
        best_child = sample_tree.get_best_child(
            sample_tree.root_id, metric_direction="higher_better"
        )

        assert best_child.score == 0.82

    def test_get_best_sibling(self, sample_tree):
        """Find best sibling (including self)."""
        root = sample_tree.get_node(sample_tree.root_id)
        child1_id = root.children[0]

        best_sib = sample_tree.get_best_sibling(
            child1_id, metric_direction="higher_better"
        )

        # Only child1 is "improved", so it should be best
        assert best_sib is not None
        assert best_sib.status == "improved"


class TestIdeaTreeBacktracking:
    """Tests for backtracking logic."""

    def test_get_backtrack_target_with_pending_siblings(self, sample_tree):
        """Backtrack to parent when siblings are pending."""
        # Create a node with pending siblings
        root = sample_tree.get_node(sample_tree.root_id)
        child1_id = root.children[0]
        child1 = sample_tree.get_node(child1_id)

        # Add a pending sibling to grandchild
        grandchild = sample_tree.get_node(child1.children[0])
        sibling = sample_tree.add_node(
            "Sibling Idea", "commit_xxx", child1.id
        )
        sibling.status = "pending"

        target = sample_tree.get_backtrack_target(grandchild.id)

        # Should backtrack to child1 (parent) to try the pending sibling
        assert target is not None
        assert target.id == child1.id

    def test_get_backtrack_target_tree_exhausted(self):
        """Returns None when tree is exhausted."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.status = "improved"
        child = tree.add_node("Child", "commit_def", root.id)
        child.status = "no_improvement"

        target = tree.get_backtrack_target(child.id)

        # No pending nodes anywhere
        assert target is None

    def test_count_crashed_children(self, sample_tree):
        """Count crashed children."""
        root_id = sample_tree.root_id

        # Initially no crashes
        assert sample_tree.count_crashed_children(root_id) == 0

        # Add a crashed child
        crashed = sample_tree.add_node("Crashed Idea", "commit_xxx", root_id)
        crashed.status = "crashed"

        assert sample_tree.count_crashed_children(root_id) == 1

    def test_count_consecutive_crashes(self):
        """Count consecutive crashes from most recent."""
        # Create fresh tree to control timestamps
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_root", None)
        root.status = "improved"

        # Add children with increasing timestamps
        child1 = tree.add_node("Success", "commit_1", root.id)
        child1.status = "improved"
        child1.timestamp = "2024-01-01T00:00:00"

        child2 = tree.add_node("Crash 1", "commit_2", root.id)
        child2.status = "crashed"
        child2.timestamp = "2024-01-01T00:01:00"

        child3 = tree.add_node("Crash 2", "commit_3", root.id)
        child3.status = "crashed"
        child3.timestamp = "2024-01-01T00:02:00"

        child4 = tree.add_node("Crash 3", "commit_4", root.id)
        child4.status = "crashed"
        child4.timestamp = "2024-01-01T00:03:00"

        # Last 3 children (most recent) are crashes
        count = tree.count_consecutive_crashes(root.id)
        assert count == 3


class TestIdeaTreeUCB1:
    """Tests for UCB1 selection algorithm."""

    def test_ucb1_score_unvisited_is_inf(self):
        """Unvisited nodes have infinite UCB1 score."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.visits = 0

        score = tree.ucb1_score(root.id)
        assert score == float("inf")

    def test_ucb1_score_with_visits(self):
        """UCB1 score decreases with more visits."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.visits = 10
        root.total_reward = 5.0

        child1 = tree.add_node("Child1", "commit_def", root.id)
        child1.visits = 5
        child1.total_reward = 3.0

        child2 = tree.add_node("Child2", "commit_ghi", root.id)
        child2.visits = 2
        child2.total_reward = 1.5

        score1 = tree.ucb1_score(child1.id)
        score2 = tree.ucb1_score(child2.id)

        # Child2 with fewer visits should have higher exploration bonus
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    def test_select_node_ucb1(self):
        """Select node using UCB1."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.status = "improved"
        root.score = 0.80
        root.visits = 10
        root.total_reward = 0.5

        child1 = tree.add_node("Child1", "commit_def", root.id)
        child1.status = "improved"
        child1.score = 0.82
        child1.visits = 5
        child1.total_reward = 0.6

        child2 = tree.add_node("Child2", "commit_ghi", root.id)
        child2.status = "no_improvement"
        child2.score = 0.78
        child2.visits = 2
        child2.total_reward = 0.1

        selected = tree.select_node_ucb1()
        assert selected is not None

    def test_update_node_reward(self):
        """Update node reward propagates to ancestors."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.visits = 0
        root.total_reward = 0.0

        child = tree.add_node("Child", "commit_def", root.id)
        child.visits = 0
        child.total_reward = 0.0

        # Update reward for child
        tree.update_node_reward(
            child.id, score=0.85, baseline_score=0.80, metric_direction="higher_better"
        )

        # Both child and root should have updated visits/rewards
        assert child.visits == 1
        assert root.visits == 1
        assert child.total_reward > 0
        assert root.total_reward > 0


class TestIdeaTreeDimensionTracking:
    """Tests for dimension-based exploration."""

    def test_add_child_node_with_dimensions(self):
        """Add child node with dimension tracking."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.open_dimensions = ["feature_engineering", "regularization", "model"]

        child = tree.add_child_node(
            parent_node_id=root.id,
            idea_title="Polynomial Features",
            parent_commit="commit_def",
            hypothesis="Polynomial features may help",
            dimension_answered="feature_engineering",
            value_chosen="polynomial",
        )

        # Child should inherit context and have one less open dimension
        assert child.dimension_answered == "feature_engineering"
        assert child.value_chosen == "polynomial"
        assert "feature_engineering" in child.fixed_context
        assert "feature_engineering" not in child.open_dimensions
        assert "regularization" in child.open_dimensions

    def test_add_sibling_node_with_dimensions(self):
        """Add sibling node with same dimension but different value."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.open_dimensions = ["feature_engineering"]

        child1 = tree.add_child_node(
            parent_node_id=root.id,
            idea_title="Polynomial Features",
            parent_commit="commit_def",
            dimension_answered="feature_engineering",
            value_chosen="polynomial",
        )

        sibling = tree.add_sibling_node(
            sibling_node_id=child1.id,
            idea_title="Log Transform",
            parent_commit="commit_def",
            value_chosen="log_transform",
        )

        # Sibling should answer same dimension with different value
        assert sibling.dimension_answered == "feature_engineering"
        assert sibling.value_chosen == "log_transform"
        assert child1.id in sibling.siblings
        assert sibling.id in child1.siblings

    def test_update_open_dimensions(self):
        """Update open dimensions after reflection."""
        tree = IdeaTree()
        root = tree.add_node("Root", "commit_abc", None)
        root.open_dimensions = ["feature_engineering"]

        # Discover new dimensions through experimentation
        tree.update_open_dimensions(
            root.id, ["regularization", "learning_rate"]
        )

        assert "feature_engineering" in root.open_dimensions
        assert "regularization" in root.open_dimensions
        assert "learning_rate" in root.open_dimensions


class TestIdeaTreePersistence:
    """Tests for tree save/load."""

    def test_save_and_load(self, tmp_path, sample_tree):
        """Test save/load round trip."""
        tree_path = tmp_path / "idea_tree.json"
        sample_tree.tree_path = tree_path
        sample_tree.competition_slug = "test-comp"
        sample_tree.run_id = "run-001"

        sample_tree.save()

        # Load into new tree
        new_tree = IdeaTree(tree_path)
        loaded = new_tree.load()

        assert loaded is True
        assert new_tree.root_id == sample_tree.root_id
        assert new_tree.competition_slug == "test-comp"
        assert len(new_tree.nodes) == len(sample_tree.nodes)

    def test_load_nonexistent(self, tmp_path):
        """Load nonexistent file returns False."""
        tree = IdeaTree(tmp_path / "nonexistent.json")
        assert tree.load() is False

    def test_load_corrupted(self, tmp_path):
        """Load corrupted file returns False."""
        tree_path = tmp_path / "corrupted.json"
        tree_path.write_text("{ invalid json }")

        tree = IdeaTree(tree_path)
        assert tree.load() is False


class TestIdeaTreeUtilities:
    """Tests for utility functions."""

    def test_slugify(self, sample_tree):
        """Test title slugification."""
        slug = sample_tree._slugify("Add Feature Engineering!")
        assert slug == "add-feature-engineering"

    def test_slugify_truncates(self, sample_tree):
        """Slugify truncates long titles."""
        long_title = "A" * 100
        slug = sample_tree._slugify(long_title)
        assert len(slug) <= 30

    def test_generate_branch_name(self, sample_tree):
        """Generate branch name from node path."""
        sample_tree.competition_slug = "titanic"
        sample_tree.run_id = "run-001"

        root = sample_tree.get_node(sample_tree.root_id)
        child_id = root.children[0]

        branch = sample_tree.generate_branch_name(
            child_id, "titanic", "run-001"
        )

        assert "titanic" in branch
        assert "run-001" in branch

    def test_get_max_depth(self, sample_tree):
        """Get maximum tree depth."""
        depth = sample_tree.get_max_depth()
        # Root (0) -> Child (1) -> Grandchild (2)
        assert depth == 2

    def test_count_by_status(self, sample_tree):
        """Count nodes by status."""
        counts = sample_tree.count_by_status()

        assert counts["improved"] == 2  # Baseline and Feature Engineering
        assert counts["no_improvement"] == 1  # Hyperparameter Tuning
        assert counts["pending"] == 1  # Add Interaction Features

    def test_get_improved_path(self, sample_tree):
        """Get path of improved nodes to best."""
        path = sample_tree.get_improved_path(metric_direction="higher_better")

        # Should be: Baseline (0.80) -> Feature Engineering (0.82)
        assert len(path) >= 1
        assert all(n.status == "improved" or n.id == sample_tree.root_id for n in path)

    def test_all_siblings_tested(self, sample_tree):
        """Check if all siblings have been tested."""
        root = sample_tree.get_node(sample_tree.root_id)
        child1_id = root.children[0]

        # Both children have been tested (not pending)
        assert sample_tree.all_siblings_tested(child1_id) is True

        # Add a pending sibling
        pending = sample_tree.add_node("Pending Idea", "commit_xxx", root.id)
        pending.status = "pending"

        assert sample_tree.all_siblings_tested(child1_id) is False

    def test_render_tree(self, sample_tree):
        """Render tree as ASCII art."""
        output = sample_tree.render_tree()

        assert "Baseline" in output
        assert "Feature Engineering" in output
        assert "✓" in output  # Improved marker

    def test_get_expansion_context(self, sample_tree):
        """Get context for node expansion."""
        root = sample_tree.get_node(sample_tree.root_id)
        child_id = root.children[0]

        context = sample_tree.get_expansion_context(child_id)

        assert "node" in context
        assert "siblings" in context
        assert "parent" in context
        assert context["parent"]["title"] == "Baseline"
