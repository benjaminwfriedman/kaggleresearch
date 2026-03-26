"""Shared test fixtures for KaggleResearch tests."""

import pytest
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.plateau import ExperimentResult
from utils.checkpoint import CheckpointState
from utils.idea_tree import IdeaTree
from utils.strategy import Idea


@pytest.fixture
def sample_experiments():
    """Sample experiment results for testing."""
    return [
        ExperimentResult("Add feature X", 0.75, "improved", 0),
        ExperimentResult("Reduce learning rate", 0.76, "improved", 1),
        ExperimentResult("Add dropout", 0.755, "no_improvement", 2),
        ExperimentResult("Try larger batch", None, "crashed", 3),
        ExperimentResult("Feature engineering", 0.77, "improved", 4),
    ]


@pytest.fixture
def sample_checkpoint():
    """Sample checkpoint state for testing."""
    return CheckpointState(
        phase="research",
        current_branch="main",
        best_score=0.85,
        baseline_score=0.80,
        ideas_pointer=3,
        plateau_window_scores=[0.82, 0.83, 0.84, 0.85],
        reresearch_attempts=0,
        total_experiments=5,
        successful_experiments=3,
        branches={},
        strategy_name="Gradient Boosting Strategy",
        problem_type="tabular-classification",
        competition_slug="titanic",
        run_id="run-001",
        current_node_id="abc123",
        exploration_mode="df",
        competition_meta={
            "name": "Titanic",
            "metric": "accuracy",
            "metric_direction": "higher_better",
        },
    )


@pytest.fixture
def sample_tree():
    """Sample idea tree for testing."""
    tree = IdeaTree()

    # Add root node
    root = tree.add_node("Baseline", "commit_abc123", None)
    root.status = "improved"
    root.score = 0.80
    tree.root_id = root.id

    # Add child nodes
    child1 = tree.add_node("Feature Engineering", "commit_def456", root.id)
    child1.status = "improved"
    child1.score = 0.82

    child2 = tree.add_node("Hyperparameter Tuning", "commit_ghi789", root.id)
    child2.status = "no_improvement"
    child2.score = 0.79

    # Add grandchild
    grandchild = tree.add_node("Add Interaction Features", "commit_jkl012", child1.id)
    grandchild.status = "pending"

    return tree


@pytest.fixture
def sample_ideas():
    """Sample ideas for testing."""
    return [
        Idea(
            title="Reduce Learning Rate",
            source="empirical",
            risk="low",
            estimated_gain="small",
            status="improved",
            hypothesis="Current LR may be too high",
            implementation="Change LR from 0.1 to 0.01",
            validation="Loss should decrease",
            index=0,
        ),
        Idea(
            title="Add Dropout",
            source="arxiv:1234.5678",
            risk="medium",
            estimated_gain="medium",
            status="pending",
            hypothesis="Regularization may help",
            implementation="Add dropout layers",
            validation="CV score should improve",
            index=1,
        ),
        Idea(
            title="Feature Engineering",
            source="derived-from-strategy",
            risk="low",
            estimated_gain="large",
            status="pending",
            hypothesis="New features may capture patterns",
            implementation="Add polynomial features",
            validation="Feature importance analysis",
            index=2,
        ),
    ]


@pytest.fixture
def sample_ideas_md_content():
    """Sample IDEAS.md file content."""
    return """# Experiment Ideas

## IDEA: Reduce Learning Rate
Source: empirical
Risk: low
Estimated gain: small
Status: improved
---
Hypothesis: Current LR may be too high causing instability
Implementation: Change LEARNING_RATE from 0.1 to 0.01 in train.py
Validation: Loss curves should be smoother, CV score should improve
===

## IDEA: Add Dropout
Source: arxiv:1234.5678
Risk: medium
Estimated gain: medium
Status: pending
---
Hypothesis: Model may be overfitting, regularization could help
Implementation: Add nn.Dropout(0.3) after each dense layer
Validation: Training/validation loss gap should decrease
===

## IDEA: Feature Engineering
Source: derived-from-strategy
Risk: low
Estimated gain: large
Status: pending
---
Hypothesis: Polynomial features may capture non-linear relationships
Implementation: Use sklearn PolynomialFeatures with degree=2
Validation: Check feature importance, CV score improvement
===
"""
