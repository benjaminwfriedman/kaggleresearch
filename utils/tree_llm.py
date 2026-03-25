"""
LLM-driven tree operations for explore/exploit loop.

This module handles:
- F2: Child/Sibling classification
- F3: I-MCTS expansion (generating experiments with context)
- F5: Backtracking depth analysis
- F6: Open dimension discovery
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Result of child/sibling classification."""
    relationship: str  # "child" or "sibling"
    dimension_answered: str  # The dimension being explored
    value_chosen: str  # The value chosen for that dimension
    reasoning: str  # Why this classification was made


@dataclass
class ExpansionResult:
    """Result of I-MCTS expansion."""
    idea_title: str
    hypothesis: str
    implementation: str  # What to change in train.py
    dimension_answered: str
    value_chosen: str
    relationship: str  # "child" or "sibling"
    reasoning: str


@dataclass
class BacktrackDecision:
    """Result of backtrack analysis."""
    should_backtrack: bool
    backtrack_depth: int  # How many levels up (0 = stay, 1 = parent, 2 = grandparent, etc.)
    reasoning: str
    suggested_dimension: Optional[str]  # What dimension to explore instead


def classify_idea_relationship(
    idea_title: str,
    idea_hypothesis: str,
    parent_node: Dict[str, Any],
    siblings: List[Dict[str, Any]],
    client: Any
) -> ClassificationResult:
    """
    Classify whether a new idea is a child or sibling (F2).

    Child: Answers an open dimension left unspecified by parent
    Sibling: Alternative answer to the same dimension as an existing sibling

    Args:
        idea_title: Title of the new idea
        idea_hypothesis: The hypothesis for this idea
        parent_node: Info about the parent node
        siblings: List of sibling nodes with their results
        client: Anthropic API client

    Returns:
        ClassificationResult with relationship type and metadata
    """
    siblings_text = ""
    if siblings:
        siblings_text = "Existing siblings at this level:\n"
        for i, sib in enumerate(siblings, 1):
            status = sib.get('status', 'unknown')
            score = sib.get('score', 'N/A')
            dim = sib.get('dimension', 'unknown')
            val = sib.get('value', 'unknown')
            siblings_text += f"  {i}. {sib['title']} (dimension: {dim}, value: {val}, status: {status}, score: {score})\n"
    else:
        siblings_text = "No existing siblings at this level."

    parent_text = ""
    if parent_node:
        open_dims = parent_node.get('open_dimensions', [])
        parent_text = f"""Parent node: {parent_node.get('title', 'Unknown')}
Parent hypothesis: {parent_node.get('hypothesis', 'N/A')}
Parent's open dimensions (unanswered questions): {', '.join(open_dims) if open_dims else 'None identified yet'}"""
    else:
        parent_text = "This is a root-level idea (no parent)."

    prompt = f"""Classify whether this new ML experiment idea is a CHILD or SIBLING of its parent.

DEFINITIONS:
- CHILD: The idea fills in an open dimension that the parent left unspecified.
  Example: Parent tried "add feature engineering", child tries "add polynomial features" (specifying HOW).

- SIBLING: The idea is an alternative answer to the same question as existing siblings.
  Example: One sibling tried "polynomial features", this idea tries "log transform" (same dimension, different value).

NEW IDEA:
Title: {idea_title}
Hypothesis: {idea_hypothesis}

{parent_text}

{siblings_text}

TASK: Classify this idea and identify:
1. Is this a CHILD or SIBLING?
2. What dimension does this answer? (e.g., "feature_type", "regularization_method", "model_architecture")
3. What value does it choose for that dimension?

Respond in JSON format:
{{
    "relationship": "child" or "sibling",
    "dimension_answered": "the dimension this idea explores",
    "value_chosen": "the specific value chosen",
    "reasoning": "brief explanation of classification"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    # Parse JSON from response
    try:
        # Find JSON block
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return ClassificationResult(
                relationship=data.get('relationship', 'child'),
                dimension_answered=data.get('dimension_answered', 'unknown'),
                value_chosen=data.get('value_chosen', idea_title),
                reasoning=data.get('reasoning', ''),
            )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: default to child
    return ClassificationResult(
        relationship='child',
        dimension_answered='experiment',
        value_chosen=idea_title,
        reasoning='Could not parse LLM response, defaulting to child',
    )


def generate_next_experiment(
    selected_node: Dict[str, Any],
    siblings: List[Dict[str, Any]],
    parent_node: Optional[Dict[str, Any]],
    strategy_md: str,
    current_score: float,
    baseline_score: float,
    client: Any
) -> Optional[ExpansionResult]:
    """
    Generate the next experiment using I-MCTS expansion pattern (F3).

    Uses context from:
    - Selected node (current position in tree)
    - All siblings with their results (what's been tried at this level)
    - Parent node (the question being answered)

    Args:
        selected_node: The node selected by UCB1
        siblings: Sibling nodes with their results
        parent_node: The parent node context
        strategy_md: Current strategy document
        current_score: Current best score
        baseline_score: Baseline score for context
        client: Anthropic API client

    Returns:
        ExpansionResult with the new experiment, or None if no ideas
    """
    # Format sibling results
    siblings_text = ""
    if siblings:
        siblings_text = "\n\nSIBLING EXPERIMENTS (same level, different approaches):\n"
        for sib in siblings:
            status_emoji = {"improved": "✓", "no_improvement": "✗", "crashed": "💥"}.get(sib['status'], "?")
            score_str = f"{sib['score']:.4f}" if sib['score'] is not None else "N/A"
            siblings_text += f"  {status_emoji} {sib['title']}: {score_str}\n"
            if sib.get('hypothesis'):
                siblings_text += f"      Hypothesis: {sib['hypothesis']}\n"
    else:
        siblings_text = "\n\nNo sibling experiments yet at this level."

    # Format parent context
    parent_text = ""
    if parent_node:
        parent_text = f"""
PARENT NODE (the question being explored):
Title: {parent_node.get('title', 'Unknown')}
Hypothesis: {parent_node.get('hypothesis', 'N/A')}
Open dimensions to explore: {', '.join(parent_node.get('open_dimensions', [])) or 'Not specified'}
"""

    # Format selected node
    selected_text = f"""
SELECTED NODE (current position):
Title: {selected_node.get('title', 'Unknown')}
Current score: {selected_node.get('score', 'N/A')}
Fixed context: {selected_node.get('fixed_context', {})}
Open dimensions: {', '.join(selected_node.get('open_dimensions', [])) or 'None identified'}
"""

    prompt = f"""You are an ML research agent exploring an idea tree. Generate the NEXT experiment to try.

CONTEXT:
- Baseline score: {baseline_score:.4f}
- Current best score: {current_score:.4f}
- Improvement so far: {current_score - baseline_score:+.4f}

{selected_text}
{parent_text}
{siblings_text}

STRATEGY EXCERPT (first 1000 chars):
{strategy_md[:1000]}

TASK: Propose ONE new experiment that either:
1. Goes DEEPER on the selected node (CHILD): Fill in an open dimension
2. Tries a SIBLING alternative: Different value for the same dimension

IMPORTANT:
- Learn from sibling results - don't repeat approaches that didn't work
- If siblings with high scores exist, consider going deeper on those
- If all siblings failed, consider a different dimension entirely

Respond in JSON format:
{{
    "idea_title": "Short descriptive title",
    "hypothesis": "One sentence on why this should improve the score",
    "implementation": "What specific changes to make in train.py",
    "dimension_answered": "The dimension this explores (e.g., feature_type, regularization)",
    "value_chosen": "The specific value chosen",
    "relationship": "child" or "sibling",
    "reasoning": "Why this is the best next experiment given the context"
}}

If there are no good ideas left to explore at this branch, respond with:
{{"no_ideas": true, "reasoning": "explanation"}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    # Parse JSON from response
    try:
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            if data.get('no_ideas'):
                return None

            return ExpansionResult(
                idea_title=data.get('idea_title', 'Unnamed Experiment'),
                hypothesis=data.get('hypothesis', ''),
                implementation=data.get('implementation', ''),
                dimension_answered=data.get('dimension_answered', 'unknown'),
                value_chosen=data.get('value_chosen', ''),
                relationship=data.get('relationship', 'child'),
                reasoning=data.get('reasoning', ''),
            )
    except (json.JSONDecodeError, AttributeError):
        pass

    return None


def analyze_backtrack_depth(
    current_node: Dict[str, Any],
    ancestors: List[Dict[str, Any]],
    recent_experiments: List[Dict[str, Any]],
    plateau_reason: str,
    client: Any
) -> BacktrackDecision:
    """
    Analyze how far to backtrack when a branch plateaus (F5).

    This is LLM-driven rather than rule-based, allowing intelligent
    decisions about where in the tree to continue exploration.

    Args:
        current_node: The node where plateau was detected
        ancestors: List of ancestor nodes from parent to root
        recent_experiments: Recent experiment results
        plateau_reason: Description of why plateau was triggered
        client: Anthropic API client

    Returns:
        BacktrackDecision with depth and reasoning
    """
    # Format ancestor chain
    ancestors_text = "ANCESTOR CHAIN (parent → grandparent → ...):\n"
    for i, anc in enumerate(ancestors):
        score_str = f"{anc['score']:.4f}" if anc.get('score') is not None else "N/A"
        ancestors_text += f"  Level {i+1}: {anc['title']} (score: {score_str})\n"
        if anc.get('open_dimensions'):
            ancestors_text += f"           Open dimensions: {', '.join(anc['open_dimensions'])}\n"

    # Format recent experiments
    recent_text = "RECENT EXPERIMENTS:\n"
    for exp in recent_experiments[-5:]:
        status_emoji = {"improved": "✓", "no_improvement": "✗", "crashed": "💥"}.get(exp.get('status'), "?")
        score_str = f"{exp['score']:.4f}" if exp.get('score') is not None else "N/A"
        recent_text += f"  {status_emoji} {exp['title']}: {score_str}\n"

    prompt = f"""You are analyzing a plateau in an ML experiment tree. Decide how far to backtrack.

CURRENT NODE (where plateau detected):
Title: {current_node.get('title', 'Unknown')}
Score: {current_node.get('score', 'N/A')}
Dimension: {current_node.get('dimension_answered', 'unknown')}

{ancestors_text}

{recent_text}

PLATEAU REASON: {plateau_reason}

TASK: Decide how far to backtrack in the tree.

Options:
- Depth 0: Stay at current level, try more siblings
- Depth 1: Go to parent, try a different approach there
- Depth 2: Go to grandparent, the current direction may be fundamentally wrong
- Depth N: Go even higher if the entire strategy branch is failing

Consider:
1. Are siblings consistently failing? → Backtrack further
2. Did the parent have good results? → Stay close, try siblings
3. Is the dimension itself problematic? → Backtrack to where that dimension was introduced

Respond in JSON format:
{{
    "should_backtrack": true or false,
    "backtrack_depth": 0, 1, 2, etc.,
    "reasoning": "explanation of decision",
    "suggested_dimension": "what dimension to explore instead (optional)"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    # Parse JSON
    try:
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return BacktrackDecision(
                should_backtrack=data.get('should_backtrack', True),
                backtrack_depth=data.get('backtrack_depth', 1),
                reasoning=data.get('reasoning', ''),
                suggested_dimension=data.get('suggested_dimension'),
            )
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: backtrack one level
    return BacktrackDecision(
        should_backtrack=True,
        backtrack_depth=1,
        reasoning='Could not parse LLM response, defaulting to parent',
        suggested_dimension=None,
    )


def discover_open_dimensions(
    experiment_result: Dict[str, Any],
    current_dimensions: List[str],
    train_py_content: str,
    client: Any
) -> List[str]:
    """
    Discover new open dimensions after an experiment (F6).

    This allows the tree to grow dynamically as we learn more
    about what variations are possible.

    Args:
        experiment_result: The result of the experiment
        current_dimensions: Currently known dimensions
        train_py_content: Current train.py code
        client: Anthropic API client

    Returns:
        List of newly discovered dimensions
    """
    prompt = f"""After running an ML experiment, identify NEW dimensions that could be explored.

EXPERIMENT:
Title: {experiment_result.get('title', 'Unknown')}
Hypothesis: {experiment_result.get('hypothesis', 'N/A')}
Result: {experiment_result.get('status', 'unknown')} (score: {experiment_result.get('score', 'N/A')})

CURRENT KNOWN DIMENSIONS: {', '.join(current_dimensions) if current_dimensions else 'None yet'}

TRAIN.PY EXCERPT (first 500 chars):
{train_py_content[:500]}

TASK: Based on the experiment and code, identify 1-3 NEW dimensions that could be explored.

A dimension is a category of choices, like:
- "feature_engineering_method" (values: polynomial, log_transform, binning)
- "model_type" (values: lgbm, xgboost, neural_net)
- "regularization_type" (values: L1, L2, dropout)
- "hyperparameter_tuning" (values: grid_search, optuna, manual)

Only list dimensions NOT already in the current list.

Respond in JSON format:
{{
    "new_dimensions": ["dimension1", "dimension2"],
    "reasoning": "why these dimensions are worth exploring"
}}

If no new dimensions are apparent, respond with:
{{"new_dimensions": [], "reasoning": "explanation"}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    try:
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('new_dimensions', [])
    except (json.JSONDecodeError, AttributeError):
        pass

    return []
