"""
Strategy selection and IDEAS.md management.
Handles LLM calls for strategy generation and idea parsing.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class Idea:
    """Represents a single experiment idea from IDEAS.md."""
    title: str
    source: str
    risk: str  # "low", "medium", "high"
    estimated_gain: str  # "small", "medium", "large"
    status: str  # "pending", "running", "improved", "no_improvement", "crashed", "skipped"
    hypothesis: str
    implementation: str
    validation: str
    index: int = 0  # Position in the file

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Idea":
        return cls(**data)


def select_strategy(
    papers_summary: str,
    competition_meta: Dict[str, Any],
    baseline_score: float,
    client: Any  # Anthropic client
) -> str:
    """
    Use LLM to select a strategy and generate STRATEGY.md content.

    Args:
        papers_summary: Formatted paper summaries
        competition_meta: Competition metadata dict
        baseline_score: Score from baseline model
        client: Anthropic API client

    Returns:
        STRATEGY.md content as string
    """
    prompt_template = Path(__file__).parent.parent / "prompts" / "strategy_selection.md"

    with open(prompt_template, 'r') as f:
        prompt = f.read()

    # Fill in template variables
    prompt = prompt.replace("{competition_name}", competition_meta.get('name', 'Unknown'))
    prompt = prompt.replace("{problem_type}", competition_meta.get('problem_type', 'unknown'))
    prompt = prompt.replace("{metric}", competition_meta.get('metric', 'unknown'))
    prompt = prompt.replace("{metric_direction}", competition_meta.get('metric_direction', 'higher_better'))
    prompt = prompt.replace("{dataset_description}", competition_meta.get('description', 'No description'))
    prompt = prompt.replace("{baseline_score}", str(baseline_score))
    prompt = prompt.replace("{baseline_model}", competition_meta.get('baseline_model', 'LightGBM'))
    prompt = prompt.replace("{paper_summaries}", papers_summary)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def generate_ideas_md(
    papers_summary: str,
    strategy_md: str,
    competition_meta: Dict[str, Any],
    baseline_score: float,
    client: Any
) -> str:
    """
    Use LLM to generate IDEAS.md content.

    Args:
        papers_summary: Formatted paper summaries
        strategy_md: Current STRATEGY.md content
        competition_meta: Competition metadata
        baseline_score: Baseline score
        client: Anthropic API client

    Returns:
        IDEAS.md content as string
    """
    prompt_template = Path(__file__).parent.parent / "prompts" / "ideas_generation.md"

    with open(prompt_template, 'r') as f:
        prompt = f.read()

    prompt = prompt.replace("{competition_name}", competition_meta.get('name', 'Unknown'))
    prompt = prompt.replace("{problem_type}", competition_meta.get('problem_type', 'unknown'))
    prompt = prompt.replace("{strategy_md}", strategy_md)
    prompt = prompt.replace("{baseline_score}", str(baseline_score))
    prompt = prompt.replace("{paper_summaries}", papers_summary)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def parse_ideas_md(path: Path) -> List[Idea]:
    """
    Parse IDEAS.md file into list of Idea objects.

    Args:
        path: Path to IDEAS.md file

    Returns:
        List of Idea objects
    """
    path = Path(path)

    if not path.exists():
        return []

    with open(path, 'r') as f:
        content = f.read()

    ideas = []

    # Split by idea headers
    idea_blocks = re.split(r'(?=## IDEA:)', content)

    for i, block in enumerate(idea_blocks):
        if not block.strip() or not block.startswith('## IDEA:'):
            continue

        idea = parse_idea_block(block, i)
        if idea:
            ideas.append(idea)

    return ideas


def parse_idea_block(block: str, index: int) -> Optional[Idea]:
    """
    Parse a single idea block into an Idea object.

    Args:
        block: Text block for one idea
        index: Index position

    Returns:
        Idea object or None if parsing fails
    """
    try:
        # Extract title
        title_match = re.search(r'## IDEA:\s*(.+)', block)
        title = title_match.group(1).strip() if title_match else "Unknown"

        # Extract metadata fields
        source_match = re.search(r'Source:\s*(.+)', block)
        source = source_match.group(1).strip() if source_match else "empirical"

        risk_match = re.search(r'Risk:\s*(\w+)', block)
        risk = risk_match.group(1).strip().lower() if risk_match else "medium"

        gain_match = re.search(r'Estimated gain:\s*(\w+)', block)
        estimated_gain = gain_match.group(1).strip().lower() if gain_match else "medium"

        status_match = re.search(r'Status:\s*(\w+)', block)
        status = status_match.group(1).strip().lower() if status_match else "pending"

        # Extract detailed fields (between --- and ===)
        detail_match = re.search(r'---\s*\n(.+?)(?:===|$)', block, re.DOTALL)
        details = detail_match.group(1) if detail_match else ""

        hyp_match = re.search(r'Hypothesis:\s*(.+?)(?=Implementation:|$)', details, re.DOTALL)
        hypothesis = hyp_match.group(1).strip() if hyp_match else ""

        impl_match = re.search(r'Implementation:\s*(.+?)(?=Validation:|$)', details, re.DOTALL)
        implementation = impl_match.group(1).strip() if impl_match else ""

        val_match = re.search(r'Validation:\s*(.+?)$', details, re.DOTALL)
        validation = val_match.group(1).strip() if val_match else ""

        return Idea(
            title=title,
            source=source,
            risk=risk,
            estimated_gain=estimated_gain,
            status=status,
            hypothesis=hypothesis,
            implementation=implementation,
            validation=validation,
            index=index,
        )

    except Exception as e:
        print(f"Error parsing idea block: {e}")
        return None


def update_idea_status(path: Path, idea_title: str, new_status: str) -> None:
    """
    Update the status of an idea in IDEAS.md.

    Args:
        path: Path to IDEAS.md
        idea_title: Title of the idea to update
        new_status: New status value
    """
    path = Path(path)

    with open(path, 'r') as f:
        content = f.read()

    # Find and replace the status line for this idea
    # Pattern: ## IDEA: <title>\n...Status: <old_status>
    pattern = rf'(## IDEA:\s*{re.escape(idea_title)}.*?Status:\s*)(\w+)'
    replacement = rf'\g<1>{new_status}'

    updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(path, 'w') as f:
        f.write(updated_content)


def append_learning_log(strategy_path: Path, note: str) -> None:
    """
    Append a note to the Learning Log section of STRATEGY.md.

    Args:
        strategy_path: Path to STRATEGY.md
        note: Note to append
    """
    strategy_path = Path(strategy_path)

    with open(strategy_path, 'r') as f:
        content = f.read()

    # Find or create Learning Log section
    if '## Learning Log' in content:
        # Append to existing section
        content = content.rstrip() + f"\n- {note}\n"
    else:
        # Create new section
        content = content.rstrip() + f"\n\n## Learning Log\n\n- {note}\n"

    with open(strategy_path, 'w') as f:
        f.write(content)


def format_learning_log_entry(
    experiment_index: int,
    idea_title: str,
    status: str,
    score: Optional[float],
    delta: Optional[float] = None
) -> str:
    """
    Format a learning log entry.

    Args:
        experiment_index: Experiment number
        idea_title: Title of the idea
        status: Result status
        score: Achieved score
        delta: Score improvement

    Returns:
        Formatted log entry string
    """
    status_upper = status.upper()

    if status == 'improved' and delta is not None:
        return f"Experiment {experiment_index} ({status_upper}): {idea_title} — improved score by {delta:.4f} ({delta/score*100:.2f}%)"
    elif score is not None:
        return f"Experiment {experiment_index} ({status_upper}): {idea_title} — score: {score:.4f}"
    else:
        return f"Experiment {experiment_index} ({status_upper}): {idea_title}"


def get_next_pending_idea(ideas: List[Idea]) -> Optional[Idea]:
    """
    Get the next pending idea to try.

    Args:
        ideas: List of all ideas

    Returns:
        Next pending Idea or None if all done
    """
    for idea in ideas:
        if idea.status == 'pending':
            return idea
    return None


def count_ideas_by_status(ideas: List[Idea]) -> Dict[str, int]:
    """
    Count ideas by their status.

    Args:
        ideas: List of ideas

    Returns:
        Dict mapping status to count
    """
    counts = {
        'pending': 0,
        'running': 0,
        'improved': 0,
        'no_improvement': 0,
        'crashed': 0,
        'skipped': 0,
    }

    for idea in ideas:
        status = idea.status.lower()
        if status in counts:
            counts[status] += 1

    return counts


def append_ideas_to_file(path: Path, new_ideas_md: str) -> None:
    """
    Append new ideas to existing IDEAS.md file.

    Args:
        path: Path to IDEAS.md
        new_ideas_md: New ideas content to append
    """
    path = Path(path)

    with open(path, 'r') as f:
        existing = f.read()

    # Add separator
    combined = existing.rstrip() + "\n\n---\n\n# Re-research Ideas\n\n" + new_ideas_md

    with open(path, 'w') as f:
        f.write(combined)


def archive_strategy(strategy_path: Path, archive_path: Path) -> None:
    """
    Archive current STRATEGY.md before pivot.

    Args:
        strategy_path: Path to current STRATEGY.md
        archive_path: Path to archived_strategies.md
    """
    strategy_path = Path(strategy_path)
    archive_path = Path(archive_path)

    with open(strategy_path, 'r') as f:
        current_strategy = f.read()

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to archive with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    archive_entry = f"\n\n---\n\n# Archived Strategy ({timestamp})\n\n{current_strategy}"

    with open(archive_path, 'a') as f:
        f.write(archive_entry)
