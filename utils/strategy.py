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

    # Fill in template variables (use 'or' to handle None values)
    prompt = prompt.replace("{competition_name}", competition_meta.get('name') or 'Unknown')
    prompt = prompt.replace("{problem_type}", competition_meta.get('problem_type') or 'unknown')
    prompt = prompt.replace("{metric}", competition_meta.get('metric') or 'unknown')
    prompt = prompt.replace("{metric_direction}", competition_meta.get('metric_direction') or 'higher_better')
    prompt = prompt.replace("{dataset_description}", competition_meta.get('description') or 'No description')
    prompt = prompt.replace("{baseline_score}", str(baseline_score))
    prompt = prompt.replace("{baseline_model}", competition_meta.get('baseline_model') or 'LightGBM')
    prompt = prompt.replace("{paper_summaries}", papers_summary or '')

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

    prompt = prompt.replace("{competition_name}", competition_meta.get('name') or 'Unknown')
    prompt = prompt.replace("{problem_type}", competition_meta.get('problem_type') or 'unknown')
    prompt = prompt.replace("{strategy_md}", strategy_md or '')
    prompt = prompt.replace("{baseline_score}", str(baseline_score))
    prompt = prompt.replace("{paper_summaries}", papers_summary or '')

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


def sanitize_idea_block(block: str) -> Optional[str]:
    """
    Sanitize an idea block to ensure proper IDEAS.md format.

    Args:
        block: Raw idea text that may be malformed

    Returns:
        Properly formatted idea block or None if unfixable
    """
    if not block.strip():
        return None

    lines = block.strip().split('\n')

    # Extract title - look for ## IDEA: or **IDEA:** or just a title-like first line
    title = None
    title_line_idx = -1

    for i, line in enumerate(lines):
        if line.startswith('## IDEA:'):
            title = line.replace('## IDEA:', '').strip()
            title_line_idx = i
            break
        elif '**IDEA:' in line or '**Idea:' in line or '**idea:' in line:
            # Convert markdown bold to proper format - strip both prefix and trailing **
            title = re.sub(r'\*\*[Ii][Dd][Ee][Aa]:\s*', '', line)
            title = title.strip().rstrip('*').strip()
            title_line_idx = i
            break

    # If no IDEA: marker found, this isn't a valid idea block
    if not title:
        return None

    # Build the rest of the content
    rest_content = '\n'.join(lines[title_line_idx + 1:]) if title_line_idx >= 0 else '\n'.join(lines[1:])

    # Check for required fields, extract or add defaults
    source = "re-research"
    risk = "medium"
    gain = "medium"

    source_match = re.search(r'[Ss]ource:\s*(.+)', rest_content)
    if source_match:
        source = source_match.group(1).strip()

    risk_match = re.search(r'[Rr]isk:\s*(\w+)', rest_content)
    if risk_match:
        risk = risk_match.group(1).strip().lower()

    gain_match = re.search(r'[Ee]stimated [Gg]ain:\s*(\w+)', rest_content)
    if gain_match:
        gain = gain_match.group(1).strip().lower()

    # Extract hypothesis, implementation, validation
    hypothesis = ""
    implementation = ""
    validation = ""

    # Try various formats for these fields
    hyp_match = re.search(r'[Hh]ypothesis:\s*(.+?)(?=[Ii]mplementation:|$)', rest_content, re.DOTALL)
    if hyp_match:
        hypothesis = hyp_match.group(1).strip()

    impl_match = re.search(r'[Ii]mplementation:\s*(.+?)(?=[Vv]alidation:|$)', rest_content, re.DOTALL)
    if impl_match:
        implementation = impl_match.group(1).strip()

    val_match = re.search(r'[Vv]alidation:\s*(.+?)$', rest_content, re.DOTALL)
    if val_match:
        validation = val_match.group(1).strip()

    # If we couldn't extract structured fields, use the whole content as implementation
    if not implementation and not hypothesis:
        # Remove any metadata lines we found
        impl_text = rest_content
        impl_text = re.sub(r'[Ss]ource:.*\n?', '', impl_text)
        impl_text = re.sub(r'[Rr]isk:.*\n?', '', impl_text)
        impl_text = re.sub(r'[Ee]stimated [Gg]ain:.*\n?', '', impl_text)
        impl_text = re.sub(r'[Ss]tatus:.*\n?', '', impl_text)
        impl_text = re.sub(r'---+', '', impl_text)
        impl_text = re.sub(r'===+', '', impl_text)
        implementation = impl_text.strip()
        hypothesis = "May improve model performance based on re-research findings."
        validation = "Compare CV score before and after."

    # Build properly formatted block
    formatted = f"""## IDEA: {title}
Source: {source}
Risk: {risk}
Estimated gain: {gain}
Status: pending
---
Hypothesis: {hypothesis}
Implementation: {implementation}
Validation: {validation}
==="""

    return formatted


def append_ideas_to_file(path: Path, new_ideas_md: str) -> int:
    """
    Append new ideas to existing IDEAS.md file with validation.

    Args:
        path: Path to IDEAS.md
        new_ideas_md: New ideas content to append

    Returns:
        Number of valid ideas appended
    """
    path = Path(path)

    with open(path, 'r') as f:
        existing = f.read()

    # Extract existing idea titles to avoid duplicates
    existing_titles = set()
    for match in re.finditer(r'## IDEA:\s*(.+)', existing):
        title = match.group(1).strip().lower()
        # Also match partial titles (first 30 chars) to catch truncated duplicates
        existing_titles.add(title)
        if len(title) > 30:
            existing_titles.add(title[:30])

    # Split input by potential idea boundaries
    # Look for ## IDEA:, **IDEA:, numbered items, or double newlines
    raw_blocks = re.split(r'(?=## IDEA:|(?=\*\*[Ii]dea:)|(?=\d+\.\s+\*\*)|(?:\n\n(?=[A-Z])))', new_ideas_md)

    valid_ideas = []
    for block in raw_blocks:
        sanitized = sanitize_idea_block(block)
        if sanitized:
            # Extract title from sanitized block and check for duplicates
            title_match = re.search(r'## IDEA:\s*(.+)', sanitized)
            if title_match:
                title = title_match.group(1).strip().lower()
                # Check both full title and prefix
                if title in existing_titles or (len(title) > 30 and title[:30] in existing_titles):
                    print(f"  Skipping duplicate idea: {title[:50]}...")
                    continue
                existing_titles.add(title)
                if len(title) > 30:
                    existing_titles.add(title[:30])
            valid_ideas.append(sanitized)

    if not valid_ideas:
        print("  Warning: No valid ideas found in re-research output")
        return 0

    # Add separator and append valid ideas
    ideas_text = "\n\n".join(valid_ideas)
    combined = existing.rstrip() + "\n\n---\n\n# Re-research Ideas\n\n" + ideas_text

    with open(path, 'w') as f:
        f.write(combined)

    print(f"  Appended {len(valid_ideas)} sanitized ideas to IDEAS.md")
    return len(valid_ideas)


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
