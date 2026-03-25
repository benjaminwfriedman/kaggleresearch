"""
Re-research logic for when the experiment loop plateaus.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .literature import (
    search_papers,
    load_search_history,
    save_search_history,
    cache_papers,
    build_search_query,
    format_papers_for_prompt,
    Paper,
)


@dataclass
class ReresearchResult:
    """Result of a re-research attempt."""
    outcome: str  # "new_ideas", "pivot", "no_new_ideas"
    new_ideas_md: Optional[str] = None
    pivot_strategy_md: Optional[str] = None
    pivot_strategy_name: Optional[str] = None
    reasoning: str = ""


def log_reresearch_response(
    log_dir: Path,
    attempt_type: str,
    response_text: str,
    parsed_result: "ReresearchResult"
) -> None:
    """
    Log re-research LLM response for debugging.

    Args:
        log_dir: Directory to save logs (typically literature_dir)
        attempt_type: "new_angle" or "reread"
        response_text: Raw LLM response
        parsed_result: Parsed result object
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"reresearch_{attempt_type}_{timestamp}.log"

    log_content = f"""=== Re-research Log ===
Timestamp: {datetime.now().isoformat()}
Attempt Type: {attempt_type}
Parsed Outcome: {parsed_result.outcome}
Parsed Reasoning: {parsed_result.reasoning}

=== Raw LLM Response ===
{response_text}

=== Parsed new_ideas_md ===
{parsed_result.new_ideas_md or "(none)"}

=== Parsed pivot_strategy_md ===
{parsed_result.pivot_strategy_md or "(none)"}
"""

    with open(log_path, 'w') as f:
        f.write(log_content)

    print(f"  Re-research response logged to: {log_path.name}")


def reresearch_new_angle(
    strategy_md: str,
    failure_summary: str,
    search_history_path: Path,
    literature_cache_path: Path,
    problem_type: str,
    metric: str,
    literature_depth: int,
    client: Any,  # Anthropic client
    log_dir: Optional[Path] = None
) -> ReresearchResult:
    """
    Attempt 1 of re-research: Search for new papers with failure context.

    Args:
        strategy_md: Current STRATEGY.md content
        failure_summary: Summary of failed experiments
        search_history_path: Path to search_history.json
        literature_cache_path: Path to papers.json
        problem_type: Competition problem type
        metric: Evaluation metric
        literature_depth: Number of papers to retrieve
        client: Anthropic API client
        log_dir: Directory to save debug logs (optional)

    Returns:
        ReresearchResult with outcome and optional new content
    """
    # Build new search query informed by failures
    exclude_queries = load_search_history(search_history_path)

    query = build_search_query(
        problem_type=problem_type,
        metric=metric,
        context=failure_summary[:200],  # Truncate context
    )

    # Search for new papers
    new_papers = search_papers(
        query=query,
        n=literature_depth,
        exclude_queries=exclude_queries,
        problem_type=problem_type,
    )

    # Save search history
    save_search_history(query, search_history_path)

    # Cache new papers
    if new_papers:
        cache_papers(new_papers, literature_cache_path)

    papers_summary = format_papers_for_prompt(new_papers)

    # Load re-research prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "reresearch.md"
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    # Extract learning log from strategy
    learning_log = extract_learning_log(strategy_md)

    prompt = prompt_template.replace("{strategy_md}", strategy_md)
    prompt = prompt.replace("{failed_experiments}", failure_summary)
    prompt = prompt.replace("{learning_log}", learning_log)
    prompt = prompt.replace("{new_paper_summaries}", papers_summary)

    # Call LLM
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text

    # Parse response
    result = parse_reresearch_response(response_text)

    # Log for debugging
    if log_dir:
        log_reresearch_response(log_dir, "new_angle", response_text, result)

    return result


def reresearch_reread(
    cached_papers_path: Path,
    strategy_md: str,
    failure_summary: str,
    client: Any,
    log_dir: Optional[Path] = None
) -> ReresearchResult:
    """
    Attempt 2 of re-research: Re-read existing papers with failure context.

    Args:
        cached_papers_path: Path to papers.json
        strategy_md: Current STRATEGY.md content
        failure_summary: Summary of failed experiments
        client: Anthropic API client
        log_dir: Directory to save debug logs (optional)

    Returns:
        ReresearchResult with outcome and optional new content
    """
    from .literature import load_cached_papers, format_papers_for_prompt

    # Load all cached papers
    papers = load_cached_papers(cached_papers_path)
    papers_summary = format_papers_for_prompt(papers, max_papers=15)

    learning_log = extract_learning_log(strategy_md)

    prompt = f"""You previously read these papers:

{papers_summary}

Since then, the following experiments have failed:

{failure_summary}

The current STRATEGY.md learning log is:

{learning_log}

Re-read the papers with this new context. Are there methods you previously
overlooked that might explain why the current approach is stuck?

If you find new applicable ideas, produce new IDEAS.md entries following this format:

## IDEA: [Short descriptive title]
Source: [Paper title] (arxiv:[ID]) | empirical | derived-from-strategy
Risk: low | medium | high
Estimated gain: small | medium | large
Status: pending
---
Hypothesis: [One sentence on why this should help]
Implementation: [Precise description of what to change in train.py]
Validation: [How to know it worked]
===

If you cannot find any new applicable ideas, respond with exactly: NO_NEW_IDEAS

Your response (either new IDEAS.md entries or NO_NEW_IDEAS):"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    if "NO_NEW_IDEAS" in response_text.upper():
        result = ReresearchResult(
            outcome="no_new_ideas",
            reasoning="Re-reading papers did not yield new applicable ideas.",
        )
    elif "## IDEA:" in response_text:
        result = ReresearchResult(
            outcome="new_ideas",
            new_ideas_md=response_text,
            reasoning="Found new ideas by re-reading papers with failure context.",
        )
    else:
        result = ReresearchResult(
            outcome="no_new_ideas",
            reasoning="Response did not contain valid IDEAS.md entries.",
        )

    # Log for debugging
    if log_dir:
        log_reresearch_response(log_dir, "reread", response_text, result)

    return result


def parse_reresearch_response(response_text: str) -> ReresearchResult:
    """
    Parse the LLM response from re-research.

    Args:
        response_text: Raw response from LLM

    Returns:
        Parsed ReresearchResult
    """
    import re

    # Try to parse as JSON first
    try:
        # Find JSON block - look for ```json code blocks first
        json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_block_match:
            json_str = json_block_match.group(1)
        else:
            # Try to find raw JSON object with balanced braces
            # Find the first { and match to its closing }
            start_idx = response_text.find('{')
            if start_idx != -1:
                depth = 0
                end_idx = start_idx
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                json_str = response_text[start_idx:end_idx]
            else:
                json_str = None

        if json_str:
            data = json.loads(json_str)

            decision = data.get('decision', '').upper()

            if 'NEW_IDEAS' in decision:
                new_ideas_md = data.get('new_ideas_md', '')
                # Validate that new_ideas_md actually contains parseable ideas
                if new_ideas_md and '## IDEA:' in new_ideas_md:
                    return ReresearchResult(
                        outcome="new_ideas",
                        new_ideas_md=new_ideas_md,
                        reasoning=data.get('reasoning', ''),
                    )
                else:
                    # LLM said NEW_IDEAS but didn't provide valid format
                    return ReresearchResult(
                        outcome="no_new_ideas",
                        reasoning=f"LLM returned NEW_IDEAS but content was not in valid format. Reasoning: {data.get('reasoning', '')}",
                    )
            elif 'PIVOT' in decision:
                return ReresearchResult(
                    outcome="pivot",
                    pivot_strategy_md=data.get('pivot_strategy_md', ''),
                    pivot_strategy_name=data.get('pivot_strategy_name', 'New Strategy'),
                    reasoning=data.get('reasoning', ''),
                )
            else:
                return ReresearchResult(
                    outcome="no_new_ideas",
                    reasoning=data.get('reasoning', 'No new ideas found.'),
                )

    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: try to parse from text
    response_upper = response_text.upper()

    if 'PIVOT' in response_upper:
        # Try to extract new strategy content
        strategy_match = re.search(r'# Strategy:(.+?)(?=##|$)', response_text, re.DOTALL)
        strategy_content = strategy_match.group(0) if strategy_match else response_text

        return ReresearchResult(
            outcome="pivot",
            pivot_strategy_md=strategy_content,
            pivot_strategy_name="Alternative Strategy",
            reasoning="Pivot recommended based on paper review.",
        )

    if 'NO_NEW_IDEAS' in response_upper or 'NO NEW IDEAS' in response_upper:
        return ReresearchResult(
            outcome="no_new_ideas",
            reasoning="No new ideas found in re-research.",
        )

    # Check for IDEAS.md format
    if '## IDEA:' in response_text:
        return ReresearchResult(
            outcome="new_ideas",
            new_ideas_md=response_text,
            reasoning="Found new ideas to try.",
        )

    # Default to no new ideas
    return ReresearchResult(
        outcome="no_new_ideas",
        reasoning="Could not parse re-research response.",
    )


def extract_learning_log(strategy_md: str) -> str:
    """
    Extract the Learning Log section from STRATEGY.md.

    Args:
        strategy_md: Full STRATEGY.md content

    Returns:
        Learning log content or empty string
    """
    import re

    match = re.search(r'## Learning Log\s*\n(.+?)(?=\n##|$)', strategy_md, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def should_attempt_reresearch(
    reresearch_attempts: int,
    max_attempts: int = 2
) -> bool:
    """
    Check if another re-research attempt should be made.

    Args:
        reresearch_attempts: Number of attempts already made
        max_attempts: Maximum allowed attempts

    Returns:
        True if should attempt, False otherwise
    """
    return reresearch_attempts < max_attempts


def handle_reresearch_result(
    result: ReresearchResult,
    ideas_path: Path,
    strategy_path: Path,
    archive_path: Path,
) -> Dict[str, Any]:
    """
    Handle the result of a re-research attempt.

    Args:
        result: ReresearchResult from re-research
        ideas_path: Path to IDEAS.md
        strategy_path: Path to STRATEGY.md
        archive_path: Path to archived_strategies.md

    Returns:
        Dict with action to take and any relevant data
    """
    from .strategy import append_ideas_to_file, archive_strategy

    if result.outcome == "new_ideas" and result.new_ideas_md:
        # Append new ideas to existing file (with sanitization)
        num_added = append_ideas_to_file(ideas_path, result.new_ideas_md)
        if num_added > 0:
            return {
                "action": "continue_exploit",
                "message": f"Added {num_added} new ideas from re-research. Continuing exploit phase.",
            }
        else:
            # Sanitization failed to extract valid ideas
            return {
                "action": "halt",
                "message": "Re-research returned content but no valid ideas could be parsed.",
            }

    elif result.outcome == "pivot" and result.pivot_strategy_md:
        # Archive current strategy
        archive_strategy(strategy_path, archive_path)

        return {
            "action": "start_branch",
            "new_strategy_md": result.pivot_strategy_md,
            "new_strategy_name": result.pivot_strategy_name,
            "message": f"Pivoting to new strategy: {result.pivot_strategy_name}",
        }

    else:
        return {
            "action": "halt",
            "message": "Re-research exhausted. No new ideas or pivots found.",
        }
