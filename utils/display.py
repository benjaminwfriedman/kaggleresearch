"""
Display utilities for Colab notebook output.
Renders tables, widgets, and summaries.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


def render_live_table(
    experiments: List[Dict],
    current_branch: str,
    best_score: float,
    baseline_score: float,
    metric_direction: str = "higher_better"
) -> str:
    """
    Render a live experiment progress table.

    Args:
        experiments: List of experiment log dicts
        current_branch: Current git branch
        best_score: Current best score
        baseline_score: Baseline score
        metric_direction: "higher_better" or "lower_better"

    Returns:
        HTML string for display
    """
    # Calculate improvement
    if metric_direction == "higher_better":
        improvement = best_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score else 0
    else:
        improvement = baseline_score - best_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score else 0

    # Build header
    html = f"""
    <style>
        .kr-table {{ font-family: monospace; border-collapse: collapse; width: 100%; }}
        .kr-table th, .kr-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .kr-table th {{ background-color: #4CAF50; color: white; }}
        .kr-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .kr-improved {{ color: #2e7d32; font-weight: bold; }}
        .kr-failed {{ color: #c62828; }}
        .kr-crashed {{ color: #ff6f00; }}
        .kr-header {{ margin-bottom: 10px; }}
    </style>

    <div class="kr-header">
        <strong>Branch:</strong> {current_branch} |
        <strong>Best Score:</strong> {best_score:.6f} |
        <strong>Baseline:</strong> {baseline_score:.6f} |
        <strong>Improvement:</strong> <span class="{'kr-improved' if improvement > 0 else 'kr-failed'}">
            {'+' if improvement > 0 else ''}{improvement:.6f} ({improvement_pct:+.2f}%)
        </span>
    </div>

    <table class="kr-table">
        <tr>
            <th>#</th>
            <th>Idea</th>
            <th>Status</th>
            <th>Score</th>
            <th>Delta</th>
            <th>Duration</th>
        </tr>
    """

    for i, exp in enumerate(experiments[-20:], 1):  # Show last 20
        status = exp.get('status', 'unknown')
        score = exp.get('score')
        prev_best = exp.get('previous_best', baseline_score)

        # Determine CSS class
        if status == 'improved':
            css_class = 'kr-improved'
        elif status == 'crashed':
            css_class = 'kr-crashed'
        else:
            css_class = 'kr-failed'

        # Calculate delta
        if score is not None and prev_best is not None:
            delta = score - prev_best
            delta_str = f"{'+' if delta > 0 else ''}{delta:.6f}"
        else:
            delta_str = "N/A"

        # Format duration
        duration = exp.get('duration_seconds', 0)
        duration_str = f"{duration:.1f}s" if duration < 60 else f"{duration/60:.1f}m"

        # For crashed experiments, show error in title tooltip
        idea_title = exp.get('idea_title', 'Unknown')[:40]
        error_msg = exp.get('error_message', '')
        if status == 'crashed' and error_msg:
            # Truncate error for tooltip and escape HTML
            error_preview = error_msg[:200].replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
            title_cell = f'<span title="{error_preview}" style="cursor:help">{idea_title}</span>'
        else:
            title_cell = idea_title

        html += f"""
        <tr class="{css_class}">
            <td>{i}</td>
            <td>{title_cell}</td>
            <td>{status}</td>
            <td>{f'{score:.6f}' if score else 'N/A'}</td>
            <td>{delta_str}</td>
            <td>{duration_str}</td>
        </tr>
        """

    html += "</table>"

    return html


def render_strategy_and_ideas_widget(
    strategy_path: Path,
    ideas_path: Path
) -> str:
    """
    Render STRATEGY.md and IDEAS.md side by side for review.

    Args:
        strategy_path: Path to STRATEGY.md
        ideas_path: Path to IDEAS.md

    Returns:
        HTML string for display
    """
    strategy_path = Path(strategy_path)
    ideas_path = Path(ideas_path)

    strategy_content = ""
    if strategy_path.exists():
        with open(strategy_path, 'r') as f:
            strategy_content = f.read()

    ideas_content = ""
    if ideas_path.exists():
        with open(ideas_path, 'r') as f:
            ideas_content = f.read()

    html = f"""
    <style>
        .kr-review-container {{ display: flex; gap: 20px; }}
        .kr-review-panel {{ flex: 1; max-height: 500px; overflow-y: auto;
                           border: 1px solid #ddd; padding: 10px;
                           background-color: #fafafa; }}
        .kr-review-panel h3 {{ position: sticky; top: 0; background: #4CAF50;
                              color: white; padding: 10px; margin: -10px -10px 10px -10px; }}
        .kr-review-panel pre {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>

    <div class="kr-review-container">
        <div class="kr-review-panel">
            <h3>STRATEGY.md</h3>
            <pre>{_escape_html(strategy_content)}</pre>
        </div>
        <div class="kr-review-panel">
            <h3>IDEAS.md</h3>
            <pre>{_escape_html(ideas_content)}</pre>
        </div>
    </div>

    <p><em>Review and edit these files in Google Drive before clicking "Approve Strategy & Start Loop"</em></p>
    """

    return html


def render_summary(
    log_path: Path,
    checkpoint: Dict[str, Any],
    baseline_score: float,
    strategy_md_path: Optional[Path] = None,
    tree_path: Optional[Path] = None
) -> str:
    """
    Render the final summary after the loop completes.

    Args:
        log_path: Path to experiment_log.db
        checkpoint: Checkpoint state dict
        baseline_score: Baseline score
        strategy_md_path: Optional path to STRATEGY.md
        tree_path: Optional path to idea_tree.json

    Returns:
        Markdown string for display
    """
    from .experiment_runner import load_experiments

    experiments = load_experiments(log_path)
    final_score = checkpoint.get('best_score', baseline_score)

    # Calculate statistics
    total = len(experiments)
    improved = sum(1 for e in experiments if e.get('status') == 'improved')
    crashed = sum(1 for e in experiments if e.get('status') == 'crashed')
    no_improvement = total - improved - crashed

    # Improvement calculation
    if checkpoint.get('metric_direction', 'higher_better') == 'higher_better':
        improvement = final_score - baseline_score
    else:
        improvement = baseline_score - final_score

    improvement_pct = (improvement / baseline_score * 100) if baseline_score else 0

    md = f"""# KaggleResearch Summary

## Executive Summary

| Metric | Value |
|--------|-------|
| Baseline Score | {baseline_score:.6f} |
| Final Score | {final_score:.6f} |
| Improvement | {improvement:+.6f} ({improvement_pct:+.2f}%) |
| Total Experiments | {total} |
| Successful | {improved} ({improved/total*100:.1f}%) |
| No Improvement | {no_improvement} |
| Crashed | {crashed} |

"""

    # Show crashed experiments with errors
    crashed_exps = [e for e in experiments if e.get('status') == 'crashed']
    if crashed_exps:
        md += "### Crash Details\n\n"
        md += "| Idea | Error |\n"
        md += "|------|-------|\n"
        for exp in crashed_exps[-10:]:  # Show last 10 crashes
            idea = exp.get('idea_title', 'Unknown')[:30]
            error = exp.get('error_message', 'Unknown error')
            # Truncate and escape error for table
            error_short = error[:80].replace('|', '\\|').replace('\n', ' ')
            if len(error) > 80:
                error_short += '...'
            md += f"| {idea} | {error_short} |\n"
        md += "\n"

    md += """## Strategy Timeline



"""

    # Add strategy history
    branches = checkpoint.get('branches', {})
    if branches:
        md += "| Branch | Strategy | Best Score | Status |\n"
        md += "|--------|----------|------------|--------|\n"
        for branch_name, info in branches.items():
            md += f"| {branch_name} | {info.get('strategy_name', 'Unknown')} | "
            md += f"{info.get('best_score', 0):.6f} | {info.get('status', 'unknown')} |\n"

    # Improvement waterfall
    md += "\n## Improvement Waterfall\n\n"

    improved_exps = [e for e in experiments if e.get('status') == 'improved']
    improved_exps.sort(key=lambda x: (x.get('score', 0) - x.get('previous_best', 0)), reverse=True)

    if improved_exps:
        md += "| Rank | Idea | Score Delta | Final Score |\n"
        md += "|------|------|-------------|-------------|\n"
        for i, exp in enumerate(improved_exps[:10], 1):
            delta = exp.get('score', 0) - exp.get('previous_best', 0)
            md += f"| {i} | {exp.get('idea_title', 'Unknown')[:30]} | "
            md += f"{delta:+.6f} | {exp.get('score', 0):.6f} |\n"
    else:
        md += "*No successful experiments*\n"

    # Learning log
    if strategy_md_path and Path(strategy_md_path).exists():
        with open(strategy_md_path, 'r') as f:
            strategy_content = f.read()

        if '## Learning Log' in strategy_content:
            log_start = strategy_content.find('## Learning Log')
            learning_log = strategy_content[log_start:]
            md += f"\n{learning_log}\n"

    # Git log
    md += "\n## Winning Commits\n\n"

    winning_commits = [e for e in experiments if e.get('status') == 'improved']
    if winning_commits:
        md += "| Experiment | Idea | Score |\n"
        md += "|------------|------|-------|\n"
        for exp in winning_commits:
            md += f"| {exp.get('id', 'N/A')[:8]} | {exp.get('idea_title', 'Unknown')[:25]} | "
            md += f"{exp.get('score', 0):.6f} |\n"
    else:
        md += "*No winning commits*\n"

    # Exploration tree
    if tree_path and Path(tree_path).exists():
        md += "\n## Exploration Tree\n\n"
        md += render_idea_tree(tree_path, checkpoint.get('metric_direction', 'higher_better'))

    # Run info
    md += "\n## Run Information\n\n"
    md += f"- **Run ID**: {checkpoint.get('run_id', 'N/A')}\n"
    md += f"- **Exploration Mode**: {checkpoint.get('exploration_mode', 'df')}\n"
    md += f"- **Competition**: {checkpoint.get('competition_slug', 'N/A')}\n"
    md += f"- **Phase**: {checkpoint.get('phase', 'N/A')}\n"

    return md


def render_idea_tree(tree_path: Path, metric_direction: str = 'higher_better') -> str:
    """
    Render the idea exploration tree as markdown.

    Args:
        tree_path: Path to idea_tree.json
        metric_direction: "higher_better" or "lower_better"

    Returns:
        Markdown string with tree visualization
    """
    try:
        from .idea_tree import IdeaTree

        tree = IdeaTree(tree_path)
        if not tree.load():
            return "*No exploration tree found*\n"

        # Get tree stats
        stats = tree.count_by_status()
        md = f"**Nodes**: {len(tree.nodes)} | "
        md += f"**Max Depth**: {tree.get_max_depth()} | "
        md += f"**Improved**: {stats.get('improved', 0)} | "
        md += f"**Crashed**: {stats.get('crashed', 0)}\n\n"

        # Render ASCII tree
        md += "```\n"
        md += tree.render_tree(metric_direction)
        md += "\n```\n"

        # Best path
        best_path = tree.get_improved_path(metric_direction)
        if best_path:
            md += "\n**Best Path**: "
            md += " → ".join([f"{n.idea_title} ({n.score:.4f})" if n.score else n.idea_title for n in best_path])
            md += "\n"

        return md

    except Exception as e:
        return f"*Error loading tree: {e}*\n"


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def display_in_colab(html_or_md: str, is_html: bool = True) -> None:
    """
    Display content in Colab notebook.

    Args:
        html_or_md: Content to display
        is_html: True if HTML, False if Markdown
    """
    try:
        from IPython.display import display, HTML, Markdown

        if is_html:
            display(HTML(html_or_md))
        else:
            display(Markdown(html_or_md))
    except ImportError:
        # Not in notebook environment
        print(html_or_md)


def create_approval_button(callback) -> None:
    """
    Create an approval button widget for Colab.

    Args:
        callback: Function to call when button is clicked
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display

        button = widgets.Button(
            description='Approve Strategy & Start Loop',
            button_style='success',
            icon='check',
            layout=widgets.Layout(width='300px', height='50px')
        )

        output = widgets.Output()

        def on_click(b):
            with output:
                print("Strategy approved! Starting experiment loop...")
            callback()

        button.on_click(on_click)

        display(button, output)

    except ImportError:
        print("ipywidgets not available. Please run the next cell to start the loop.")


def format_time_remaining(
    experiments_done: int,
    total_ideas: int,
    avg_duration_seconds: float
) -> str:
    """
    Format estimated time remaining.

    Args:
        experiments_done: Number of experiments completed
        total_ideas: Total ideas to try
        avg_duration_seconds: Average experiment duration

    Returns:
        Formatted time string
    """
    remaining = total_ideas - experiments_done
    if remaining <= 0:
        return "Complete"

    total_seconds = remaining * avg_duration_seconds

    if total_seconds < 60:
        return f"~{int(total_seconds)}s"
    elif total_seconds < 3600:
        return f"~{int(total_seconds/60)}m"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"~{hours}h {minutes}m"
