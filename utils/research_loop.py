"""
Main research loop for KaggleResearch.

This module runs the core experiment cycle:
1. Select next node to explore (UCB1 or sequential)
2. Generate/implement experiment
3. Run and evaluate
4. Update tree, backtrack if needed
5. Re-research when exhausted
"""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

from .checkpoint import (
    CheckpointState, load_checkpoint, save_checkpoint,
    update_checkpoint_after_experiment
)
from .strategy import (
    parse_ideas_md, update_idea_status, append_learning_log,
    get_next_pending_idea, format_learning_log_entry
)
from .plateau import plateau_triggered, summarise_failures, ExperimentResult
from .experiment_runner import (
    implement_idea, validate_patch, run_experiment,
    git_commit, log_experiment, load_experiments,
    session_timeout_imminent, generate_experiment_id, get_file_hash,
    ExperimentLog, backup_train_py, restore_train_py
)
from .branching import get_current_branch, git_checkout, get_current_commit
from .idea_tree import IdeaTree


@dataclass
class ResearchConfig:
    """Configuration for the research loop."""
    # Paths
    project_dir: Path
    repo_dir: Path
    checkpoint_path: Path
    strategy_path: Path
    ideas_path: Path
    db_path: Path
    tree_path: Path
    literature_dir: Path
    kaggleresearch_path: Path

    # Timing
    time_budget_min: float
    plateau_window: int = 5
    plateau_min_gain_pct: float = 1.0

    # Tree navigation
    exploration_mode: str = "df"  # "df" or "bf"
    bf_sibling_count: int = 5

    # Literature
    literature_depth: int = 10

    # Metric
    metric_direction: str = "higher_better"


@dataclass
class ResearchResult:
    """Result of running the research loop."""
    final_score: float
    baseline_score: float
    total_experiments: int
    successful_experiments: int
    exit_reason: str  # "timeout", "halted", "interrupted", "ideas_exhausted"


def initialize_tree(
    tree_path: Path,
    repo_dir: Path,
    checkpoint: CheckpointState
) -> IdeaTree:
    """Initialize or load the idea tree."""
    tree = IdeaTree(tree_path)

    if not tree.load():
        print("Initializing new idea tree...")

    # Create root if needed
    if not tree.root_id:
        current_commit = get_current_commit(repo_dir)
        root = tree.add_node("Baseline", current_commit, None)
        root.status = "improved"
        root.score = checkpoint.baseline_score
        tree.root_id = root.id
        tree.competition_slug = checkpoint.competition_slug
        tree.run_id = checkpoint.run_id
        tree.save()

    return tree


def recover_from_interruption(
    tree: IdeaTree,
    checkpoint: CheckpointState,
    repo_dir: Path
) -> None:
    """Recover from a mid-experiment interruption."""
    current_node = tree.get_node(checkpoint.current_node_id) if checkpoint.current_node_id else None

    if current_node and current_node.status == 'running':
        print(f"Detected interrupted experiment: {current_node.idea_title}")
        print(f"  Restoring to parent state...")

        if (repo_dir / 'train.py.backup').exists():
            restore_train_py(repo_dir)
        elif current_node.parent_commit:
            git_checkout(repo_dir, current_node.parent_commit)

        tree.update_status(current_node.id, 'crashed', None, "Session interrupted")
        tree.save()


def run_reresearch(
    config: ResearchConfig,
    checkpoint: CheckpointState,
    tree: IdeaTree,
    client: Any
) -> Tuple[str, CheckpointState]:
    """
    Run re-research when ideas are exhausted or plateau detected.

    Returns:
        Tuple of (action, checkpoint) where action is "continue", "pivot", or "halt"
    """
    from .literature import load_cached_papers, format_papers_for_prompt
    from .strategy import generate_ideas_md, archive_strategy
    from .kaggle_api import get_template_for_problem_type
    from .branching import archive_current_branch, start_new_branch, get_next_branch_version
    from .checkpoint import update_checkpoint_for_branch
    from .reresearch import (
        reresearch_new_angle, reresearch_reread,
        handle_reresearch_result, should_attempt_reresearch
    )

    print("\n" + "=" * 50)
    print("=== Re-research Phase ===")
    print("=" * 50)

    # Get failure summary
    experiments = load_experiments(config.db_path)
    exp_results = [
        ExperimentResult(
            idea_title=e['idea_title'],
            score=e['score'],
            status=e['status'],
            idea_index=e['idea_index']
        ) for e in experiments
    ]
    failure_summary = summarise_failures(exp_results)

    checkpoint_dict = checkpoint.to_dict()
    comp_meta = checkpoint_dict.get('competition_meta', {})

    with open(config.strategy_path, 'r') as f:
        strategy_md = f.read()

    # Attempt 1: New angle
    print("\n  Attempt 1: Searching for new angles...")
    result = reresearch_new_angle(
        strategy_md=strategy_md,
        failure_summary=failure_summary,
        search_history_path=config.literature_dir / 'search_history.json',
        literature_cache_path=config.literature_dir / 'papers.json',
        problem_type=checkpoint.problem_type,
        metric=comp_meta.get('metric', 'accuracy'),
        literature_depth=config.literature_depth,
        client=client
    )

    print(f"  Result: {result.outcome}")

    if result.outcome == 'new_ideas':
        action = handle_reresearch_result(
            result, config.ideas_path, config.strategy_path,
            config.literature_dir / 'archived_strategies.md'
        )
        print(f"  {action['message']}")

        # Attach new ideas to best node in tree
        best_node = tree.get_best_node(config.metric_direction)
        if best_node:
            print(f"  Attaching new ideas to best node: {best_node.idea_title}")
            git_checkout(config.repo_dir, best_node.parent_commit)
            tree.set_current_node(best_node.id)

        checkpoint.phase = 'research'
        checkpoint.reresearch_attempts = 0
        checkpoint.plateau_window_scores = []
        return ("continue", checkpoint)

    elif result.outcome == 'pivot':
        print("\n  === Pivot: Creating new strategy branch ===")

        version = get_next_branch_version(config.repo_dir)
        new_branch = tree.generate_branch_name(
            tree.root_id if tree.root_id else "root",
            checkpoint.competition_slug,
            checkpoint.run_id
        )
        old_branch = checkpoint.current_branch

        archive_current_branch(config.repo_dir, old_branch)
        archive_strategy(config.strategy_path, config.literature_dir / 'archived_strategies.md')

        template_name = get_template_for_problem_type(checkpoint.problem_type)
        template_path = config.kaggleresearch_path / 'templates' / template_name
        with open(template_path, 'r') as f:
            baseline_train = f.read()

        start_new_branch(config.repo_dir, new_branch, baseline_train)

        with open(config.strategy_path, 'w') as f:
            f.write(result.pivot_strategy_md)

        papers = load_cached_papers(config.literature_dir / 'papers.json')
        papers_summary = format_papers_for_prompt(papers)

        new_ideas = generate_ideas_md(
            papers_summary=papers_summary,
            strategy_md=result.pivot_strategy_md,
            competition_meta={'name': comp_meta.get('name'), 'problem_type': checkpoint.problem_type},
            baseline_score=checkpoint.baseline_score,
            client=client
        )
        with open(config.ideas_path, 'w') as f:
            f.write(new_ideas)

        checkpoint = update_checkpoint_for_branch(
            checkpoint, new_branch, result.pivot_strategy_name
        )
        checkpoint.phase = 'research'
        checkpoint.plateau_window_scores = []

        current_commit = get_current_commit(config.repo_dir)
        pivot_node = tree.add_node(
            f"PIVOT: {result.pivot_strategy_name}",
            current_commit,
            tree.root_id,
            branch_name=new_branch
        )
        pivot_node.status = 'improved'
        pivot_node.score = checkpoint.baseline_score
        tree.set_current_node(pivot_node.id)

        print(f"  Created new branch: {new_branch}")
        print(f"  New strategy: {result.pivot_strategy_name}")

        return ("pivot", checkpoint)

    # no_new_ideas - try Attempt 2
    if should_attempt_reresearch(checkpoint.reresearch_attempts + 1, max_attempts=2):
        print("\n  Attempt 2: Re-reading papers with failure context...")
        checkpoint.reresearch_attempts += 1

        result2 = reresearch_reread(
            config.literature_dir / 'papers.json',
            strategy_md,
            failure_summary,
            client
        )

        if result2.outcome == 'new_ideas':
            action = handle_reresearch_result(
                result2, config.ideas_path, config.strategy_path,
                config.literature_dir / 'archived_strategies.md'
            )
            print(f"  {action['message']}")

            best_node = tree.get_best_node(config.metric_direction)
            if best_node:
                git_checkout(config.repo_dir, best_node.parent_commit)
                tree.set_current_node(best_node.id)

            checkpoint.phase = 'research'
            checkpoint.plateau_window_scores = []
            return ("continue", checkpoint)

    print("\n  Re-research exhausted. No new ideas found.")
    checkpoint.phase = 'halted'
    return ("halt", checkpoint)


def run_single_experiment(
    config: ResearchConfig,
    checkpoint: CheckpointState,
    tree: IdeaTree,
    current_node,
    next_idea,
    client: Any
) -> Tuple[str, Optional[float], CheckpointState]:
    """
    Run a single experiment.

    Returns:
        Tuple of (status, score, updated_checkpoint)
    """
    # Create tree node for this idea
    current_commit = get_current_commit(config.repo_dir)
    new_node = tree.add_node(
        next_idea.title,
        current_commit,
        current_node.id,
        branch_name=checkpoint.current_branch
    )

    # Mark as running BEFORE experiment (for warm restart detection)
    tree.update_status(new_node.id, 'running', None)
    tree.save()
    checkpoint.current_node_id = new_node.id
    save_checkpoint(config.checkpoint_path, checkpoint)

    update_idea_status(config.ideas_path, next_idea.title, 'running')

    print(f"\n--- Experiment {checkpoint.total_experiments + 1}: {next_idea.title} ---")
    print(f"    Tree depth: {new_node.depth}, Node ID: {new_node.id}")
    start_time = time.time()

    # Backup current train.py
    original_train_py = backup_train_py(config.repo_dir)

    # Get metric.py hash for validation
    metric_path = config.repo_dir / 'metric.py'
    metric_hash = get_file_hash(metric_path) if metric_path.exists() else None

    # Implement the idea
    try:
        new_train_py = implement_idea(
            idea=next_idea,
            train_py_path=config.repo_dir / 'train.py',
            strategy_md_path=config.strategy_path,
            client=client
        )
    except Exception as e:
        print(f"  Implementation error: {e}")
        restore_train_py(config.repo_dir)
        tree.update_status(new_node.id, 'crashed', None, str(e))
        update_idea_status(config.ideas_path, next_idea.title, 'crashed')
        tree.save()
        # Reset to parent node so next idea is a sibling
        tree.set_current_node(current_node.id)
        checkpoint.current_node_id = current_node.id
        return ('crashed', None, checkpoint)

    # Validate patch
    is_valid, error = validate_patch(config.repo_dir, new_train_py, metric_hash, original_train_py)
    if not is_valid:
        print(f"  Validation failed: {error}")
        restore_train_py(config.repo_dir)
        tree.update_status(new_node.id, 'crashed', None, error)
        update_idea_status(config.ideas_path, next_idea.title, 'crashed')
        tree.save()
        # Reset to parent node
        tree.set_current_node(current_node.id)
        checkpoint.current_node_id = current_node.id
        return ('crashed', None, checkpoint)

    # Write new train.py
    with open(config.repo_dir / 'train.py', 'w') as f:
        f.write(new_train_py)

    # Run experiment
    score, error = run_experiment(config.repo_dir, config.time_budget_min)
    duration = time.time() - start_time

    # Determine outcome
    current_branch = get_current_branch(config.repo_dir)

    if score is None:
        # Crashed
        print(f"  CRASHED: {error[:100] if error else 'Unknown error'}")
        restore_train_py(config.repo_dir)
        status = 'crashed'
        tree.update_status(new_node.id, 'crashed', None, error)
        update_idea_status(config.ideas_path, next_idea.title, 'crashed')
        # Reset to parent node
        tree.set_current_node(current_node.id)
        checkpoint.current_node_id = current_node.id

    elif _is_improvement(score, checkpoint.best_score, config.metric_direction):
        # Improved!
        delta = score - checkpoint.best_score
        print(f"  IMPROVED: {checkpoint.best_score:.6f} -> {score:.6f} ({delta:+.6f})")

        branch_name = tree.generate_branch_name(
            new_node.id, checkpoint.competition_slug, checkpoint.run_id
        )
        git_commit(config.repo_dir, f"IMPROVE [{branch_name}]: {next_idea.title} | {checkpoint.best_score:.4f} -> {score:.4f}")

        tree.update_status(new_node.id, 'improved', score)
        update_idea_status(config.ideas_path, next_idea.title, 'improved')

        log_entry = format_learning_log_entry(
            checkpoint.total_experiments + 1,
            next_idea.title,
            'improved',
            score,
            delta
        )
        append_learning_log(config.strategy_path, log_entry)

        status = 'improved'
        checkpoint.best_score = score

        # Update UCB1 rewards
        tree.update_node_reward(new_node.id, score, checkpoint.baseline_score, config.metric_direction)

        # In DF mode, continue depth-first from this new node
        if checkpoint.exploration_mode == 'df':
            tree.set_current_node(new_node.id)
            checkpoint.current_node_id = new_node.id

    else:
        # No improvement
        print(f"  NO IMPROVEMENT: {score:.6f} (best: {checkpoint.best_score:.6f})")
        restore_train_py(config.repo_dir)
        status = 'no_improvement'
        tree.update_status(new_node.id, 'no_improvement', score)
        update_idea_status(config.ideas_path, next_idea.title, 'no_improvement')

        # Update UCB1 rewards even for no improvement
        tree.update_node_reward(new_node.id, score, checkpoint.baseline_score, config.metric_direction)

        # KEY FIX: Reset to parent node so next idea is a sibling, not a child
        tree.set_current_node(current_node.id)
        checkpoint.current_node_id = current_node.id

    # Log experiment to DB
    exp_log = ExperimentLog(
        id=generate_experiment_id(next_idea.title, datetime.now().isoformat()),
        idea_title=next_idea.title,
        idea_index=next_idea.index,
        branch=current_branch,
        status=status,
        score=score,
        previous_best=checkpoint.best_score if status != 'improved' else checkpoint.best_score - (score - checkpoint.best_score) if score else checkpoint.best_score,
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        train_py_hash=get_file_hash(config.repo_dir / 'train.py'),
        error_message=error,
    )
    log_experiment(config.db_path, exp_log)

    # Update checkpoint
    checkpoint = update_checkpoint_after_experiment(
        checkpoint, score, status == 'improved', next_idea.index
    )

    tree.save()
    save_checkpoint(config.checkpoint_path, checkpoint)

    return (status, score, checkpoint)


def _is_improvement(score: float, best_score: float, metric_direction: str) -> bool:
    """Check if score is an improvement over best_score."""
    if metric_direction == 'higher_better':
        return score > best_score
    else:
        return score < best_score


def run_research(
    config: ResearchConfig,
    client: Any,
    display_callback=None
) -> ResearchResult:
    """
    Run the main research loop.

    Args:
        config: Research configuration
        client: Anthropic API client
        display_callback: Optional callback(experiments, checkpoint) for UI updates

    Returns:
        ResearchResult with final state
    """
    # Load checkpoint
    checkpoint = load_checkpoint(config.checkpoint_path)
    if checkpoint is None:
        raise ValueError("No checkpoint found. Run bootstrap phase first.")

    # Initialize tree
    tree = initialize_tree(config.tree_path, config.repo_dir, checkpoint)

    # Recover from any interruption
    recover_from_interruption(tree, checkpoint, config.repo_dir)

    session_start = datetime.now()
    exit_reason = "completed"

    print("=== Research Loop ===")
    print(f"Run ID: {checkpoint.run_id}")
    print(f"Exploration mode: {checkpoint.exploration_mode}")
    print(f"Starting from: {checkpoint.best_score:.6f}")
    print(f"Baseline: {checkpoint.baseline_score:.6f}")
    print(f"Tree nodes: {len(tree.nodes)}, Max depth: {tree.get_max_depth()}")
    print("\nRunning experiments...\n")

    try:
        while True:
            # Check for timeout
            if session_timeout_imminent(session_start):
                print("\nSession timeout approaching. Saving state...")
                exit_reason = "timeout"
                break

            # Get current position in tree
            current_node = tree.get_current_node()
            if not current_node:
                current_node = tree.get_node(tree.root_id)
                tree.set_current_node(tree.root_id)

            # Check for plateau
            if plateau_triggered(
                checkpoint.plateau_window_scores,
                config.plateau_window,
                config.plateau_min_gain_pct
            ):
                print(f"\nPlateau detected on branch: {current_node.idea_title}")
                tree.update_status(current_node.id, "plateau", current_node.score)

                backtrack_target = tree.get_backtrack_target(current_node.id)

                if backtrack_target is None:
                    print("  No untried branches remaining. Running re-research...")
                    action, checkpoint = run_reresearch(config, checkpoint, tree, client)
                    tree.save()
                    save_checkpoint(config.checkpoint_path, checkpoint)

                    if action == "halt":
                        exit_reason = "halted"
                        break
                    continue

                print(f"  Backtracking to: {backtrack_target.idea_title}")
                git_checkout(config.repo_dir, backtrack_target.parent_commit)
                tree.set_current_node(backtrack_target.id)
                checkpoint.plateau_window_scores = []
                checkpoint.current_node_id = backtrack_target.id
                save_checkpoint(config.checkpoint_path, checkpoint)
                tree.save()
                continue

            # Load ideas and get next pending
            ideas = parse_ideas_md(config.ideas_path)
            next_idea = get_next_pending_idea(ideas)

            if next_idea is None:
                # Current level exhausted
                if checkpoint.exploration_mode == 'bf':
                    if tree.all_siblings_tested(current_node.id):
                        best_sibling = tree.get_best_sibling(current_node.id, config.metric_direction)
                        if best_sibling and best_sibling.id != current_node.id:
                            print(f"\n  BF mode: Best sibling is {best_sibling.idea_title}")
                            git_checkout(config.repo_dir, best_sibling.parent_commit)
                            tree.set_current_node(best_sibling.id)
                            checkpoint.current_node_id = best_sibling.id
                            save_checkpoint(config.checkpoint_path, checkpoint)
                            tree.save()

                pending_siblings = tree.get_pending_siblings(current_node.id)
                if pending_siblings:
                    continue

                print("\nAll current ideas exhausted. Running re-research...")
                action, checkpoint = run_reresearch(config, checkpoint, tree, client)
                tree.save()
                save_checkpoint(config.checkpoint_path, checkpoint)

                if action == "halt":
                    exit_reason = "halted"
                    break
                continue

            # Run single experiment
            status, score, checkpoint = run_single_experiment(
                config, checkpoint, tree, current_node, next_idea, client
            )

            # UI callback
            if display_callback:
                experiments = load_experiments(config.db_path)
                display_callback(experiments, checkpoint, tree)

    except KeyboardInterrupt:
        print("\n\nLoop interrupted by user.")
        exit_reason = "interrupted"

    finally:
        tree.save()
        save_checkpoint(config.checkpoint_path, checkpoint)

    print(f"\n=== Research Loop Complete ===")
    print(f"Final best score: {checkpoint.best_score:.6f}")
    print(f"Improvement from baseline: {checkpoint.best_score - checkpoint.baseline_score:+.6f}")
    print(f"Tree nodes explored: {len(tree.nodes)}")
    print(f"Exit reason: {exit_reason}")
    print(f"\nExploration Tree:")
    print(tree.render_tree(config.metric_direction))

    return ResearchResult(
        final_score=checkpoint.best_score,
        baseline_score=checkpoint.baseline_score,
        total_experiments=checkpoint.total_experiments,
        successful_experiments=checkpoint.successful_experiments,
        exit_reason=exit_reason,
    )
