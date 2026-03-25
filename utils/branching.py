"""
Git branch management for strategy pivots.
Preserves prior work in named branches.
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import shutil


def run_git_command(repo_path: Path, *args) -> Tuple[bool, str]:
    """
    Run a git command in the specified repository.

    Args:
        repo_path: Path to the repository
        *args: Git command arguments

    Returns:
        Tuple of (success, output)
    """
    try:
        result = subprocess.run(
            ['git'] + list(args),
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        success = result.returncode == 0
        output = result.stdout if success else result.stderr
        return success, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except FileNotFoundError:
        return False, "Git not found. Please install git."
    except Exception as e:
        return False, str(e)


def init_repo(repo_path: Path) -> bool:
    """
    Initialize a git repository if it doesn't exist.

    Args:
        repo_path: Path to the repository

    Returns:
        True if successful
    """
    repo_path = Path(repo_path)
    git_dir = repo_path / '.git'

    if git_dir.exists():
        return True

    success, _ = run_git_command(repo_path, 'init')
    if not success:
        return False

    # Initial commit
    success, _ = run_git_command(repo_path, 'add', '-A')
    if success:
        run_git_command(repo_path, 'commit', '-m', 'Initial commit: baseline')

    return True


def get_current_branch(repo_path: Path) -> str:
    """
    Get the current branch name.

    Args:
        repo_path: Path to the repository

    Returns:
        Branch name or "main" as default
    """
    success, output = run_git_command(repo_path, 'rev-parse', '--abbrev-ref', 'HEAD')
    return output if success else "main"


def get_current_commit(repo_path: Path) -> str:
    """
    Get the current commit SHA.

    Args:
        repo_path: Path to the repository

    Returns:
        Commit SHA or empty string if not a git repo
    """
    success, output = run_git_command(repo_path, 'rev-parse', 'HEAD')
    return output if success else ""


def git_checkout(repo_path: Path, ref: str) -> Tuple[bool, str]:
    """
    Checkout a specific commit or branch.

    Args:
        repo_path: Path to the repository
        ref: Commit SHA or branch name to checkout

    Returns:
        Tuple of (success, message)
    """
    if not ref:
        return False, "No ref specified"

    success, output = run_git_command(repo_path, 'checkout', ref)

    if success:
        return True, f"Checked out: {ref[:8] if len(ref) > 8 else ref}"

    return False, f"Failed to checkout {ref}: {output}"


def archive_current_branch(
    repo_path: Path,
    branch_name: str
) -> Tuple[bool, str]:
    """
    Archive the current state to a named branch.

    Args:
        repo_path: Path to the repository
        branch_name: Name for the archive branch

    Returns:
        Tuple of (success, message)
    """
    repo_path = Path(repo_path)

    # Ensure all changes are committed
    success, _ = run_git_command(repo_path, 'add', '-A')
    if success:
        run_git_command(repo_path, 'commit', '-m', f'Archive before pivot to {branch_name}')

    # Create the archive branch from current state
    success, output = run_git_command(repo_path, 'checkout', '-b', branch_name)

    if not success:
        # Branch might already exist, try switching
        success, output = run_git_command(repo_path, 'checkout', branch_name)

    if success:
        return True, f"Archived current state to branch: {branch_name}"

    return False, f"Failed to archive branch: {output}"


def start_new_branch(
    repo_path: Path,
    branch_name: str,
    baseline_train_py: str
) -> Tuple[bool, str]:
    """
    Start a new branch for a pivot, resetting train.py to baseline.

    Args:
        repo_path: Path to the repository
        branch_name: Name for the new branch
        baseline_train_py: Content of the baseline train.py

    Returns:
        Tuple of (success, message)
    """
    repo_path = Path(repo_path)

    # Create new branch
    success, output = run_git_command(repo_path, 'checkout', '-b', branch_name)

    if not success:
        # Branch might exist, just checkout
        success, output = run_git_command(repo_path, 'checkout', branch_name)
        if not success:
            return False, f"Failed to create/checkout branch: {output}"

    # Reset train.py to baseline
    train_path = repo_path / 'train.py'
    with open(train_path, 'w') as f:
        f.write(baseline_train_py)

    # Commit the reset
    success, _ = run_git_command(repo_path, 'add', 'train.py')
    if success:
        run_git_command(repo_path, 'commit', '-m', f'Reset train.py to baseline for new strategy')

    return True, f"Started new branch: {branch_name}"


def compare_branches(
    old_best_score: float,
    new_best_score: float,
    metric_direction: str = "higher_better"
) -> Tuple[str, str]:
    """
    Compare two branches and determine the winner.

    Args:
        old_best_score: Best score from old branch
        new_best_score: Best score from new branch
        metric_direction: "higher_better" or "lower_better"

    Returns:
        Tuple of (winner, loser) - "old" or "new"
    """
    if metric_direction == "higher_better":
        if new_best_score > old_best_score:
            return "new", "old"
        else:
            return "old", "new"
    else:  # lower_better
        if new_best_score < old_best_score:
            return "new", "old"
        else:
            return "old", "new"


def promote_branch(
    repo_path: Path,
    winner_branch: str,
    main_branch: str = "main"
) -> Tuple[bool, str]:
    """
    Promote the winning branch to main.

    Args:
        repo_path: Path to the repository
        winner_branch: Name of the winning branch
        main_branch: Name of the main branch

    Returns:
        Tuple of (success, message)
    """
    repo_path = Path(repo_path)

    # Checkout main
    success, _ = run_git_command(repo_path, 'checkout', main_branch)

    if not success:
        # Main might not exist, create it
        success, output = run_git_command(repo_path, 'checkout', '-b', main_branch)
        if not success:
            return False, f"Failed to checkout main: {output}"

    # Merge winner into main
    success, output = run_git_command(repo_path, 'merge', winner_branch, '-m',
                                       f'Merge winning branch {winner_branch}')

    if success:
        return True, f"Promoted {winner_branch} to {main_branch}"

    # Try reset --hard as fallback
    success, output = run_git_command(repo_path, 'reset', '--hard', winner_branch)
    if success:
        return True, f"Reset {main_branch} to {winner_branch}"

    return False, f"Failed to promote branch: {output}"


def switch_to_branch(repo_path: Path, branch_name: str) -> Tuple[bool, str]:
    """
    Switch to a specific branch.

    Args:
        repo_path: Path to the repository
        branch_name: Branch to switch to

    Returns:
        Tuple of (success, message)
    """
    success, output = run_git_command(repo_path, 'checkout', branch_name)

    if success:
        return True, f"Switched to branch: {branch_name}"

    return False, f"Failed to switch branch: {output}"


def list_branches(repo_path: Path) -> Dict[str, bool]:
    """
    List all branches and indicate which is current.

    Args:
        repo_path: Path to the repository

    Returns:
        Dict mapping branch names to whether they're current
    """
    success, output = run_git_command(repo_path, 'branch')

    if not success:
        return {}

    branches = {}
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('* '):
            branches[line[2:]] = True
        elif line:
            branches[line] = False

    return branches


def get_branch_commits(repo_path: Path, branch: str, n: int = 10) -> list:
    """
    Get recent commits from a branch.

    Args:
        repo_path: Path to the repository
        branch: Branch name
        n: Number of commits to retrieve

    Returns:
        List of commit info dicts
    """
    success, output = run_git_command(
        repo_path, 'log', branch,
        f'-{n}', '--pretty=format:%h|%s|%ai'
    )

    if not success:
        return []

    commits = []
    for line in output.split('\n'):
        if '|' in line:
            parts = line.split('|')
            commits.append({
                'hash': parts[0],
                'message': parts[1] if len(parts) > 1 else '',
                'date': parts[2] if len(parts) > 2 else '',
            })

    return commits


def generate_branch_name(strategy_name: str, version: int) -> str:
    """
    Generate a branch name from strategy name.

    Args:
        strategy_name: Name of the strategy
        version: Version number

    Returns:
        Valid git branch name
    """
    import re

    # Clean the strategy name
    clean_name = re.sub(r'[^a-zA-Z0-9\-]', '-', strategy_name.lower())
    clean_name = re.sub(r'-+', '-', clean_name).strip('-')

    # Truncate if too long
    if len(clean_name) > 30:
        clean_name = clean_name[:30].rstrip('-')

    return f"branch/strategy-v{version}-{clean_name}"


def get_next_branch_version(repo_path: Path) -> int:
    """
    Get the next available branch version number.

    Args:
        repo_path: Path to the repository

    Returns:
        Next version number
    """
    branches = list_branches(repo_path)

    max_version = 0
    for branch in branches:
        if branch.startswith('branch/strategy-v'):
            try:
                # Extract version number
                import re
                match = re.search(r'strategy-v(\d+)', branch)
                if match:
                    version = int(match.group(1))
                    max_version = max(max_version, version)
            except ValueError:
                pass

    return max_version + 1
