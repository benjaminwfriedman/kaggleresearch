"""
Experiment runner for the autoresearch loop.
Handles idea implementation, execution, and logging.
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from .strategy import Idea
from .branching import run_git_command


@dataclass
class ExperimentLog:
    """Log entry for a single experiment."""
    id: str
    idea_title: str
    idea_index: int
    branch: str
    status: str  # "running", "improved", "no_improvement", "crashed"
    score: Optional[float]
    previous_best: float
    duration_seconds: float
    timestamp: str
    train_py_hash: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def implement_idea(
    idea: Idea,
    train_py_path: Path,
    strategy_md_path: Path,
    client: Any  # Anthropic client
) -> str:
    """
    Use LLM to implement an idea by modifying train.py.

    Args:
        idea: The idea to implement
        train_py_path: Path to current train.py
        strategy_md_path: Path to STRATEGY.md
        client: Anthropic API client

    Returns:
        Modified train.py content
    """
    # Load current files
    with open(train_py_path, 'r') as f:
        current_train_py = f.read()

    with open(strategy_md_path, 'r') as f:
        strategy_md = f.read()

    # Load code agent prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "code_agent.md"
    with open(prompt_path, 'r') as f:
        system_prompt = f.read()

    # Format the idea
    idea_text = f"""## IDEA: {idea.title}
Source: {idea.source}
Risk: {idea.risk}
Estimated gain: {idea.estimated_gain}
---
Hypothesis: {idea.hypothesis}
Implementation: {idea.implementation}
Validation: {idea.validation}
==="""

    user_prompt = f"""Here is the current train.py:

```python
{current_train_py}
```

Here is the current STRATEGY.md:

{strategy_md}

Here is the IDEA to implement:

{idea_text}

Return ONLY the complete modified train.py file. No explanations, no markdown fences, just the Python code."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    response = message.content[0].text

    # Clean up response - remove markdown fences if present
    if response.startswith('```python'):
        response = response[9:]
    if response.startswith('```'):
        response = response[3:]
    if response.endswith('```'):
        response = response[:-3]

    return response.strip()


def validate_patch(
    repo_path: Path,
    patched_train_py: str,
    original_metric_py_hash: Optional[str] = None,
    original_train_py: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate that a patch doesn't violate constraints.

    Args:
        repo_path: Path to the repo
        patched_train_py: New train.py content
        original_metric_py_hash: Hash of original metric.py
        original_train_py: Original train.py content for comparison

    Returns:
        Tuple of (is_valid, error_message)
    """
    repo_path = Path(repo_path)

    # Check for truncated response (common LLM failure mode)
    if len(patched_train_py) < 500:
        return False, "Generated code is too short - likely truncated"

    # Check that response contains essential parts
    required_patterns = ['def main(', 'if __name__']
    for pattern in required_patterns:
        if pattern not in patched_train_py:
            return False, f"Missing required pattern: {pattern}"

    # Check for incomplete lines (truncation indicator)
    lines = patched_train_py.split('\n')
    for i, line in enumerate(lines[-5:]):  # Check last 5 lines
        stripped = line.rstrip()
        if stripped and not stripped.endswith((':',')',')',']','}','"',"'",'#','\\',',')):
            # Line doesn't end with a valid Python line ending
            if stripped.endswith('.') or stripped.endswith('str.e'):
                return False, f"Code appears truncated at line {len(lines)-5+i}"

    # Check syntax by compiling
    try:
        compile(patched_train_py, '<train.py>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error in train.py: {e}"

    # If we have original, check it's not drastically shorter
    if original_train_py and len(patched_train_py) < len(original_train_py) * 0.5:
        return False, "Generated code is less than 50% of original length"

    # Check that metric.py wasn't modified
    metric_path = repo_path / 'metric.py'
    if metric_path.exists() and original_metric_py_hash:
        with open(metric_path, 'r') as f:
            current_hash = hashlib.md5(f.read().encode()).hexdigest()
        if current_hash != original_metric_py_hash:
            return False, "metric.py was modified — this is forbidden"

    # Check for forbidden imports (packages requiring pip install)
    forbidden_imports = [
        'catboost',  # Not pre-installed
        'optuna',    # Not pre-installed
        'ray',       # Not pre-installed
    ]

    for forbidden in forbidden_imports:
        if f'import {forbidden}' in patched_train_py or f'from {forbidden}' in patched_train_py:
            return False, f"Forbidden import: {forbidden} requires pip install"

    return True, ""


def backup_train_py(repo_path: Path) -> str:
    """Backup current train.py and return its content."""
    train_path = Path(repo_path) / 'train.py'
    backup_path = Path(repo_path) / 'train.py.backup'

    with open(train_path, 'r') as f:
        content = f.read()

    with open(backup_path, 'w') as f:
        f.write(content)

    return content


def restore_train_py(repo_path: Path) -> bool:
    """Restore train.py from backup."""
    train_path = Path(repo_path) / 'train.py'
    backup_path = Path(repo_path) / 'train.py.backup'

    if backup_path.exists():
        with open(backup_path, 'r') as f:
            content = f.read()
        with open(train_path, 'w') as f:
            f.write(content)
        return True
    return False


def run_experiment(
    repo_path: Path,
    time_budget_min: float,
    python_executable: str = "python"
) -> Tuple[Optional[float], Optional[str]]:
    """
    Run train.py and return the score.

    Args:
        repo_path: Path to the repo containing train.py
        time_budget_min: Time budget in minutes
        python_executable: Python executable to use

    Returns:
        Tuple of (score or None if crashed, error message if any)
    """
    repo_path = Path(repo_path)
    train_path = repo_path / 'train.py'

    timeout_seconds = int(time_budget_min * 60)

    try:
        result = subprocess.run(
            [python_executable, str(train_path)],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode != 0:
            return None, f"Train.py failed:\n{result.stderr[-1000:]}"

        # Extract score from output
        # Look for patterns like "Final CV Score: 0.8534" or "CV Mean Score: 0.8534"
        import re
        output = result.stdout

        score_patterns = [
            r'Final CV (?:Score|RMSE|Dice|AUC):\s*([\d.]+)',
            r'CV Mean (?:Score|RMSE|Dice|AUC):\s*([\d.]+)',
            r'Best Score:\s*([\d.]+)',
            r'Score:\s*([\d.]+)',
        ]

        for pattern in score_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return score, None
                except ValueError:
                    continue

        return None, "Could not extract score from output"

    except subprocess.TimeoutExpired:
        return None, f"Experiment timed out after {time_budget_min} minutes"
    except Exception as e:
        return None, f"Experiment error: {str(e)}"


def git_commit(repo_path: Path, message: str) -> bool:
    """
    Commit current changes with the given message.

    Args:
        repo_path: Path to the repo
        message: Commit message

    Returns:
        True if successful
    """
    success, _ = run_git_command(repo_path, 'add', '-A')
    if not success:
        return False

    success, _ = run_git_command(repo_path, 'commit', '-m', message)
    return success


def git_reset_hard(repo_path: Path) -> bool:
    """
    Reset the repo to the last commit, discarding changes.

    Args:
        repo_path: Path to the repo

    Returns:
        True if successful
    """
    success, _ = run_git_command(repo_path, 'reset', '--hard', 'HEAD')
    return success


def log_experiment(
    db_path: Path,
    experiment: ExperimentLog
) -> None:
    """
    Log an experiment to the SQLite database.

    Args:
        db_path: Path to experiment_log.db
        experiment: ExperimentLog to save
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            idea_title TEXT,
            idea_index INTEGER,
            branch TEXT,
            status TEXT,
            score REAL,
            previous_best REAL,
            duration_seconds REAL,
            timestamp TEXT,
            train_py_hash TEXT,
            error_message TEXT
        )
    ''')

    # Insert experiment
    cursor.execute('''
        INSERT OR REPLACE INTO experiments
        (id, idea_title, idea_index, branch, status, score, previous_best,
         duration_seconds, timestamp, train_py_hash, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        experiment.id,
        experiment.idea_title,
        experiment.idea_index,
        experiment.branch,
        experiment.status,
        experiment.score,
        experiment.previous_best,
        experiment.duration_seconds,
        experiment.timestamp,
        experiment.train_py_hash,
        experiment.error_message,
    ))

    conn.commit()
    conn.close()


def load_experiments(db_path: Path, branch: Optional[str] = None) -> list:
    """
    Load experiments from database.

    Args:
        db_path: Path to experiment_log.db
        branch: Optional branch filter

    Returns:
        List of experiment dicts
    """
    db_path = Path(db_path)

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    if branch:
        cursor.execute(
            'SELECT * FROM experiments WHERE branch = ? ORDER BY timestamp',
            (branch,)
        )
    else:
        cursor.execute('SELECT * FROM experiments ORDER BY timestamp')

    columns = [desc[0] for desc in cursor.description]
    experiments = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()
    return experiments


def session_timeout_imminent(
    session_start: datetime,
    max_session_hours: float = 12.0,
    buffer_minutes: float = 10.0
) -> bool:
    """
    Check if the Colab session timeout is approaching.

    Args:
        session_start: When the session started
        max_session_hours: Maximum session length (Colab default is 12h)
        buffer_minutes: Buffer time before timeout

    Returns:
        True if timeout is imminent
    """
    elapsed = datetime.now() - session_start
    elapsed_hours = elapsed.total_seconds() / 3600

    timeout_threshold = max_session_hours - (buffer_minutes / 60)

    return elapsed_hours >= timeout_threshold


def generate_experiment_id(idea_title: str, timestamp: str) -> str:
    """
    Generate a unique experiment ID.

    Args:
        idea_title: Title of the idea
        timestamp: Timestamp string

    Returns:
        Unique ID string
    """
    content = f"{idea_title}:{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def get_file_hash(path: Path) -> str:
    """
    Get MD5 hash of a file's contents.

    Args:
        path: Path to file

    Returns:
        MD5 hash string
    """
    with open(path, 'r') as f:
        return hashlib.md5(f.read().encode()).hexdigest()


def scale_time_budget(base_budget: float, gpu_name: str) -> float:
    """
    Scale time budget based on GPU type.

    Args:
        base_budget: Base time budget in minutes
        gpu_name: Name of the GPU

    Returns:
        Scaled time budget
    """
    gpu_lower = gpu_name.lower()

    if 'a100' in gpu_lower:
        return base_budget  # A100 is the baseline
    elif 't4' in gpu_lower:
        return base_budget * 1.5
    elif 'v100' in gpu_lower:
        return base_budget * 1.2
    elif 'p100' in gpu_lower:
        return base_budget * 2.0
    else:
        # CPU or unknown
        return base_budget * 3.0


def detect_gpu() -> Tuple[str, bool]:
    """
    Detect available GPU.

    Returns:
        Tuple of (gpu_name, has_gpu)
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), True
    except ImportError:
        pass

    return "CPU", False
