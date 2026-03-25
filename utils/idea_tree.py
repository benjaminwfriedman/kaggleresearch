"""
Idea Tree - Tree-based navigation for explore/exploit loop.

Supports depth-first (DF) and breadth-first (BF) exploration strategies.
Each node tracks the git commit SHA for backtracking.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class IdeaNode:
    """
    A node in the idea exploration tree.

    Tree Structure:
    - Child: A new experiment that fills in an open dimension left by parent
    - Sibling: An alternative answer to the same question (differs on one dimension)

    The tree uses UCB1 for node selection and tracks dimensions to enable
    intelligent backtracking and exploration.
    """
    id: str
    idea_title: str
    parent_commit: str  # Git commit SHA to restore on backtrack
    parent_node_id: Optional[str]  # ID of parent node (None for root)
    children: List[str] = field(default_factory=list)  # IDs of child nodes
    siblings: List[str] = field(default_factory=list)  # IDs of sibling nodes (same parent)
    status: str = "pending"  # pending, running, improved, no_improvement, crashed, plateau
    score: Optional[float] = None
    depth: int = 0
    branch_name: str = ""
    timestamp: str = ""
    error_message: Optional[str] = None

    # New fields for tree-based exploration (F1 from spec)
    hypothesis: str = ""  # One-sentence hypothesis for this experiment
    dimension_answered: str = ""  # Which dimension this node answers (e.g., "feature_engineering")
    value_chosen: str = ""  # The specific value chosen for that dimension
    fixed_context: Dict[str, str] = field(default_factory=dict)  # Inherited from ancestors {dim: value}
    open_dimensions: List[str] = field(default_factory=list)  # Dimensions still to explore

    # UCB1 tracking
    visits: int = 0  # Number of times this node was selected/expanded
    total_reward: float = 0.0  # Sum of rewards from this subtree

    # Relationship type
    relationship: str = "child"  # "child" or "sibling" - how this relates to parent

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdeaNode':
        # Handle missing fields for backward compatibility
        defaults = {
            'siblings': [],
            'hypothesis': '',
            'dimension_answered': '',
            'value_chosen': '',
            'fixed_context': {},
            'open_dimensions': [],
            'visits': 0,
            'total_reward': 0.0,
            'relationship': 'child',
        }
        for key, default_val in defaults.items():
            if key not in data:
                data[key] = default_val
        return cls(**data)


class IdeaTree:
    """
    Manages the idea exploration tree.

    The tree tracks which ideas have been tried, their outcomes,
    and the git commit state for each node (enabling backtracking).
    """

    def __init__(self, tree_path: Optional[Path] = None):
        self.tree_path = tree_path
        self.nodes: Dict[str, IdeaNode] = {}
        self.root_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        self.competition_slug: str = ""
        self.run_id: str = ""

    def add_node(
        self,
        idea_title: str,
        parent_commit: str,
        parent_node_id: Optional[str] = None,
        branch_name: str = ""
    ) -> IdeaNode:
        """Add a new idea node to the tree."""
        node_id = str(uuid.uuid4())[:8]

        # Calculate depth
        depth = 0
        if parent_node_id and parent_node_id in self.nodes:
            depth = self.nodes[parent_node_id].depth + 1

        node = IdeaNode(
            id=node_id,
            idea_title=idea_title,
            parent_commit=parent_commit,
            parent_node_id=parent_node_id,
            depth=depth,
            branch_name=branch_name,
            timestamp=datetime.now().isoformat(),
        )

        self.nodes[node_id] = node

        # Update parent's children list
        if parent_node_id and parent_node_id in self.nodes:
            self.nodes[parent_node_id].children.append(node_id)

        # Set as root if no parent
        if parent_node_id is None:
            self.root_id = node_id

        # Set as current node
        self.current_node_id = node_id

        return node

    def get_node(self, node_id: str) -> Optional[IdeaNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def update_status(
        self,
        node_id: str,
        status: str,
        score: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Update node status after experiment."""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            if score is not None:
                self.nodes[node_id].score = score
            if error_message:
                self.nodes[node_id].error_message = error_message

    def get_current_node(self) -> Optional[IdeaNode]:
        """Get the node currently being explored."""
        if self.current_node_id:
            return self.nodes.get(self.current_node_id)
        return None

    def set_current_node(self, node_id: str):
        """Set the current node pointer."""
        if node_id in self.nodes:
            self.current_node_id = node_id

    def get_parent(self, node_id: str) -> Optional[IdeaNode]:
        """Get the parent node."""
        node = self.nodes.get(node_id)
        if node and node.parent_node_id:
            return self.nodes.get(node.parent_node_id)
        return None

    def get_siblings(self, node_id: str) -> List[IdeaNode]:
        """Get sibling nodes (same parent)."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_node_id:
            return []

        parent = self.nodes.get(node.parent_node_id)
        if not parent:
            return []

        return [
            self.nodes[child_id]
            for child_id in parent.children
            if child_id != node_id and child_id in self.nodes
        ]

    def get_pending_siblings(self, node_id: str) -> List[IdeaNode]:
        """Get untried sibling nodes."""
        return [s for s in self.get_siblings(node_id) if s.status == 'pending']

    def get_children(self, node_id: str) -> List[IdeaNode]:
        """Get child nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_pending_children(self, node_id: str) -> List[IdeaNode]:
        """Get pending child nodes."""
        return [c for c in self.get_children(node_id) if c.status == 'pending']

    def get_best_child(self, node_id: str, metric_direction: str = 'higher_better') -> Optional[IdeaNode]:
        """Get the best-performing child node."""
        children = self.get_children(node_id)
        scored_children = [c for c in children if c.score is not None]

        if not scored_children:
            return None

        if metric_direction == 'higher_better':
            return max(scored_children, key=lambda c: c.score)
        else:
            return min(scored_children, key=lambda c: c.score)

    def get_best_sibling(self, node_id: str, metric_direction: str = 'higher_better') -> Optional[IdeaNode]:
        """For BF mode: get the best-performing sibling (including self)."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_node_id:
            return node

        parent = self.nodes.get(node.parent_node_id)
        if not parent:
            return node

        all_siblings = [self.nodes[cid] for cid in parent.children if cid in self.nodes]
        scored = [s for s in all_siblings if s.score is not None and s.status == 'improved']

        if not scored:
            return None

        if metric_direction == 'higher_better':
            return max(scored, key=lambda s: s.score)
        else:
            return min(scored, key=lambda s: s.score)

    def get_best_node(self, metric_direction: str = 'higher_better') -> Optional[IdeaNode]:
        """Get the best-performing node in the entire tree."""
        scored_nodes = [n for n in self.nodes.values() if n.score is not None]

        if not scored_nodes:
            return self.nodes.get(self.root_id)

        if metric_direction == 'higher_better':
            return max(scored_nodes, key=lambda n: n.score)
        else:
            return min(scored_nodes, key=lambda n: n.score)

    def get_backtrack_target(self, node_id: str) -> Optional[IdeaNode]:
        """
        Find the parent node to backtrack to.

        Walks up the tree to find a node with untried siblings or children.
        Returns None if entire tree is exhausted.
        """
        current = self.nodes.get(node_id)

        while current:
            # Check for pending siblings at this level
            pending_siblings = self.get_pending_siblings(current.id)
            if pending_siblings:
                # Return parent - we'll try a sibling from there
                parent = self.get_parent(current.id)
                return parent

            # Move up to parent
            if current.parent_node_id:
                current = self.nodes.get(current.parent_node_id)
            else:
                break

        return None  # Tree exhausted

    # ─── UCB1 Selection (F4 from spec) ────────────────────────────────────────

    def ucb1_score(
        self,
        node_id: str,
        exploration_constant: float = 1.414,
        metric_direction: str = 'higher_better'
    ) -> float:
        """
        Calculate UCB1 score for a node.

        UCB1 = exploitation + C * sqrt(log(total_runs) / node_visits)

        Args:
            node_id: The node to score
            exploration_constant: C parameter (default sqrt(2))
            metric_direction: 'higher_better' or 'lower_better'

        Returns:
            UCB1 score (higher = should explore more)
        """
        import math

        node = self.nodes.get(node_id)
        if not node:
            return float('-inf')

        total_runs = sum(n.visits for n in self.nodes.values())
        if total_runs == 0:
            return float('inf')  # Prioritize unvisited nodes

        if node.visits == 0:
            return float('inf')  # Unvisited node gets priority

        # Exploitation: average reward (normalized score)
        avg_reward = node.total_reward / node.visits

        # For lower_better metrics, we negate so UCB1 still works
        if metric_direction == 'lower_better':
            avg_reward = -avg_reward

        # Exploration bonus
        exploration = exploration_constant * math.sqrt(math.log(total_runs) / node.visits)

        return avg_reward + exploration

    def select_node_ucb1(
        self,
        exploration_constant: float = 1.414,
        metric_direction: str = 'higher_better',
        min_score_threshold: Optional[float] = None
    ) -> Optional[IdeaNode]:
        """
        Select the next node to expand using UCB1.

        Excludes:
        - Nodes with status 'plateau' or 'crashed'
        - Nodes below min_score_threshold (if provided)

        Args:
            exploration_constant: C parameter for UCB1
            metric_direction: 'higher_better' or 'lower_better'
            min_score_threshold: Optional minimum score to consider

        Returns:
            The node with highest UCB1 score, or None if tree exhausted
        """
        candidates = []

        for node in self.nodes.values():
            # Skip pruned nodes
            if node.status in ('plateau', 'crashed'):
                continue

            # Skip nodes below threshold (F8 pruning)
            if min_score_threshold is not None and node.score is not None:
                if metric_direction == 'higher_better' and node.score < min_score_threshold:
                    continue
                if metric_direction == 'lower_better' and node.score > min_score_threshold:
                    continue

            # Only select nodes that have been evaluated (or root)
            if node.status in ('improved', 'no_improvement') or node.id == self.root_id:
                candidates.append(node)

        if not candidates:
            return None

        # Calculate UCB1 for each candidate
        scored = [
            (node, self.ucb1_score(node.id, exploration_constant, metric_direction))
            for node in candidates
        ]

        # Sort by UCB1 score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0] if scored else None

    def update_node_reward(
        self,
        node_id: str,
        score: float,
        baseline_score: float,
        metric_direction: str = 'higher_better'
    ):
        """
        Update a node's reward after an experiment.

        Propagates the reward up to ancestors (MCTS-style backpropagation).

        Args:
            node_id: The node that was just evaluated
            score: The experiment score
            baseline_score: The baseline score for normalization
            metric_direction: 'higher_better' or 'lower_better'
        """
        # Normalize reward to [0, 1] range approximately
        if metric_direction == 'higher_better':
            # Reward = improvement over baseline (can be negative)
            reward = (score - baseline_score) / max(abs(baseline_score), 0.001)
        else:
            # For lower_better, improvement means lower score
            reward = (baseline_score - score) / max(abs(baseline_score), 0.001)

        # Clamp to reasonable range
        reward = max(-1.0, min(1.0, reward))

        # Propagate up the tree
        current = self.nodes.get(node_id)
        while current:
            current.visits += 1
            current.total_reward += reward

            if current.parent_node_id:
                current = self.nodes.get(current.parent_node_id)
            else:
                break

    def get_expansion_context(self, node_id: str) -> Dict[str, Any]:
        """
        Get context for I-MCTS expansion (F3 from spec).

        Returns the selected node, its siblings with results, and its parent.
        This context is passed to the LLM when generating new experiments.

        Args:
            node_id: The node to expand from

        Returns:
            Dict with 'node', 'siblings', 'parent' keys
        """
        node = self.nodes.get(node_id)
        if not node:
            return {'node': None, 'siblings': [], 'parent': None}

        siblings = self.get_siblings(node_id)
        parent = self.get_parent(node_id)

        # Format siblings with their results
        sibling_info = []
        for sib in siblings:
            sibling_info.append({
                'title': sib.idea_title,
                'hypothesis': sib.hypothesis,
                'dimension': sib.dimension_answered,
                'value': sib.value_chosen,
                'status': sib.status,
                'score': sib.score,
            })

        return {
            'node': {
                'title': node.idea_title,
                'hypothesis': node.hypothesis,
                'fixed_context': node.fixed_context,
                'open_dimensions': node.open_dimensions,
                'score': node.score,
            },
            'siblings': sibling_info,
            'parent': {
                'title': parent.idea_title if parent else None,
                'hypothesis': parent.hypothesis if parent else None,
                'open_dimensions': parent.open_dimensions if parent else [],
            } if parent else None,
        }

    def add_child_node(
        self,
        parent_node_id: str,
        idea_title: str,
        parent_commit: str,
        hypothesis: str = "",
        dimension_answered: str = "",
        value_chosen: str = "",
        branch_name: str = ""
    ) -> IdeaNode:
        """
        Add a child node (answers an open dimension from parent).

        Args:
            parent_node_id: ID of the parent node
            idea_title: Title of the new idea
            parent_commit: Git commit SHA
            hypothesis: One-sentence hypothesis
            dimension_answered: Which dimension this answers
            value_chosen: The value chosen for that dimension
            branch_name: Git branch name

        Returns:
            The new child node
        """
        parent = self.nodes.get(parent_node_id)
        if not parent:
            raise ValueError(f"Parent node {parent_node_id} not found")

        # Inherit fixed context from parent and add this dimension
        fixed_context = dict(parent.fixed_context)
        if dimension_answered:
            fixed_context[dimension_answered] = value_chosen

        # Open dimensions are inherited minus the one we just answered
        open_dims = [d for d in parent.open_dimensions if d != dimension_answered]

        node = self.add_node(idea_title, parent_commit, parent_node_id, branch_name)
        node.hypothesis = hypothesis
        node.dimension_answered = dimension_answered
        node.value_chosen = value_chosen
        node.fixed_context = fixed_context
        node.open_dimensions = open_dims
        node.relationship = "child"

        return node

    def add_sibling_node(
        self,
        sibling_node_id: str,
        idea_title: str,
        parent_commit: str,
        hypothesis: str = "",
        value_chosen: str = "",
        branch_name: str = ""
    ) -> IdeaNode:
        """
        Add a sibling node (alternative answer to same dimension).

        Args:
            sibling_node_id: ID of an existing sibling
            idea_title: Title of the new idea
            parent_commit: Git commit SHA (should be parent's commit)
            hypothesis: One-sentence hypothesis
            value_chosen: The alternative value for the same dimension
            branch_name: Git branch name

        Returns:
            The new sibling node
        """
        sibling = self.nodes.get(sibling_node_id)
        if not sibling:
            raise ValueError(f"Sibling node {sibling_node_id} not found")

        parent_node_id = sibling.parent_node_id
        if not parent_node_id:
            raise ValueError("Cannot add sibling to root node")

        parent = self.nodes.get(parent_node_id)

        # Inherit same fixed context as sibling (same parent context)
        # but with different value for the same dimension
        fixed_context = dict(parent.fixed_context) if parent else {}
        dimension = sibling.dimension_answered
        if dimension:
            fixed_context[dimension] = value_chosen

        node = self.add_node(idea_title, parent_commit, parent_node_id, branch_name)
        node.hypothesis = hypothesis
        node.dimension_answered = dimension
        node.value_chosen = value_chosen
        node.fixed_context = fixed_context
        node.open_dimensions = list(sibling.open_dimensions)  # Same open dims as sibling
        node.relationship = "sibling"

        # Track sibling relationship
        sibling.siblings.append(node.id)
        node.siblings.append(sibling_node_id)
        # Also track other existing siblings
        for other_sib_id in sibling.siblings:
            if other_sib_id != node.id:
                node.siblings.append(other_sib_id)
                other_sib = self.nodes.get(other_sib_id)
                if other_sib:
                    other_sib.siblings.append(node.id)

        return node

    def update_open_dimensions(self, node_id: str, new_dimensions: List[str]):
        """
        Update open dimensions for a node after reflection (F6 from spec).

        Called after an experiment to discover new exploration dimensions.

        Args:
            node_id: The node to update
            new_dimensions: Newly discovered dimensions to add
        """
        node = self.nodes.get(node_id)
        if node:
            # Add new dimensions, avoiding duplicates
            for dim in new_dimensions:
                if dim not in node.open_dimensions:
                    node.open_dimensions.append(dim)

    def get_next_pending_at_level(self, parent_node_id: Optional[str]) -> Optional[IdeaNode]:
        """Get next pending node at a specific level (children of parent)."""
        if parent_node_id is None:
            # Root level
            if self.root_id and self.nodes[self.root_id].status == 'pending':
                return self.nodes[self.root_id]
            return None

        parent = self.nodes.get(parent_node_id)
        if not parent:
            return None

        for child_id in parent.children:
            child = self.nodes.get(child_id)
            if child and child.status == 'pending':
                return child

        return None

    def mark_branch_exhausted(self, node_id: str):
        """Mark entire branch from this node as exhausted (plateau)."""
        node = self.nodes.get(node_id)
        if node:
            node.status = 'plateau'
            for child_id in node.children:
                self.mark_branch_exhausted(child_id)

    def get_idea_path(self, node_id: str) -> List[str]:
        """Get the path of idea titles from root to this node."""
        path = []
        current = self.nodes.get(node_id)

        while current:
            path.insert(0, self._slugify(current.idea_title))
            if current.parent_node_id:
                current = self.nodes.get(current.parent_node_id)
            else:
                break

        return path

    def _slugify(self, title: str) -> str:
        """Convert idea title to URL-safe slug."""
        import re
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        slug = re.sub(r'-+', '-', slug)
        return slug[:30].strip('-')

    def generate_branch_name(
        self,
        node_id: str,
        competition_slug: str,
        run_id: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Generate branch name for a node."""
        timestamp = timestamp or datetime.now()
        ts_str = timestamp.strftime("%Y%m%d-%H%M")
        idea_path = self.get_idea_path(node_id)
        path_str = "/".join(idea_path)
        return f"{competition_slug}/{run_id}/{path_str}-{ts_str}"

    def get_max_depth(self) -> int:
        """Get the maximum depth of the tree."""
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def count_by_status(self) -> Dict[str, int]:
        """Count nodes by status."""
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.status] = counts.get(node.status, 0) + 1
        return counts

    def get_improved_path(self, metric_direction: str = 'higher_better') -> List[IdeaNode]:
        """Get the path of improved nodes leading to best score."""
        best = self.get_best_node(metric_direction)
        if not best:
            return []

        path = []
        current = best

        while current:
            if current.status == 'improved' or current.id == self.root_id:
                path.insert(0, current)
            if current.parent_node_id:
                current = self.nodes.get(current.parent_node_id)
            else:
                break

        return path

    def all_siblings_tested(self, node_id: str) -> bool:
        """Check if all siblings (including self) have been tested."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_node_id:
            return True

        parent = self.nodes.get(node.parent_node_id)
        if not parent:
            return True

        for child_id in parent.children:
            child = self.nodes.get(child_id)
            if child and child.status == 'pending':
                return False

        return True

    def save(self):
        """Persist tree to disk."""
        if not self.tree_path:
            return

        data = {
            'root_id': self.root_id,
            'current_node_id': self.current_node_id,
            'competition_slug': self.competition_slug,
            'run_id': self.run_id,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

        self.tree_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tree_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        """Load tree from disk. Returns True if loaded successfully."""
        if not self.tree_path or not self.tree_path.exists():
            return False

        try:
            with open(self.tree_path, 'r') as f:
                data = json.load(f)

            self.root_id = data.get('root_id')
            self.current_node_id = data.get('current_node_id')
            self.competition_slug = data.get('competition_slug', '')
            self.run_id = data.get('run_id', '')
            self.nodes = {
                nid: IdeaNode.from_dict(ndata)
                for nid, ndata in data.get('nodes', {}).items()
            }
            return True

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load idea tree: {e}")
            return False

    def render_tree(self, metric_direction: str = 'higher_better') -> str:
        """Render tree as ASCII art for display."""
        if not self.root_id:
            return "Empty tree"

        best_path = set(n.id for n in self.get_improved_path(metric_direction))
        lines = []

        def render_node(node_id: str, prefix: str = "", is_last: bool = True):
            node = self.nodes.get(node_id)
            if not node:
                return

            # Status symbols
            status_symbols = {
                'improved': '✓',
                'no_improvement': '✗',
                'crashed': '💥',
                'plateau': '⏸',
                'running': '⏳',
                'pending': '○',
            }
            symbol = status_symbols.get(node.status, '?')

            # Score display
            score_str = f": {node.score:.4f}" if node.score is not None else ""

            # Best path marker
            best_marker = " ← BEST" if node.id in best_path and node.id == self.get_best_node(metric_direction).id else ""

            # Current node marker
            current_marker = " *" if node.id == self.current_node_id else ""

            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}[{node.idea_title}{score_str}] {symbol}{best_marker}{current_marker}")

            # Render children
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = [self.nodes[cid] for cid in node.children if cid in self.nodes]
            for i, child in enumerate(children):
                render_node(child.id, child_prefix, i == len(children) - 1)

        # Start with root
        root = self.nodes.get(self.root_id)
        if root:
            score_str = f": {root.score:.4f}" if root.score is not None else ""
            symbol = '✓' if root.status == 'improved' else '○'
            lines.append(f"[{root.idea_title}{score_str}] {symbol}")

            children = [self.nodes[cid] for cid in root.children if cid in self.nodes]
            for i, child in enumerate(children):
                render_node(child.id, "", i == len(children) - 1)

        return "\n".join(lines)


def reconstruct_tree_from_git(repo_dir: Path, competition_slug: str, run_id: str) -> IdeaTree:
    """
    Emergency recovery: rebuild tree from git commit history.

    This is a fallback when idea_tree.json is lost/corrupted.
    """
    import subprocess
    import re

    tree = IdeaTree()
    tree.competition_slug = competition_slug
    tree.run_id = run_id

    # Get all commits with IMPROVE in message
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '--grep=IMPROVE', '--format=%H %s'],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return tree

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                commit_sha, message = parts
                commits.append((commit_sha, message))

        # Process in reverse order (oldest first)
        commits.reverse()

        prev_node_id = None
        for commit_sha, message in commits:
            # Parse: "IMPROVE: {idea_title} | {old_score} -> {new_score}"
            match = re.search(r'IMPROVE.*?:\s*(.+?)\s*\|', message)
            if match:
                idea_title = match.group(1).strip()
            else:
                idea_title = message.replace('IMPROVE:', '').strip()[:50]

            # Extract score
            score_match = re.search(r'-> ([\d.]+)', message)
            score = float(score_match.group(1)) if score_match else None

            # Get parent commit
            parent_result = subprocess.run(
                ['git', 'rev-parse', f'{commit_sha}^'],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            parent_commit = parent_result.stdout.strip() if parent_result.returncode == 0 else ""

            node = tree.add_node(
                idea_title=idea_title,
                parent_commit=parent_commit,
                parent_node_id=prev_node_id,
            )
            node.status = 'improved'
            node.score = score

            prev_node_id = node.id

    except Exception as e:
        print(f"Warning: Failed to reconstruct tree from git: {e}")

    return tree
