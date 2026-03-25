"""
Literature search and caching utilities.
Searches arXiv, Semantic Scholar, and Tavily for relevant papers and web content.
"""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
import urllib.parse
import urllib.request
import urllib.error


@dataclass
class Paper:
    """Represents a research paper."""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: str = ""
    year: int = 0
    citation_count: int = 0
    source: str = ""  # "arxiv" or "semantic_scholar"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        return cls(**data)

    def summary(self, max_abstract_len: int = 500) -> str:
        """Generate a short summary for LLM context."""
        abstract = self.abstract[:max_abstract_len]
        if len(self.abstract) > max_abstract_len:
            abstract += "..."

        return f"""
Title: {self.title}
Authors: {', '.join(self.authors[:3])}{'...' if len(self.authors) > 3 else ''}
Year: {self.year}
ArXiv: {self.arxiv_id or 'N/A'}
Citations: {self.citation_count}
Abstract: {abstract}
"""


def search_arxiv(
    query: str,
    n: int = 10,
    exclude_queries: Optional[Set[str]] = None,
    categories: Optional[List[str]] = None
) -> List[Paper]:
    """
    Search arXiv for papers matching the query.

    Args:
        query: Search query string
        n: Maximum number of papers to return
        exclude_queries: Set of queries already searched (to avoid duplicates)
        categories: List of arXiv categories to filter by (e.g., ['cs.LG', 'stat.ML'])

    Returns:
        List of Paper objects
    """
    try:
        import arxiv
    except ImportError:
        print("arxiv package not installed. Run: pip install arxiv")
        return []

    papers = []

    # Build search query with category filters
    search_query = query
    if categories:
        cat_filter = ' OR '.join(f'cat:{cat}' for cat in categories)
        search_query = f'({query}) AND ({cat_filter})'

    try:
        search = arxiv.Search(
            query=search_query,
            max_results=n,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in search.results():
            paper = Paper(
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                arxiv_id=result.entry_id.split('/')[-1],
                url=result.entry_id,
                year=result.published.year if result.published else 0,
                source="arxiv",
            )
            papers.append(paper)

    except Exception as e:
        print(f"arXiv search error: {e}")

    return papers


def search_semantic_scholar(
    query: str,
    n: int = 10,
    exclude_queries: Optional[Set[str]] = None,
    fields: Optional[List[str]] = None,
    max_retries: int = 3
) -> List[Paper]:
    """
    Search Semantic Scholar for papers matching the query.

    Args:
        query: Search query string
        n: Maximum number of papers to return
        exclude_queries: Set of queries already searched
        fields: List of fields to retrieve
        max_retries: Maximum number of retries on rate limit

    Returns:
        List of Paper objects
    """
    papers = []

    if fields is None:
        fields = ['title', 'authors', 'abstract', 'year', 'citationCount', 'externalIds', 'url']

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query,
        'limit': min(n, 100),  # API limit
        'fields': ','.join(fields),
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'KaggleResearch/1.0')

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())

            for item in data.get('data', []):
                arxiv_id = None
                if 'externalIds' in item and item['externalIds']:
                    arxiv_id = item['externalIds'].get('ArXiv')

                authors = []
                if 'authors' in item and item['authors']:
                    authors = [a.get('name', '') for a in item['authors'] if a.get('name')]

                paper = Paper(
                    title=item.get('title', ''),
                    authors=authors,
                    abstract=item.get('abstract', '') or '',
                    arxiv_id=arxiv_id,
                    url=item.get('url', ''),
                    year=item.get('year', 0) or 0,
                    citation_count=item.get('citationCount', 0) or 0,
                    source="semantic_scholar",
                )
                papers.append(paper)

            # Success - rate limit delay before returning
            time.sleep(1.0)
            return papers

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited - exponential backoff
                wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                print(f"  Semantic Scholar rate limited. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"Semantic Scholar HTTP error: {e}")
                break
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            break

    return papers


def search_tavily(
    query: str,
    n: int = 10,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> List[Paper]:
    """
    Search Tavily for web content related to the query.

    Tavily is useful for finding recent blog posts, tutorials, discussions,
    and other web content that may not be indexed in academic databases.

    Args:
        query: Search query string
        n: Maximum number of results to return
        include_domains: List of domains to include (e.g., ['arxiv.org', 'github.com'])
        exclude_domains: List of domains to exclude

    Returns:
        List of Paper objects (using Paper as a generic content container)
    """
    api_key = os.environ.get('TAVILY_API_KEY')
    if not api_key:
        print("  TAVILY_API_KEY not set. Skipping Tavily search.")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        print("  tavily-python package not installed. Run: pip install tavily-python")
        return []

    papers = []

    try:
        client = TavilyClient(api_key=api_key)

        search_kwargs = {
            'query': query,
            'max_results': min(n, 20),  # Tavily limit
            'search_depth': 'basic',  # Use basic to save API credits
            'include_answer': False,
        }

        if include_domains:
            search_kwargs['include_domains'] = include_domains
        if exclude_domains:
            search_kwargs['exclude_domains'] = exclude_domains

        response = client.search(**search_kwargs)

        for result in response.get('results', []):
            # Extract year from URL or content if possible
            year = datetime.now().year  # Default to current year

            # Create Paper object from Tavily result
            paper = Paper(
                title=result.get('title', ''),
                authors=[],  # Tavily doesn't provide authors
                abstract=result.get('content', ''),
                url=result.get('url', ''),
                year=year,
                citation_count=0,  # Not available from Tavily
                source="tavily",
            )
            papers.append(paper)

        # Small delay to be respectful of API
        time.sleep(0.5)

    except Exception as e:
        print(f"  Tavily search error: {e}")

    return papers


def search_papers(
    query: str,
    n: int = 10,
    exclude_queries: Optional[Set[str]] = None,
    problem_type: Optional[str] = None,
    include_tavily: bool = True
) -> List[Paper]:
    """
    Search arXiv, Semantic Scholar, and optionally Tavily. Deduplicate and rank results.

    Args:
        query: Search query string
        n: Maximum number of papers to return total
        exclude_queries: Set of queries already searched
        problem_type: Problem type for category filtering
        include_tavily: Whether to include Tavily web search results

    Returns:
        List of Paper objects, ranked by relevance and citations
    """
    # Determine arXiv categories based on problem type
    categories = ['cs.LG', 'stat.ML']
    if problem_type:
        if 'image' in problem_type:
            categories.append('cs.CV')
        if 'nlp' in problem_type:
            categories.extend(['cs.CL', 'cs.IR'])

    # Search academic sources
    arxiv_papers = search_arxiv(query, n=n, exclude_queries=exclude_queries, categories=categories)
    ss_papers = search_semantic_scholar(query, n=n, exclude_queries=exclude_queries)

    # Search web via Tavily (useful for blogs, tutorials, recent discussions)
    tavily_papers = []
    if include_tavily:
        # Focus on technical domains for ML/AI content
        tavily_papers = search_tavily(
            query,
            n=n,
            include_domains=[
                'arxiv.org', 'github.com', 'paperswithcode.com',
                'kaggle.com', 'huggingface.co', 'medium.com',
                'towardsdatascience.com', 'machinelearningmastery.com',
            ],
        )

    # Deduplicate by title similarity
    seen_titles = set()
    all_papers = []

    for paper in arxiv_papers + ss_papers + tavily_papers:
        # Simple normalization for dedup
        normalized = paper.title.lower().strip()
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            all_papers.append(paper)

    # Rank by citation count and recency
    current_year = datetime.now().year

    def score(p: Paper) -> float:
        # Citation score (log scale)
        cite_score = (p.citation_count + 1) ** 0.5
        # Recency bonus (papers from last 2 years get bonus)
        recency_bonus = max(0, 3 - (current_year - p.year)) * 2 if p.year else 0
        return cite_score + recency_bonus

    all_papers.sort(key=score, reverse=True)

    return all_papers[:n]


def cache_papers(papers: List[Paper], cache_path: Path) -> None:
    """
    Cache papers to JSON file.

    Args:
        papers: List of papers to cache
        cache_path: Path to papers.json
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    existing = []
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                existing = json.load(f)
        except Exception:
            pass

    # Merge new papers
    existing_titles = {p.get('title', '').lower() for p in existing}

    for paper in papers:
        if paper.title.lower() not in existing_titles:
            existing.append(paper.to_dict())
            existing_titles.add(paper.title.lower())

    with open(cache_path, 'w') as f:
        json.dump(existing, f, indent=2)


def load_cached_papers(cache_path: Path) -> List[Paper]:
    """
    Load papers from cache.

    Args:
        cache_path: Path to papers.json

    Returns:
        List of Paper objects
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return []

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return [Paper.from_dict(p) for p in data]
    except Exception as e:
        print(f"Error loading paper cache: {e}")
        return []


def save_search_history(query: str, history_path: Path) -> None:
    """
    Save a search query to history.

    Args:
        query: Query that was searched
        history_path: Path to search_history.json
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception:
            pass

    entry = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
    }
    history.append(entry)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def load_search_history(history_path: Path) -> Set[str]:
    """
    Load search history as set of queries.

    Args:
        history_path: Path to search_history.json

    Returns:
        Set of previously searched queries
    """
    history_path = Path(history_path)

    if not history_path.exists():
        return set()

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        return {entry['query'] for entry in history}
    except Exception:
        return set()


def build_search_query(
    problem_type: str,
    metric: str,
    context: Optional[str] = None,
    year: Optional[int] = None
) -> str:
    """
    Build a search query for literature review.

    Args:
        problem_type: Classification of the problem
        metric: Evaluation metric
        context: Additional context (e.g., failure summary)
        year: Year to include in search

    Returns:
        Search query string
    """
    if year is None:
        year = datetime.now().year

    # Map problem types to search terms
    type_terms = {
        'tabular-classification': 'tabular data classification gradient boosting',
        'tabular-regression': 'tabular data regression gradient boosting',
        'image-classification': 'image classification deep learning CNN',
        'image-segmentation': 'image segmentation semantic U-Net',
        'nlp-classification': 'text classification transformer BERT',
        'nlp-regression': 'text regression transformer NLP',
        'time-series': 'time series forecasting prediction',
        'other': 'machine learning',
    }

    base_query = type_terms.get(problem_type, 'machine learning')

    # Add metric to query
    metric_terms = metric.replace('_', ' ').lower()

    query = f"{base_query} {metric_terms} SOTA {year}"

    if context:
        # Add context but keep query reasonable length
        context_words = context.split()[:10]
        query += ' ' + ' '.join(context_words)

    return query


def format_papers_for_prompt(papers: List[Paper], max_papers: int = 10) -> str:
    """
    Format papers for inclusion in LLM prompt.

    Args:
        papers: List of papers
        max_papers: Maximum number to include

    Returns:
        Formatted string for prompt
    """
    if not papers:
        return "No relevant papers found."

    formatted = []
    for i, paper in enumerate(papers[:max_papers], 1):
        formatted.append(f"### Paper {i}\n{paper.summary()}")

    return '\n'.join(formatted)
