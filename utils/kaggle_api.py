"""
Kaggle API utilities for competition parsing and data download.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import subprocess


@dataclass
class CompetitionMeta:
    """Metadata about a Kaggle competition."""
    slug: str
    name: str
    description: str
    metric: str
    metric_direction: str  # "higher_better" or "lower_better"
    deadline: Optional[str] = None
    data_files: List[str] = field(default_factory=list)
    submission_format: Dict[str, str] = field(default_factory=dict)
    target_column: str = "target"
    id_column: str = "id"
    url: str = ""


def extract_slug_from_url(url: str) -> str:
    """
    Extract competition slug from Kaggle URL.

    Args:
        url: Full Kaggle competition URL

    Returns:
        Competition slug

    Examples:
        https://www.kaggle.com/competitions/titanic -> titanic
        https://www.kaggle.com/c/titanic -> titanic
    """
    # Handle both /competitions/ and /c/ URL formats
    patterns = [
        r'kaggle\.com/competitions/([^/\?]+)',
        r'kaggle\.com/c/([^/\?]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If no pattern matched, assume the URL is already the slug
    return url.strip().split('/')[-1]


def check_kaggle_credentials() -> Tuple[bool, str]:
    """
    Check if Kaggle API credentials are configured.

    Returns:
        Tuple of (is_configured, message)
    """
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

    if kaggle_json.exists():
        return True, "Kaggle credentials found"

    # Check environment variables
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        return True, "Kaggle credentials found in environment"

    # For Colab, check if file was uploaded
    colab_kaggle = Path('/root/.kaggle/kaggle.json')
    if colab_kaggle.exists():
        return True, "Kaggle credentials found (Colab)"

    return False, """
Kaggle API credentials not found. Please set up authentication:

Option 1 (Recommended for Colab):
  1. Go to kaggle.com -> Account -> API -> Create New Token
  2. Upload kaggle.json to Colab
  3. Run: !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Option 2 (Environment variables):
  Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
"""


def parse_competition(url: str) -> CompetitionMeta:
    """
    Parse competition metadata from Kaggle API.

    Args:
        url: Kaggle competition URL

    Returns:
        CompetitionMeta with competition details
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    slug = extract_slug_from_url(url)

    api = KaggleApi()
    api.authenticate()

    # Get competition details
    competitions = api.competitions_list(search=slug)
    comp = None
    for c in competitions:
        if c.ref == slug:
            comp = c
            break

    if comp is None:
        # Try direct lookup
        try:
            comp = api.competition_view(slug)
        except Exception:
            pass

    # Build metadata
    meta = CompetitionMeta(
        slug=slug,
        name=getattr(comp, 'title', slug) if comp else slug,
        description=getattr(comp, 'description', '') if comp else '',
        metric=getattr(comp, 'evaluationMetric', 'unknown') if comp else 'unknown',
        metric_direction=infer_metric_direction(getattr(comp, 'evaluationMetric', '') if comp else ''),
        deadline=str(getattr(comp, 'deadline', '')) if comp else None,
        url=url,
    )

    # Get data file list
    try:
        files = api.competition_list_files(slug)
        meta.data_files = [f.name for f in files]
    except Exception:
        meta.data_files = []

    # Infer submission format from sample submission if available
    meta.submission_format = infer_submission_format(meta.data_files)

    return meta


def infer_metric_direction(metric: str) -> str:
    """
    Infer whether higher or lower is better for a metric.

    Args:
        metric: Metric name string

    Returns:
        "higher_better" or "lower_better"
    """
    metric_lower = metric.lower()

    # Metrics where lower is better
    lower_better = [
        'rmse', 'mse', 'mae', 'mape', 'loss', 'error',
        'log_loss', 'logloss', 'cross_entropy', 'perplexity',
        'rae', 'rse', 'smape'
    ]

    for lb in lower_better:
        if lb in metric_lower:
            return "lower_better"

    # Default: higher is better (accuracy, f1, auc, etc.)
    return "higher_better"


def infer_submission_format(data_files: List[str]) -> Dict[str, str]:
    """
    Infer submission format from data files.

    Args:
        data_files: List of data file names

    Returns:
        Dict with 'id_column' and 'target_column' guesses
    """
    format_info = {
        'id_column': 'id',
        'target_column': 'target',
    }

    # Look for sample submission file
    sample_files = [f for f in data_files if 'sample' in f.lower() and 'submission' in f.lower()]

    if sample_files:
        format_info['sample_submission'] = sample_files[0]

    return format_info


def download_competition_data(slug: str, data_dir: Path) -> List[str]:
    """
    Download competition data to specified directory.

    Args:
        slug: Competition slug
        data_dir: Directory to download data to

    Returns:
        List of downloaded file paths
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download and unzip
    api.competition_download_files(slug, path=str(data_dir), quiet=False)

    # Unzip if needed
    zip_files = list(data_dir.glob('*.zip'))
    for zip_file in zip_files:
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(data_dir)
        zip_file.unlink()  # Remove zip after extraction

    # Return list of files
    return [str(f) for f in data_dir.iterdir() if f.is_file()]


def classify_problem_type(meta: CompetitionMeta, data_dir: Path) -> str:
    """
    Classify competition problem type based on metadata and data inspection.

    Args:
        meta: Competition metadata
        data_dir: Path to downloaded data

    Returns:
        Problem type string
    """
    import pandas as pd

    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    data_dir = Path(data_dir)

    has_images = False
    for ext in image_extensions:
        if list(data_dir.rglob(f'*{ext}')):
            has_images = True
            break

    # Check metric for segmentation hints
    metric_lower = meta.metric.lower()
    if has_images:
        if any(x in metric_lower for x in ['iou', 'dice', 'segment']):
            return 'image-segmentation'
        return 'image-classification'

    # Load train.csv to inspect
    train_path = data_dir / 'train.csv'
    if not train_path.exists():
        # Try to find any CSV
        csvs = list(data_dir.glob('*.csv'))
        train_candidates = [c for c in csvs if 'train' in c.name.lower()]
        if train_candidates:
            train_path = train_candidates[0]
        elif csvs:
            train_path = csvs[0]

    if train_path.exists():
        try:
            df = pd.read_csv(train_path, nrows=100)

            # Check for text columns (long strings)
            text_cols = []
            for col in df.select_dtypes(include=['object']).columns:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 50:  # Likely text data
                    text_cols.append(col)

            if text_cols:
                # Determine if classification or regression based on metric
                if any(x in metric_lower for x in ['rmse', 'mse', 'mae', 'mape']):
                    return 'nlp-regression'
                return 'nlp-classification'

            # Check for time series indicators
            date_cols = []
            for col in df.columns:
                if any(x in col.lower() for x in ['date', 'time', 'timestamp', 'day', 'month', 'year']):
                    date_cols.append(col)

            if date_cols and len(df.columns) < 20:  # Likely time series
                return 'time-series'

            # Default to tabular based on metric
            if any(x in metric_lower for x in ['rmse', 'mse', 'mae', 'mape', 'rmsle']):
                return 'tabular-regression'

            return 'tabular-classification'

        except Exception:
            pass

    # Fallback based on metric alone
    if any(x in metric_lower for x in ['rmse', 'mse', 'mae', 'mape']):
        return 'tabular-regression'

    return 'tabular-classification'


def get_template_for_problem_type(problem_type: str) -> str:
    """
    Get the template filename for a problem type.

    Args:
        problem_type: Classified problem type

    Returns:
        Template filename
    """
    mapping = {
        'tabular-classification': 'tabular_classification.py',
        'tabular-regression': 'tabular_regression.py',
        'image-classification': 'image_classification.py',
        'image-segmentation': 'image_segmentation.py',
        'nlp-classification': 'nlp_classification.py',
        'nlp-regression': 'nlp_regression.py',
        'time-series': 'time_series.py',
        'other': 'other.py',
    }
    return mapping.get(problem_type, 'other.py')
