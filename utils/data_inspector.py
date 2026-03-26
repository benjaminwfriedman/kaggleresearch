"""
Data inspection utilities for competition data analysis.

This module provides functions to analyze downloaded competition data
and generate a structured summary for the code agent to use when
generating baseline train.py files.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import random


@dataclass
class FileInfo:
    """Information about a single file."""
    path: str
    name: str
    extension: str
    size_bytes: int
    size_human: str


@dataclass
class DirectoryInfo:
    """Information about a directory."""
    path: str
    name: str
    file_count: int
    subdirs: List[str]
    sample_files: List[str]


@dataclass
class CSVInfo:
    """Information about a CSV file."""
    path: str
    name: str
    num_rows: int
    num_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    sample_rows: List[Dict[str, Any]]
    null_counts: Dict[str, int]
    unique_counts: Dict[str, int]


@dataclass
class DataInspectionResult:
    """Complete inspection result for a competition dataset."""
    # File structure
    root_files: List[FileInfo]
    directories: List[DirectoryInfo]

    # CSV details
    csv_files: List[CSVInfo]

    # Media files
    audio_files: List[str]
    image_files: List[str]

    # File counts by extension
    extension_counts: Dict[str, int]

    # Inferred problem type
    inferred_problem_type: str
    problem_type_reasoning: str

    # Total sizes
    total_size_bytes: int
    total_file_count: int


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _get_file_info(file_path: Path) -> FileInfo:
    """Get info about a single file."""
    stat = file_path.stat()
    return FileInfo(
        path=str(file_path),
        name=file_path.name,
        extension=file_path.suffix.lower(),
        size_bytes=stat.st_size,
        size_human=_human_readable_size(stat.st_size)
    )


def _get_directory_info(dir_path: Path, max_sample_files: int = 10) -> DirectoryInfo:
    """Get info about a directory."""
    files = []
    subdirs = []

    for item in dir_path.iterdir():
        if item.is_file():
            files.append(item.name)
        elif item.is_dir():
            subdirs.append(item.name)

    # Sample files randomly if too many
    sample_files = files[:max_sample_files] if len(files) <= max_sample_files else random.sample(files, max_sample_files)

    return DirectoryInfo(
        path=str(dir_path),
        name=dir_path.name,
        file_count=len(files),
        subdirs=subdirs,
        sample_files=sorted(sample_files)
    )


def _inspect_csv(csv_path: Path, max_sample_rows: int = 5) -> Optional[CSVInfo]:
    """Inspect a CSV file and return structured info."""
    try:
        import pandas as pd

        # Read just the header first to get column count
        df_sample = pd.read_csv(csv_path, nrows=100)

        # Get full row count efficiently
        with open(csv_path, 'r') as f:
            num_rows = sum(1 for _ in f) - 1  # Subtract header

        # Get sample rows as dicts
        sample_df = df_sample.head(max_sample_rows)
        sample_rows = sample_df.to_dict('records')

        # Convert numpy types to Python types for JSON serialization
        for row in sample_rows:
            for k, v in row.items():
                if hasattr(v, 'item'):  # numpy scalar
                    row[k] = v.item()
                elif pd.isna(v):
                    row[k] = None

        # Get dtype info
        dtypes = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}

        # Null and unique counts
        null_counts = df_sample.isnull().sum().to_dict()
        unique_counts = {col: int(df_sample[col].nunique()) for col in df_sample.columns}

        return CSVInfo(
            path=str(csv_path),
            name=csv_path.name,
            num_rows=num_rows,
            num_cols=len(df_sample.columns),
            columns=list(df_sample.columns),
            dtypes=dtypes,
            sample_rows=sample_rows,
            null_counts=null_counts,
            unique_counts=unique_counts
        )
    except Exception as e:
        print(f"Warning: Could not inspect CSV {csv_path}: {e}")
        return None


def _count_files_by_extension(data_dir: Path) -> Dict[str, int]:
    """Count files by extension recursively."""
    counts: Dict[str, int] = {}

    for file_path in data_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1

    return counts


def _find_media_files(data_dir: Path, extensions: set, max_files: int = 20) -> List[str]:
    """Find media files with given extensions."""
    files = []
    for file_path in data_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            # Return relative path from data_dir
            rel_path = file_path.relative_to(data_dir)
            files.append(str(rel_path))
            if len(files) >= max_files:
                break
    return files


def _infer_problem_type(
    extension_counts: Dict[str, int],
    csv_files: List[CSVInfo],
    metric: str = ""
) -> Tuple[str, str]:
    """
    Infer the problem type from data inspection results.

    Returns:
        Tuple of (problem_type, reasoning)
    """
    metric_lower = metric.lower()

    # Audio detection
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    audio_count = sum(extension_counts.get(ext, 0) for ext in audio_extensions)

    if audio_count > 0:
        return ('audio-classification',
                f"Found {audio_count} audio files. This is an audio classification problem.")

    # Image detection
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_count = sum(extension_counts.get(ext, 0) for ext in image_extensions)

    if image_count > 0:
        if any(x in metric_lower for x in ['iou', 'dice', 'segment']):
            return ('image-segmentation',
                    f"Found {image_count} images and segmentation metric. This is an image segmentation problem.")
        if any(x in metric_lower for x in ['rmse', 'mse', 'mae', 'mape', 'rmsle', 'r2', 'r-squared']):
            return ('image-regression',
                    f"Found {image_count} images with regression metric ({metric}). This is an image regression problem.")
        return ('image-classification',
                f"Found {image_count} images. This is an image classification problem.")

    # Text detection (from CSV analysis)
    has_text_columns = False
    for csv_info in csv_files:
        if csv_info and 'train' in csv_info.name.lower():
            for col, dtype in csv_info.dtypes.items():
                if 'object' in dtype.lower():
                    # Check if it's likely text (high unique count or long strings)
                    unique = csv_info.unique_counts.get(col, 0)
                    if unique > 100:  # Likely text, not categorical
                        has_text_columns = True
                        break

    if has_text_columns:
        if any(x in metric_lower for x in ['rmse', 'mse', 'mae', 'mape']):
            return ('nlp-regression',
                    "Found text columns in training data with regression metric.")
        return ('nlp-classification',
                "Found text columns in training data.")

    # Time series detection
    for csv_info in csv_files:
        if csv_info:
            time_cols = [c for c in csv_info.columns
                        if any(x in c.lower() for x in ['date', 'time', 'timestamp', 'day', 'month', 'year'])]
            if time_cols and csv_info.num_cols < 30:
                return ('time-series',
                        f"Found time-related columns: {time_cols}. This is a time series problem.")

    # Default to tabular
    if any(x in metric_lower for x in ['rmse', 'mse', 'mae', 'mape', 'rmsle']):
        return ('tabular-regression',
                "Tabular data with regression metric.")

    return ('tabular-classification',
            "Tabular data with classification metric.")


def inspect_competition_data(
    data_dir: Path,
    metric: str = "",
    max_csv_sample_rows: int = 5
) -> DataInspectionResult:
    """
    Inspect competition data directory and return structured summary.

    Args:
        data_dir: Path to the data directory
        metric: Competition evaluation metric (for problem type inference)
        max_csv_sample_rows: Number of sample rows to include per CSV

    Returns:
        DataInspectionResult with complete data summary
    """
    data_dir = Path(data_dir)

    # Get root-level files and directories
    root_files = []
    directories = []

    for item in data_dir.iterdir():
        if item.is_file():
            root_files.append(_get_file_info(item))
        elif item.is_dir():
            directories.append(_get_directory_info(item))

    # Count files by extension
    extension_counts = _count_files_by_extension(data_dir)

    # Find and inspect CSV files
    csv_files = []
    for csv_path in data_dir.rglob('*.csv'):
        csv_info = _inspect_csv(csv_path, max_csv_sample_rows)
        if csv_info:
            csv_files.append(csv_info)

    # Find media files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    audio_files = _find_media_files(data_dir, audio_extensions)
    image_files = _find_media_files(data_dir, image_extensions)

    # Calculate totals
    total_size = sum(f.size_bytes for f in root_files)
    total_count = sum(extension_counts.values())

    # Infer problem type
    problem_type, reasoning = _infer_problem_type(extension_counts, csv_files, metric)

    return DataInspectionResult(
        root_files=root_files,
        directories=directories,
        csv_files=csv_files,
        audio_files=audio_files,
        image_files=image_files,
        extension_counts=extension_counts,
        inferred_problem_type=problem_type,
        problem_type_reasoning=reasoning,
        total_size_bytes=total_size,
        total_file_count=total_count
    )


def format_inspection_for_prompt(result: DataInspectionResult) -> str:
    """
    Format inspection result as a string for inclusion in LLM prompts.

    Args:
        result: DataInspectionResult from inspect_competition_data

    Returns:
        Formatted string describing the data structure
    """
    lines = []

    lines.append("## Data Structure Overview")
    lines.append("")
    lines.append(f"**Problem Type:** {result.inferred_problem_type}")
    lines.append(f"**Reasoning:** {result.problem_type_reasoning}")
    lines.append(f"**Total Files:** {result.total_file_count}")
    lines.append("")

    # File type breakdown
    lines.append("### File Types")
    for ext, count in sorted(result.extension_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"- `{ext}`: {count} files")
    lines.append("")

    # Root files
    if result.root_files:
        lines.append("### Root-Level Files")
        for f in result.root_files:
            lines.append(f"- `{f.name}` ({f.size_human})")
        lines.append("")

    # Directories
    if result.directories:
        lines.append("### Directories")
        for d in result.directories:
            lines.append(f"- `{d.name}/` — {d.file_count} files")
            if d.subdirs:
                lines.append(f"  - Subdirectories: {', '.join(d.subdirs[:5])}")
            if d.sample_files:
                lines.append(f"  - Sample files: {', '.join(d.sample_files[:5])}")
        lines.append("")

    # CSV files
    if result.csv_files:
        lines.append("### CSV Files")
        for csv in result.csv_files:
            lines.append(f"\n#### `{csv.name}`")
            lines.append(f"- Rows: {csv.num_rows:,}")
            lines.append(f"- Columns: {csv.num_cols}")
            lines.append(f"- Columns: `{', '.join(csv.columns)}`")
            lines.append("")
            lines.append("**Column Types:**")
            for col, dtype in csv.dtypes.items():
                null_count = csv.null_counts.get(col, 0)
                unique_count = csv.unique_counts.get(col, 0)
                lines.append(f"- `{col}`: {dtype} (nulls: {null_count}, unique: {unique_count})")
            lines.append("")
            lines.append("**Sample Rows:**")
            lines.append("```json")
            lines.append(json.dumps(csv.sample_rows, indent=2, default=str))
            lines.append("```")
        lines.append("")

    # Audio files
    if result.audio_files:
        lines.append("### Audio Files (sample)")
        for f in result.audio_files[:10]:
            lines.append(f"- `{f}`")
        if len(result.audio_files) > 10:
            total_audio = sum(1 for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
                            if ext in result.extension_counts)
            lines.append(f"- ... and more (see extension counts above)")
        lines.append("")

    # Image files
    if result.image_files:
        lines.append("### Image Files (sample)")
        for f in result.image_files[:10]:
            lines.append(f"- `{f}`")
        if len(result.image_files) > 10:
            lines.append(f"- ... and more (see extension counts above)")
        lines.append("")

    return "\n".join(lines)
