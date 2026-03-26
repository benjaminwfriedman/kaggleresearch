"""Tests for Kaggle API utilities."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.kaggle_api import (
    extract_slug_from_url,
    infer_metric_direction,
    infer_submission_format,
    get_template_for_problem_type,
    CompetitionMeta,
)


class TestExtractSlugFromUrl:
    """Tests for URL slug extraction."""

    def test_competitions_url_format(self):
        """Extract from /competitions/ URL format."""
        url = "https://www.kaggle.com/competitions/titanic"
        assert extract_slug_from_url(url) == "titanic"

    def test_c_url_format(self):
        """Extract from /c/ URL format (legacy)."""
        url = "https://www.kaggle.com/c/house-prices-advanced-regression-techniques"
        assert extract_slug_from_url(url) == "house-prices-advanced-regression-techniques"

    def test_url_with_query_params(self):
        """Handle URL with query parameters."""
        url = "https://www.kaggle.com/competitions/titanic?tab=overview"
        assert extract_slug_from_url(url) == "titanic"

    def test_url_with_trailing_slash(self):
        """Handle URL with trailing path components."""
        url = "https://www.kaggle.com/competitions/titanic/overview"
        assert extract_slug_from_url(url) == "titanic"

    def test_plain_slug(self):
        """Handle plain slug (no URL)."""
        assert extract_slug_from_url("titanic") == "titanic"

    def test_slug_with_hyphens(self):
        """Handle slug with hyphens."""
        url = "https://www.kaggle.com/competitions/digit-recognizer"
        assert extract_slug_from_url(url) == "digit-recognizer"

    def test_http_url(self):
        """Handle http (not https) URL."""
        url = "http://www.kaggle.com/competitions/titanic"
        assert extract_slug_from_url(url) == "titanic"


class TestInferMetricDirection:
    """Tests for metric direction inference."""

    def test_rmse_is_lower_better(self):
        """RMSE is lower-is-better."""
        assert infer_metric_direction("RMSE") == "lower_better"
        assert infer_metric_direction("rmse") == "lower_better"

    def test_mse_is_lower_better(self):
        """MSE is lower-is-better."""
        assert infer_metric_direction("MSE") == "lower_better"
        assert infer_metric_direction("mean squared error") == "lower_better"

    def test_mae_is_lower_better(self):
        """MAE is lower-is-better."""
        assert infer_metric_direction("MAE") == "lower_better"
        assert infer_metric_direction("mean absolute error") == "lower_better"

    def test_log_loss_is_lower_better(self):
        """Log loss is lower-is-better."""
        assert infer_metric_direction("log_loss") == "lower_better"
        assert infer_metric_direction("logloss") == "lower_better"
        assert infer_metric_direction("logistic loss") == "lower_better"

    def test_cross_entropy_is_lower_better(self):
        """Cross entropy is lower-is-better."""
        assert infer_metric_direction("cross_entropy") == "lower_better"
        assert infer_metric_direction("categorical cross entropy") == "lower_better"

    def test_accuracy_is_higher_better(self):
        """Accuracy is higher-is-better."""
        assert infer_metric_direction("accuracy") == "higher_better"
        assert infer_metric_direction("Accuracy") == "higher_better"

    def test_auc_is_higher_better(self):
        """AUC is higher-is-better."""
        assert infer_metric_direction("auc") == "higher_better"
        assert infer_metric_direction("AUC-ROC") == "higher_better"

    def test_f1_is_higher_better(self):
        """F1 score is higher-is-better."""
        assert infer_metric_direction("f1") == "higher_better"
        assert infer_metric_direction("f1_score") == "higher_better"

    def test_map_is_higher_better(self):
        """mAP is higher-is-better."""
        assert infer_metric_direction("map") == "higher_better"
        assert infer_metric_direction("mean average precision") == "higher_better"

    def test_mape_is_lower_better(self):
        """MAPE is lower-is-better."""
        assert infer_metric_direction("mape") == "lower_better"
        assert infer_metric_direction("MAPE") == "lower_better"

    def test_unknown_metric_defaults_to_higher(self):
        """Unknown metric defaults to higher-is-better."""
        assert infer_metric_direction("custom_metric") == "higher_better"
        assert infer_metric_direction("") == "higher_better"


class TestInferSubmissionFormat:
    """Tests for submission format inference."""

    def test_finds_sample_submission(self):
        """Finds sample submission file."""
        files = ["train.csv", "test.csv", "sample_submission.csv"]
        fmt = infer_submission_format(files)

        assert "sample_submission" in fmt
        assert fmt["sample_submission"] == "sample_submission.csv"

    def test_finds_sample_submission_variants(self):
        """Finds variant naming for sample submission."""
        files = ["train.csv", "sampleSubmission.csv"]
        fmt = infer_submission_format(files)

        assert "sample_submission" in fmt

    def test_no_sample_submission(self):
        """Handles missing sample submission."""
        files = ["train.csv", "test.csv"]
        fmt = infer_submission_format(files)

        # Should still have default columns
        assert fmt["id_column"] == "id"
        assert fmt["target_column"] == "target"
        assert "sample_submission" not in fmt

    def test_empty_files_list(self):
        """Handles empty files list."""
        fmt = infer_submission_format([])

        assert fmt["id_column"] == "id"
        assert fmt["target_column"] == "target"


class TestGetTemplateForProblemType:
    """Tests for template selection."""

    def test_tabular_classification(self):
        """Tabular classification template."""
        template = get_template_for_problem_type("tabular-classification")
        assert template == "tabular_classification.py"

    def test_tabular_regression(self):
        """Tabular regression template."""
        template = get_template_for_problem_type("tabular-regression")
        assert template == "tabular_regression.py"

    def test_image_classification(self):
        """Image classification template."""
        template = get_template_for_problem_type("image-classification")
        assert template == "image_classification.py"

    def test_image_segmentation(self):
        """Image segmentation template."""
        template = get_template_for_problem_type("image-segmentation")
        assert template == "image_segmentation.py"

    def test_nlp_classification(self):
        """NLP classification template."""
        template = get_template_for_problem_type("nlp-classification")
        assert template == "nlp_classification.py"

    def test_nlp_regression(self):
        """NLP regression template."""
        template = get_template_for_problem_type("nlp-regression")
        assert template == "nlp_regression.py"

    def test_time_series(self):
        """Time series template."""
        template = get_template_for_problem_type("time-series")
        assert template == "time_series.py"

    def test_other(self):
        """Other/unknown type falls back to other.py."""
        template = get_template_for_problem_type("other")
        assert template == "other.py"

    def test_unknown_type(self):
        """Unknown type falls back to other.py."""
        template = get_template_for_problem_type("unknown-type")
        assert template == "other.py"


class TestCompetitionMeta:
    """Tests for CompetitionMeta dataclass."""

    def test_create_with_defaults(self):
        """Create CompetitionMeta with minimal fields."""
        meta = CompetitionMeta(
            slug="titanic",
            name="Titanic",
            description="Predict survival",
            metric="accuracy",
            metric_direction="higher_better",
        )

        assert meta.slug == "titanic"
        assert meta.data_files == []
        assert meta.submission_format == {}
        assert meta.target_column == "target"
        assert meta.id_column == "id"

    def test_create_with_all_fields(self):
        """Create CompetitionMeta with all fields."""
        meta = CompetitionMeta(
            slug="house-prices",
            name="House Prices",
            description="Predict house prices",
            metric="rmse",
            metric_direction="lower_better",
            deadline="2024-12-31",
            data_files=["train.csv", "test.csv"],
            submission_format={"id": "Id", "target": "SalePrice"},
            target_column="SalePrice",
            id_column="Id",
            url="https://kaggle.com/c/house-prices",
        )

        assert meta.metric == "rmse"
        assert meta.metric_direction == "lower_better"
        assert len(meta.data_files) == 2
        assert meta.target_column == "SalePrice"
