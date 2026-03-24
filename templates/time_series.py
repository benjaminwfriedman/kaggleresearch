"""
Baseline template for time series problems.
Uses LightGBM with manually crafted lag and rolling features.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Do not modify these paths
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submissions/submission.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model and training configuration
# ═══════════════════════════════════════════════════════════════════════════════

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000

# Lag and rolling window configurations
LAG_FEATURES = [1, 2, 3, 7, 14, 28]  # Days of lag
ROLLING_WINDOWS = [7, 14, 28]  # Rolling window sizes
ROLLING_AGGS = ["mean", "std", "min", "max"]  # Aggregation functions

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Data loading and preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load train and test data."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def parse_dates(df, date_col="date"):
    """Parse date column and extract date features."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract date components
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["dayofyear"] = df[date_col].dt.dayofyear
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    return df


def create_lag_features(df, target_col, group_cols=None, lags=None):
    """Create lag features for target variable."""
    df = df.copy()
    lags = lags or LAG_FEATURES

    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        if group_cols:
            df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)

    return df


def create_rolling_features(df, target_col, group_cols=None, windows=None, aggs=None):
    """Create rolling window features."""
    df = df.copy()
    windows = windows or ROLLING_WINDOWS
    aggs = aggs or ROLLING_AGGS

    for window in windows:
        for agg in aggs:
            col_name = f"{target_col}_rolling_{window}_{agg}"
            if group_cols:
                rolled = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).agg(agg)
                )
            else:
                rolled = df[target_col].shift(1).rolling(window=window, min_periods=1).agg(agg)
            df[col_name] = rolled

    return df


def get_target_and_features(train_df, test_df, target_col, id_col, date_col):
    """
    Extract target column and feature columns.
    Returns: X_train, y_train, X_test, feature_names, test_ids
    """
    # Features to exclude
    exclude_cols = [target_col, id_col, date_col]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].copy()
    test_ids = test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))

    return X_train, y_train, X_test, feature_cols, test_ids


def preprocess_features(X_train, X_test, feature_cols):
    """Preprocess features: handle categoricals, missing values."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Fill missing values
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("missing")
        X_test[col] = X_test[col].fillna("missing")

    # Encode categorical columns
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    return X_train, X_test


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict(X_train, y_train, X_test):
    """Train model with time series cross-validation and generate predictions."""

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    cv_scores = []

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"Fold {fold + 1}/{N_SPLITS}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=100),
            ],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / N_SPLITS

        fold_score = np.sqrt(mean_squared_error(y_val, val_pred))
        cv_scores.append(fold_score)
        print(f"Fold {fold + 1} RMSE: {fold_score:.6f}")

    print(f"\nCV Mean RMSE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

    return oof_preds, test_preds, cv_scores


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Submission generation
# ═══════════════════════════════════════════════════════════════════════════════

def create_submission(test_ids, test_preds):
    """Create submission file."""
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        "id": test_ids,
        "target": test_preds
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    return submission


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # EDITABLE: Column names
    target_col = "target"
    id_col = "id"
    date_col = "date"
    group_cols = None  # EDITABLE: Set to list of columns for grouped time series, e.g., ["store_id", "item_id"]

    print("Loading data...")
    train_df, test_df = load_data()

    print("Parsing dates...")
    train_df = parse_dates(train_df, date_col)
    test_df = parse_dates(test_df, date_col)

    # Sort by date
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    test_df = test_df.sort_values(date_col).reset_index(drop=True)

    print("Creating lag features...")
    train_df = create_lag_features(train_df, target_col, group_cols)
    # For test, we need to handle this carefully - often lags come from train
    # This is a simplified version; real implementation may need to concat train+test
    test_df = create_lag_features(test_df, target_col, group_cols)

    print("Creating rolling features...")
    train_df = create_rolling_features(train_df, target_col, group_cols)
    test_df = create_rolling_features(test_df, target_col, group_cols)

    print("Extracting features and target...")
    X_train, y_train, X_test, feature_cols, test_ids = get_target_and_features(
        train_df, test_df, target_col, id_col, date_col
    )

    print("Preprocessing features...")
    X_train, X_test = preprocess_features(X_train, X_test, feature_cols)

    print("Training model...")
    oof_preds, test_preds, cv_scores = train_and_predict(X_train, y_train, X_test)

    print("Creating submission...")
    create_submission(test_ids, test_preds)

    return np.mean(cv_scores)


if __name__ == "__main__":
    score = main()
    print(f"\nFinal CV RMSE: {score:.6f}")
