"""
Baseline template for 'other' / unknown problem types.
Uses LightGBM as a generic fallback.
This template should be modified based on competition specifics.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import lightgbm as lgb

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Do not modify these paths
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submissions/submission.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Problem type configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Set this based on competition requirements
TASK_TYPE = "regression"  # "regression" or "classification"
IS_BINARY = True  # Only relevant if TASK_TYPE == "classification"

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model and training configuration
# ═══════════════════════════════════════════════════════════════════════════════

def get_lgbm_params():
    """Get LightGBM parameters based on task type."""
    base_params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    if TASK_TYPE == "regression":
        base_params["objective"] = "regression"
        base_params["metric"] = "rmse"
    elif TASK_TYPE == "classification":
        if IS_BINARY:
            base_params["objective"] = "binary"
            base_params["metric"] = "binary_logloss"
        else:
            base_params["objective"] = "multiclass"
            base_params["metric"] = "multi_logloss"

    return base_params

N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Data loading and preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load train and test data. Modify paths as needed."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def get_target_and_features(train_df, test_df):
    """
    Extract target column and feature columns.
    Returns: X_train, y_train, X_test, feature_names, id_column_values
    """
    # EDITABLE: Modify these based on competition data
    target_col = "target"
    id_col = "id"

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

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
    """Train model with k-fold CV and generate predictions."""
    params = get_lgbm_params()

    # Handle multiclass
    if TASK_TYPE == "classification" and not IS_BINARY:
        n_classes = len(np.unique(y_train))
        params["num_class"] = n_classes
        oof_preds = np.zeros((len(X_train), n_classes))
        test_preds = np.zeros((len(X_test), n_classes))
    else:
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))

    cv_scores = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}/{N_FOLDS}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
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
        test_preds += model.predict(X_test) / N_FOLDS

        # Calculate fold score based on task type
        if TASK_TYPE == "regression":
            fold_score = np.sqrt(mean_squared_error(y_val, val_pred))
            metric_name = "RMSE"
        elif TASK_TYPE == "classification":
            if IS_BINARY:
                fold_score = roc_auc_score(y_val, val_pred)
                metric_name = "AUC"
            else:
                fold_score = accuracy_score(y_val, np.argmax(val_pred, axis=1))
                metric_name = "Accuracy"

        cv_scores.append(fold_score)
        print(f"Fold {fold + 1} {metric_name}: {fold_score:.6f}")

    print(f"\nCV Mean {metric_name}: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

    return oof_preds, test_preds, cv_scores


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Submission generation
# ═══════════════════════════════════════════════════════════════════════════════

def create_submission(test_ids, test_preds):
    """Create submission file."""
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TASK_TYPE == "regression":
        submission = pd.DataFrame({
            "id": test_ids,
            "target": test_preds
        })
    elif TASK_TYPE == "classification":
        if IS_BINARY:
            submission = pd.DataFrame({
                "id": test_ids,
                "target": test_preds
            })
        else:
            submission = pd.DataFrame({
                "id": test_ids,
                "target": np.argmax(test_preds, axis=1)
            })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    return submission


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Task type: {TASK_TYPE}")
    print(f"Binary classification: {IS_BINARY}" if TASK_TYPE == "classification" else "")

    print("\nLoading data...")
    train_df, test_df = load_data()

    print("Extracting features and target...")
    X_train, y_train, X_test, feature_cols, test_ids = get_target_and_features(train_df, test_df)

    print("Preprocessing features...")
    X_train, X_test = preprocess_features(X_train, X_test, feature_cols)

    print("Training model...")
    oof_preds, test_preds, cv_scores = train_and_predict(X_train, y_train, X_test)

    print("Creating submission...")
    create_submission(test_ids, test_preds)

    return np.mean(cv_scores)


if __name__ == "__main__":
    score = main()
    print(f"\nFinal CV Score: {score:.6f}")
