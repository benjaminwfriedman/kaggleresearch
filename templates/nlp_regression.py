"""
Baseline template for NLP regression problems.
Uses DistilBERT with regression head, MSE loss, and clipped predictions.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Do not modify these paths
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submissions/submission.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model and training configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prediction clipping bounds (set based on target distribution)
CLIP_MIN = None  # Set to clip predictions, e.g., 0.0
CLIP_MAX = None  # Set to clip predictions, e.g., 10.0

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Dataset class
# ═══════════════════════════════════════════════════════════════════════════════

class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model definition
# ═══════════════════════════════════════════════════════════════════════════════

class DistilBertRegressor(nn.Module):
    """DistilBERT with regression head."""

    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.regressor(pooled).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Training functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * input_ids.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    # Clip predictions if bounds are set
    preds_array = np.array(all_preds)
    if CLIP_MIN is not None:
        preds_array = np.maximum(preds_array, CLIP_MIN)
    if CLIP_MAX is not None:
        preds_array = np.minimum(preds_array, CLIP_MAX)

    rmse = np.sqrt(mean_squared_error(all_targets, preds_array))

    return avg_loss, rmse, preds_array


def predict(model, loader, device):
    """Generate predictions on test set."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.extend(outputs.cpu().numpy())

    # Clip predictions if bounds are set
    preds_array = np.array(all_preds)
    if CLIP_MIN is not None:
        preds_array = np.maximum(preds_array, CLIP_MIN)
    if CLIP_MAX is not None:
        preds_array = np.minimum(preds_array, CLIP_MAX)

    return preds_array


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict():
    """Train model with k-fold CV and generate predictions."""
    print(f"Using device: {DEVICE}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # EDITABLE: Column names
    text_col = "text"
    target_col = "target"
    id_col = "id"

    train_texts = train_df[text_col].values
    train_targets = train_df[target_col].values
    test_texts = test_df[text_col].values

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    cv_scores = []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_fold_texts = train_texts[train_idx]
        train_fold_targets = train_targets[train_idx]
        val_fold_texts = train_texts[val_idx]
        val_fold_targets = train_targets[val_idx]

        train_dataset = TextRegressionDataset(train_fold_texts, train_fold_targets, tokenizer, MAX_LENGTH)
        val_dataset = TextRegressionDataset(val_fold_texts, val_fold_targets, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = DistilBertRegressor(MODEL_NAME).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_rmse = float("inf")
        best_preds = None

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
            val_loss, val_rmse, val_preds = validate_epoch(model, val_loader, criterion, DEVICE)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_preds = val_preds
                best_model_state = model.state_dict().copy()

        # Store OOF predictions
        oof_preds[val_idx] = best_preds
        cv_scores.append(best_rmse)

        # Generate test predictions with best model
        model.load_state_dict(best_model_state)
        test_dataset = TextRegressionDataset(test_texts, None, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_preds += predict(model, test_loader, DEVICE) / N_FOLDS

        print(f"Fold {fold + 1} Best RMSE: {best_rmse:.4f}")

    print(f"\nCV Mean RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    return train_df, test_df, oof_preds, test_preds, cv_scores, id_col


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Submission generation
# ═══════════════════════════════════════════════════════════════════════════════

def create_submission(test_df, test_preds, id_col):
    """Create submission file."""
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        id_col: test_df[id_col],
        "target": test_preds
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    return submission


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    train_df, test_df, oof_preds, test_preds, cv_scores, id_col = train_and_predict()
    create_submission(test_df, test_preds, id_col)
    return np.mean(cv_scores)


if __name__ == "__main__":
    score = main()
    print(f"\nFinal CV RMSE: {score:.4f}")
