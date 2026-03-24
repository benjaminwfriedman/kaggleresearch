"""
Baseline template for NLP classification problems.
Uses DistilBERT fine-tuning with HuggingFace Transformers.
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
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Do not modify these paths
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submissions/submission.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model and training configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "distilbert-base-uncased"
NUM_CLASSES = 2  # EDITABLE: Set based on competition
MAX_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Dataset class
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
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

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Training functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            running_loss += loss.item() * input_ids.size(0)

            if NUM_CLASSES == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
            else:
                all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    if NUM_CLASSES == 2:
        score = roc_auc_score(all_labels, all_preds)
    else:
        preds_class = np.argmax(all_preds, axis=1)
        score = f1_score(all_labels, preds_class, average="macro")

    return avg_loss, score, np.array(all_preds)


def predict(model, loader, device):
    """Generate predictions on test set."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            if NUM_CLASSES == 2:
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
            else:
                all_preds.extend(logits.cpu().numpy())

    return np.array(all_preds)


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict():
    """Train model with stratified k-fold CV and generate predictions."""
    print(f"Using device: {DEVICE}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # EDITABLE: Column names
    text_col = "text"
    label_col = "label"
    id_col = "id"

    train_texts = train_df[text_col].values
    train_labels = train_df[label_col].values
    test_texts = test_df[text_col].values

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    if NUM_CLASSES == 2:
        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
    else:
        oof_preds = np.zeros((len(train_df), NUM_CLASSES))
        test_preds = np.zeros((len(test_df), NUM_CLASSES))

    cv_scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_fold_texts = train_texts[train_idx]
        train_fold_labels = train_labels[train_idx]
        val_fold_texts = train_texts[val_idx]
        val_fold_labels = train_labels[val_idx]

        train_dataset = TextDataset(train_fold_texts, train_fold_labels, tokenizer, MAX_LENGTH)
        val_dataset = TextDataset(val_fold_texts, val_fold_labels, tokenizer, MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_CLASSES
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_score = 0.0
        best_preds = None

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            val_loss, val_score, val_preds = validate_epoch(model, val_loader, DEVICE)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")

            if val_score > best_score:
                best_score = val_score
                best_preds = val_preds
                best_model_state = model.state_dict().copy()

        # Store OOF predictions
        oof_preds[val_idx] = best_preds
        cv_scores.append(best_score)

        # Generate test predictions with best model
        model.load_state_dict(best_model_state)
        test_dataset = TextDataset(test_texts, None, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_preds += predict(model, test_loader, DEVICE) / N_FOLDS

        print(f"Fold {fold + 1} Best Score: {best_score:.4f}")

    print(f"\nCV Mean Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    return train_df, test_df, oof_preds, test_preds, cv_scores, id_col


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Submission generation
# ═══════════════════════════════════════════════════════════════════════════════

def create_submission(test_df, test_preds, id_col):
    """Create submission file."""
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    if NUM_CLASSES == 2:
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            "target": test_preds
        })
    else:
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            "target": np.argmax(test_preds, axis=1)
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
    print(f"\nFinal CV Score: {score:.4f}")
