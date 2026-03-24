"""
Baseline template for image classification problems.
Uses timm ResNet18 pretrained with basic augmentation.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import timm
from torchvision import transforms

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Do not modify these paths
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submissions/submission.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model and training configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "resnet18"
PRETRAINED = True
NUM_CLASSES = 2  # EDITABLE: Set based on competition
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Data augmentation
# ═══════════════════════════════════════════════════════════════════════════════

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Dataset class
# ═══════════════════════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_test = is_test

        # EDITABLE: Modify column names based on competition
        self.image_col = "image_id"  # or "filename", "id", etc.
        self.label_col = "label"  # or "target", "class", etc.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # EDITABLE: Modify image path construction based on competition
        img_name = row[self.image_col]
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_name = f"{img_name}.jpg"
        img_path = self.image_dir / img_name

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image

        label = row[self.label_col]
        return image, label


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model definition
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(num_classes):
    """Create a timm model with custom head."""
    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=num_classes)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Training functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            if NUM_CLASSES == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    if NUM_CLASSES == 2:
        score = roc_auc_score(all_labels, all_preds)
    else:
        score = accuracy_score(all_labels, np.argmax(all_preds, axis=1))

    return avg_loss, score, np.array(all_preds)


def predict(model, loader, device):
    """Generate predictions on test set."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)

            if NUM_CLASSES == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy())

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

    # EDITABLE: Modify paths based on competition structure
    train_image_dir = DATA_DIR / "train_images"
    test_image_dir = DATA_DIR / "test_images"

    # EDITABLE: Column names
    label_col = "label"
    id_col = "image_id"

    if NUM_CLASSES == 2:
        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
    else:
        oof_preds = np.zeros((len(train_df), NUM_CLASSES))
        test_preds = np.zeros((len(test_df), NUM_CLASSES))

    cv_scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[label_col])):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = ImageDataset(train_fold_df, train_image_dir, train_transforms)
        val_dataset = ImageDataset(val_fold_df, train_image_dir, val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = create_model(NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_score = 0.0
        best_preds = None

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_score, val_preds = validate_epoch(model, val_loader, criterion, DEVICE)
            scheduler.step()

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
        test_dataset = ImageDataset(test_df, test_image_dir, val_transforms, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
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
