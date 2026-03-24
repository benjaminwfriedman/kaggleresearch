"""
Baseline template for image segmentation problems.
Uses segmentation_models_pytorch Unet with EfficientNet-b0 encoder.
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
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Do not modify these paths
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submissions/submission.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model and training configuration
# ═══════════════════════════════════════════════════════════════════════════════

ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 1  # EDITABLE: 1 for binary segmentation, >1 for multiclass
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Data augmentation (using albumentations)
# ═══════════════════════════════════════════════════════════════════════════════

train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Dataset class
# ═══════════════════════════════════════════════════════════════════════════════

class SegmentationDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir=None, transform=None, is_test=False):
        self.df = df
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.is_test = is_test

        # EDITABLE: Modify column names based on competition
        self.image_col = "image_id"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_name = row[self.image_col]
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_name = f"{img_name}.png"
        img_path = self.image_dir / img_name
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image

        # Load mask
        mask_name = row[self.image_col]
        if not mask_name.endswith(('.jpg', '.png', '.jpeg')):
            mask_name = f"{mask_name}.png"
        mask_path = self.mask_dir / mask_name
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # Binary mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask

        return image, mask


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Model definition
# ═══════════════════════════════════════════════════════════════════════════════

def create_model():
    """Create a segmentation model using SMP."""
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Loss and metrics
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + (1 - self.bce_weight) * self.dice(pred, target)


def dice_score(pred, target, threshold=0.5):
    """Calculate Dice score."""
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Training functions
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            running_dice += dice_score(outputs, masks).item() * images.size(0)

    avg_loss = running_loss / len(loader.dataset)
    avg_dice = running_dice / len(loader.dataset)

    return avg_loss, avg_dice


def predict(model, loader, device):
    """Generate predictions on test set."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# EDITABLE — Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict():
    """Train model with k-fold CV and generate predictions."""
    print(f"Using device: {DEVICE}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # EDITABLE: Modify paths based on competition structure
    train_image_dir = DATA_DIR / "train_images"
    train_mask_dir = DATA_DIR / "train_masks"
    test_image_dir = DATA_DIR / "test_images"

    id_col = "image_id"

    cv_scores = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Store test predictions
    test_preds_sum = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

        train_dataset = SegmentationDataset(train_fold_df, train_image_dir, train_mask_dir, train_transforms)
        val_dataset = SegmentationDataset(val_fold_df, train_image_dir, train_mask_dir, val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = create_model().to(DEVICE)
        criterion = CombinedLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_dice = 0.0

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_dice = validate_epoch(model, val_loader, criterion, DEVICE)
            scheduler.step()

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                best_model_state = model.state_dict().copy()

        cv_scores.append(best_dice)

        # Generate test predictions with best model
        model.load_state_dict(best_model_state)
        test_dataset = SegmentationDataset(test_df, test_image_dir, transform=val_transforms, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        fold_preds = predict(model, test_loader, DEVICE)

        if test_preds_sum is None:
            test_preds_sum = fold_preds
        else:
            test_preds_sum += fold_preds

        print(f"Fold {fold + 1} Best Dice: {best_dice:.4f}")

    test_preds = test_preds_sum / N_FOLDS

    print(f"\nCV Mean Dice: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    return test_df, test_preds, cv_scores, id_col


# ═══════════════════════════════════════════════════════════════════════════════
# FIXED — Submission generation
# ═══════════════════════════════════════════════════════════════════════════════

def rle_encode(mask, threshold=0.5):
    """Run-length encoding for submission."""
    mask = (mask > threshold).flatten()
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submission(test_df, test_preds, id_col):
    """Create submission file with RLE encoding."""
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    rle_masks = []
    for i in range(len(test_preds)):
        mask = test_preds[i, 0]  # First channel
        rle = rle_encode(mask)
        rle_masks.append(rle)

    submission = pd.DataFrame({
        id_col: test_df[id_col],
        "rle_mask": rle_masks
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    return submission


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    test_df, test_preds, cv_scores, id_col = train_and_predict()
    create_submission(test_df, test_preds, id_col)
    return np.mean(cv_scores)


if __name__ == "__main__":
    score = main()
    print(f"\nFinal CV Dice: {score:.4f}")
