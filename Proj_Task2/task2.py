"""
============================================================================
TASK 2 - Ball Counting on 8-Ball Pool Tables (CNN-based)
============================================================================

Input : an image of an 8-ball pool table
Output: the TOTAL number of balls on the table

Approach:
    We frame Task 2 as a COUNTING problem (not detection -> that is Task 3).
    A CNN backbone pre-trained on ImageNet is fine-tuned with a small
    regression head that predicts a single scalar = number of balls.

Extra (required by the brief):
    Quantitative comparison of TWO different architectures with adequate
    metrics (MAE, RMSE, exact-count accuracy, +/-1 accuracy).

Ground truth:
    The dataset is in YOLO format. The number of balls in an image is simply
    the number of bounding boxes = number of lines in the label .txt file.

Libraries: only PyTorch, OpenCV, numpy, matplotlib (as allowed by the brief).

------------------------------------------------------------------------
Usage
------------------------------------------------------------------------
Train + evaluate + compare architectures:
    python3 task2.py --mode train

Run inference on a list of images and write the strict JSON output:
    python3 task2.py --mode predict \
        --input input.json --output output.json \
        --weights weights/resnet18_count.pth --arch resnet18

input.json  : ["path/img1.jpg", "path/img2.jpg", ...]
output.json : [{"image": "path/img1.jpg", "num_balls": 8}, ...]
============================================================================
"""

import os
import csv
import json
import random
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ==============================================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "dataset"          # expects dataset/{train,valid,test}/{images,labels}
IMG_SIZE = 224                    # standard size for ImageNet backbones
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
WEIGHTS_DIR = "weights"
SEED = 42

# ImageNet normalization stats (the backbones were pre-trained with these)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed=SEED):
    """Fix all RNG seeds so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. DATASET
# ==============================================================================
class PoolCountDataset(Dataset):
    """
    Loads pool-table images and returns (image_tensor, ball_count).

    The ball count is derived from the YOLO label file: each line is one box,
    so the number of lines = number of balls. Images with no label file are
    treated as empty tables (0 balls).
    """

    def __init__(self, split="train", augment=False):
        self.img_dir = os.path.join(DATASET_ROOT, split, "images")
        self.lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
        self.augment = augment
        self.images = [
            f for f in sorted(os.listdir(self.img_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def _count_balls(self, img_name):
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            return 0
        count = 0
        with open(lbl_path, "r") as f:
            for line in f:
                if len(line.strip().split()) >= 5:
                    count += 1
        return count

    def _augment(self, rgb):
        """Light, label-preserving augmentations (count does not change)."""
        # Horizontal flip
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
        # Vertical flip (a pool table is symmetric enough for this)
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 0)
        # Brightness / contrast jitter
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta = random.uniform(-20, 20)     # brightness
            rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
        return rgb

    def __getitem__(self, idx):
        img_name = self.images[idx]
        bgr = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        if self.augment:
            rgb = self._augment(rgb)

        # Normalize with ImageNet stats and convert to CHW tensor
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

        count = float(self._count_balls(img_name))
        return img_tensor, torch.tensor([count], dtype=torch.float32)


# ==============================================================================
# 3. MODELS (TWO ARCHITECTURES FOR COMPARISON)
# ==============================================================================
def build_model(arch="resnet18"):
    """
    Build a pre-trained CNN backbone with a small regression head that outputs
    a single scalar (the predicted ball count).
    """
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
        )
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


# ==============================================================================
# 4. METRICS
# ==============================================================================
def compute_metrics(preds, targets):
    """
    preds, targets: 1D numpy arrays of (rounded) counts.
    Returns MAE, RMSE, exact-count accuracy and +/-1 accuracy.
    """
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    abs_err = np.abs(preds - targets)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "acc_exact": float(np.mean(abs_err == 0)),
        "acc_within_1": float(np.mean(abs_err <= 1)),
    }


# ==============================================================================
# 5. TRAIN / EVALUATE
# ==============================================================================
@torch.no_grad()
def evaluate(model, loader):
    """Run the model over a loader and return metrics on integer-rounded counts."""
    model.eval()
    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(DEVICE)
        out = model(images).squeeze(1).cpu().numpy()
        preds = np.clip(np.round(out), 0, None)        # counts are >= 0 integers
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.squeeze(1).numpy().tolist())
    return compute_metrics(all_preds, all_targets), all_preds, all_targets


def train_model(arch, train_loader, val_loader, epochs=EPOCHS):
    """Train one architecture, keeping the best checkpoint by validation MAE."""
    print(f"\n========== Training {arch} ==========")
    model = build_model(arch).to(DEVICE)

    criterion = nn.SmoothL1Loss()   # robust regression loss (Huber)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_mae = float("inf")
    best_path = os.path.join(WEIGHTS_DIR, f"{arch}_count.pth")
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics, _, _ = evaluate(model, val_loader)
        history.append((epoch + 1, train_loss, val_metrics["mae"]))

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_MAE={val_metrics['mae']:.3f} "
            f"val_RMSE={val_metrics['rmse']:.3f} "
            f"val_acc={val_metrics['acc_exact']:.3f}"
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best (val_MAE={best_mae:.3f}) saved to {best_path}")

    return best_path, history


# ==============================================================================
# 6. INFERENCE (STRICT JSON I/O)
# ==============================================================================
def preprocess_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


@torch.no_grad()
def predict(input_json, output_json, weights_path, arch):
    """
    Read a list of image paths from input_json, predict the ball count for each,
    and write a list of results to output_json following the required structure.
    """
    with open(input_json, "r") as f:
        image_paths = json.load(f)

    model = build_model(arch).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    results = []
    for path in image_paths:
        img = preprocess_image(path).unsqueeze(0).to(DEVICE)
        out = model(img).item()
        count = int(max(0, round(out)))
        results.append({"image": path, "num_balls": count})

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {output_json}")


# ==============================================================================
# 7. REPORTING HELPERS
# ==============================================================================
def save_comparison(results_by_arch):
    """Print a comparison table and save it as CSV for the report."""
    print("\n================ ARCHITECTURE COMPARISON (test set) ================")
    header = f"{'arch':<18}{'MAE':>8}{'RMSE':>8}{'acc':>8}{'acc±1':>8}"
    print(header)
    print("-" * len(header))
    os.makedirs("reports", exist_ok=True)
    with open("reports/task2_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arch", "mae", "rmse", "acc_exact", "acc_within_1"])
        for arch, m in results_by_arch.items():
            print(f"{arch:<18}{m['mae']:>8.3f}{m['rmse']:>8.3f}"
                  f"{m['acc_exact']:>8.3f}{m['acc_within_1']:>8.3f}")
            writer.writerow([arch, m["mae"], m["rmse"], m["acc_exact"], m["acc_within_1"]])
    print("Saved reports/task2_comparison.csv")


def save_scatter(arch, preds, targets):
    """Scatter of predicted vs true counts -> a nice figure for the report."""
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, alpha=0.5)
    lim = max(max(targets), max(preds)) + 1
    plt.plot([0, lim], [0, lim], "r--", label="perfect")
    plt.xlabel("True count")
    plt.ylabel("Predicted count")
    plt.title(f"{arch}: predicted vs true")
    plt.legend()
    plt.tight_layout()
    out = f"reports/{arch}_scatter.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


# ==============================================================================
# 8. MAIN
# ==============================================================================
def run_training():
    set_seed()
    print(f"Using device: {DEVICE}")

    train_ds = PoolCountDataset("train", augment=True)
    val_ds = PoolCountDataset("valid", augment=False)
    test_ds = PoolCountDataset("test", augment=False)
    print(f"train={len(train_ds)}  valid={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    architectures = ["resnet18", "efficientnet_b0"]
    results_by_arch = {}

    for arch in architectures:
        best_path, _ = train_model(arch, train_loader, val_loader)
        # Reload best checkpoint and evaluate on the held-out test split
        model = build_model(arch).to(DEVICE)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        test_metrics, preds, targets = evaluate(model, test_loader)
        results_by_arch[arch] = test_metrics
        save_scatter(arch, preds, targets)

    save_comparison(results_by_arch)


def main():
    parser = argparse.ArgumentParser(description="Task 2 - CNN ball counting")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--input", default="input.json")
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--weights", default="weights/resnet18_count.pth")
    parser.add_argument("--arch", default="resnet18",
                        choices=["resnet18", "efficientnet_b0"])
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    else:
        predict(args.input, args.output, args.weights, args.arch)


if __name__ == "__main__":
    main()
