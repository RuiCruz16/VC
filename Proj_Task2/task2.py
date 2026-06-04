"""
TASK 2 - Ball Counting (CNN-based)

--- Usage ---

Train + evaluate + compare architectures:
    python3 task2.py --mode train

Run inference on a list of images and write the strict JSON output:
    python3 task2.py --mode predict \
        --input input.json --output output.json \
        --weights weights/resnet18_count.pth --arch resnet18

input.json  : ["path/img1.jpg", "path/img2.jpg", ...]
output.json : [{"image": "path/img1.jpg", "num_balls": 8}, ...]
"""

import os

# Force ROCm compatibility with the AMD RX 6600 architecture
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Disable SDMA transfer to prevent PCIe memory segmentation faults
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

import csv
import json
import random
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Disable OpenCV's internal threading to prevent segmentation faults in PyTorch's DataLoader workers
cv2.setNumThreads(0)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# CONFIGURATION & REPRODUCIBILITY
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "dataset" # expects dataset/{train,valid,test}/{images,labels}
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100 # upper bound; early stopping usually stops earlier
LR = 1e-4 # small learning rate chosen to safely fine-tune the pre-trained models
WEIGHTS_DIR = "weights"
REPORTS_DIR = "reports"
SEED = 42

# Early stopping: stop if val_MAE does not improve for `PATIENCE` epochs.
PATIENCE = 12
MIN_DELTA = 1e-3 # minimum val_MAE improvement to count as "better"

USE_AMP = torch.cuda.is_available() # Enables Automatic Mixed Precision (16-bit) to halve VRAM usage and speed up GPU training
USE_TTA = True # Uses Test-Time Augmentation (horizontal flipping) to improve prediction accuracy during inference

# ImageNet normalization stats (the backbones were pre-trained with these)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _make_loader(dataset, shuffle):
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=(NUM_WORKERS > 0),
        pin_memory=torch.cuda.is_available(),
    )

# DATASET
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

    def counts(self):
        """Return the list of ball counts (useful for dataset-distribution plots)."""
        return [self._count_balls(n) for n in self.images]

    def _augment(self, rgb):
        # Randomly apply horizontal flip
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
        
        # Randomly apply vertical flip since pool tables are symmetric
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 0)
        
        # Apply random rotation (±10°) using border replication to keep balls in frame
        if random.random() < 0.5:
            h, w = rgb.shape[:2]
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Adjust brightness and contrast to simulate different lighting conditions
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
        return rgb

    def __getitem__(self, idx):
        img_name = self.images[idx]
        bgr = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)
        if bgr is None:
            # Never crash on a corrupt/unreadable image.
            bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
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

# MODELS
class SimpleCNN(nn.Module):
    """
    CNN trained from scratch (no pre-training). Serves as a baseline to
    quantify how much ImageNet transfer learning actually helps.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))

def build_model(arch="resnet18"):
    """
    Build a CNN with a small regression head that outputs a single scalar
    (the predicted ball count).
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
    elif arch == "simplecnn":
        model = SimpleCNN()
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

# METRICS
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

# TRAIN / EVALUATE
@torch.no_grad()
def evaluate(model, loader, criterion=None):
    """
    Run the model over a loader and return (metrics, preds, targets, val_loss).
    `val_loss` is computed with the same criterion as training, so it is
    directly comparable to train_loss for overfitting diagnosis.
    """
    model.eval()
    all_preds, all_targets = [], []
    loss_sum, n = 0.0, 0
    for images, targets in loader:
        images = images.to(DEVICE)
        targets_dev = targets.to(DEVICE)
        out = model(images)
        if criterion is not None:
            loss_sum += criterion(out, targets_dev).item() * images.size(0)
            n += images.size(0)
        out = out.squeeze(1).cpu().numpy()
        preds = np.clip(np.round(out), 0, None) # counts are >= 0 integers
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.squeeze(1).numpy().tolist())
    val_loss = (loss_sum / n) if n > 0 else float("nan")
    return compute_metrics(all_preds, all_targets), all_preds, all_targets, val_loss

def train_model(arch, train_loader, val_loader, epochs=EPOCHS):
    """
    Trains the model with mixed precision, CSV logging, 
    and early stopping on validation MAE, returning 
    the best checkpoint path and history.
    """
    print(f"\n========== Training {arch} ==========")
    model = build_model(arch).to(DEVICE)

    criterion = nn.SmoothL1Loss() # robust regression loss (Huber)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    best_mae = float("inf")
    best_epoch = 0
    best_path = os.path.join(WEIGHTS_DIR, f"{arch}_count.pth")
    log_path = os.path.join(REPORTS_DIR, f"{arch}_train_log.csv")
    history = []
    epochs_no_improve = 0

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss",
                         "val_mae", "val_rmse", "val_acc", "val_acc_within_1"])

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, targets in train_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
            scheduler.step()

            train_loss = running_loss / len(train_loader.dataset)
            val_metrics, _, _, val_loss = evaluate(model, val_loader, criterion)
            lr_now = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch + 1,
                "lr": lr_now,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_acc": val_metrics["acc_exact"],
                "val_acc_within_1": val_metrics["acc_within_1"],
            }
            history.append(row)
            writer.writerow([row["epoch"], f"{lr_now:.2e}", f"{train_loss:.5f}",
                             f"{val_loss:.5f}", f"{val_metrics['mae']:.5f}",
                             f"{val_metrics['rmse']:.5f}", f"{val_metrics['acc_exact']:.5f}",
                             f"{val_metrics['acc_within_1']:.5f}"])
            log_file.flush()

            # Positive gap means val_loss > train_loss (overfitting)
            gap = val_loss - train_loss
            print(
                f"Epoch [{epoch+1:03d}/{epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"(gap={gap:+.4f}) | "
                f"val_MAE={val_metrics['mae']:.3f} "
                f"val_RMSE={val_metrics['rmse']:.3f} "
                f"val_acc={val_metrics['acc_exact']:.3f}"
            )

            # Checkpoint + early stopping bookkeeping (best by val_MAE)
            if val_metrics["mae"] < best_mae - MIN_DELTA:
                best_mae = val_metrics["mae"]
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  -> new best (val_MAE={best_mae:.3f}) saved to {best_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"  -> early stopping: no val_MAE improvement for "
                          f"{PATIENCE} epochs (best epoch={best_epoch}, "
                          f"best val_MAE={best_mae:.3f}).")
                    break

    save_loss_curves(arch, history, best_epoch)
    print(f"Saved per-epoch log to {log_path}")
    return best_path, history

# INFERENCE
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
        if USE_TTA:
            flipped = torch.flip(img, dims=[3]) # horizontal flip
            out = 0.5 * (out + model(flipped).item())
        count = int(max(0, round(out)))
        results.append({"image": path, "num_balls": count})

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {output_json}")

# REPORTING HELPERS
def save_loss_curves(arch, history, best_epoch):
    """
    Plot train_loss vs val_loss (left) and val_MAE (right). The point where the
    two loss curves start to diverge is the overfitting onset.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_mae = [h["val_mae"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    if best_epoch:
        ax1.axvline(best_epoch, color="green", ls="--", alpha=0.7,
                    label=f"best (ep {best_epoch})")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss (SmoothL1)")
    ax1.set_title(f"{arch}: train vs val loss"); ax1.legend()

    ax2.plot(epochs, val_mae, color="purple", label="val_MAE")
    if best_epoch:
        ax2.axvline(best_epoch, color="green", ls="--", alpha=0.7)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val MAE")
    ax2.set_title(f"{arch}: validation MAE"); ax2.legend()

    fig.tight_layout()
    out = os.path.join(REPORTS_DIR, f"{arch}_curves.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")

def save_comparison(results_by_arch):
    """Print a comparison table and save it as CSV."""
    print("\n================ ARCHITECTURE COMPARISON (test set) ================")
    header = f"{'arch':<18}{'MAE':>8}{'RMSE':>8}{'acc':>8}{'acc±1':>8}"
    print(header)
    print("-" * len(header))
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "task2_comparison.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arch", "mae", "rmse", "acc_exact", "acc_within_1"])
        for arch, m in results_by_arch.items():
            print(f"{arch:<18}{m['mae']:>8.3f}{m['rmse']:>8.3f}"
                  f"{m['acc_exact']:>8.3f}{m['acc_within_1']:>8.3f}")
            writer.writerow([arch, m["mae"], m["rmse"], m["acc_exact"], m["acc_within_1"]])
    print(f"Saved {REPORTS_DIR}/task2_comparison.csv")

def save_scatter(arch, preds, targets):
    """Scatter of predicted vs true counts."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, alpha=0.5)
    lim = max(max(targets), max(preds)) + 1
    plt.plot([0, lim], [0, lim], "r--", label="perfect")
    plt.xlabel("True count")
    plt.ylabel("Predicted count")
    plt.title(f"{arch}: predicted vs true")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, f"{arch}_scatter.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")

def save_error_by_count(arch, preds, targets):
    """
    Qualitative analysis: break the error down by the true ball count, so we can
    see where the model struggles most (e.g. it may nail 8-ball but miss when the
    table is crowded). Saves a per-count table as CSV and prints it.

    For each distinct true count it reports:
        n          -> how many test images have that true count
        mae        -> mean absolute error for those images
        acc_exact  -> fraction predicted exactly right
    """
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_csv = os.path.join(REPORTS_DIR, f"{arch}_error_by_count.csv")

    print(f"\n----- {arch}: error breakdown by true count -----")
    header = f"{'true_count':>11}{'n':>6}{'mae':>9}{'acc_exact':>11}"
    print(header)
    print("-" * len(header))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_count", "n", "mae", "acc_exact"])
        for c in sorted(np.unique(targets)):
            mask = targets == c
            n = int(mask.sum())
            abs_err = np.abs(preds[mask] - targets[mask])
            mae_c = float(np.mean(abs_err))
            acc_c = float(np.mean(abs_err == 0))
            print(f"{int(c):>11}{n:>6}{mae_c:>9.3f}{acc_c:>11.3f}")
            writer.writerow([int(c), n, f"{mae_c:.4f}", f"{acc_c:.4f}"])
    print(f"Saved {out_csv}")

def save_count_distribution(train_ds):
    """Histogram of ball counts in the training set."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    counts = train_ds.counts()
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=range(0, max(counts) + 2), align="left", rwidth=0.85)
    plt.xlabel("ball count")
    plt.ylabel("number of images")
    plt.title("Training set: distribution of ball counts")
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "train_count_distribution.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")

# MAIN
def run_training():
    set_seed()
    print(f"Using device: {DEVICE}  (AMP={USE_AMP})")

    train_ds = PoolCountDataset("train", augment=True)
    val_ds = PoolCountDataset("valid", augment=False)
    test_ds = PoolCountDataset("test", augment=False)
    print(f"train={len(train_ds)}  valid={len(val_ds)}  test={len(test_ds)}")

    save_count_distribution(train_ds)

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    architectures = ["resnet18", "efficientnet_b0", "simplecnn"]
    results_by_arch = {}

    for arch in architectures:
        best_path, _ = train_model(arch, train_loader, val_loader)
        # Reload best checkpoint and evaluate on the held-out test split
        model = build_model(arch).to(DEVICE)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        test_metrics, preds, targets, _ = evaluate(model, test_loader)
        results_by_arch[arch] = test_metrics
        save_scatter(arch, preds, targets)
        save_error_by_count(arch, preds, targets)

    save_comparison(results_by_arch)

def main():
    parser = argparse.ArgumentParser(description="Task 2 - CNN ball counting")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--input", default="input.json")
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--weights", default="weights/resnet18_count.pth")
    parser.add_argument("--arch", default="resnet18",
                        choices=["resnet18", "efficientnet_b0", "simplecnn"])
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    else:
        predict(args.input, args.output, args.weights, args.arch)

if __name__ == "__main__":
    main()
