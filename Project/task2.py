"""
Task 2 — Ball Counting via Deep Learning
========================================

Self-contained PyTorch script that trains/evaluates/predicts the number of
pool balls on an 8-ball pool table. Supports three CNN-based approaches
(for the extra-credit comparison):

    cls : classification head over 0..MAX_BALLS classes  (ResNet18)
    reg : regression head predicting the count as a scalar (ResNet18)
    det : Faster R-CNN (single class "ball") — count = number of detections

Training data
-------------
Primary: Roboflow `8-ball-pool-l530o` exported in **YOLOv8** format. Download
it manually from
https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o
and unzip into `./roboflow/` so the layout is:

    roboflow/
        data.yaml
        train/{images,labels}/
        valid/{images,labels}/
        test/{images,labels}/

Each `labels/*.txt` follows YOLO format:
    <class_id> <cx> <cy> <w> <h>      (all normalised)

The script treats every annotated object as a ball regardless of its class id
(so the original solid/stripe/cue/8-ball classes collapse to a single "ball"
super-class for counting/detection). The per-image count is simply the number
of lines in the label file.

Usage
-----
Train every model and write a comparison table:

    python task2.py train --data roboflow --epochs 8

Predict on the project's input JSON (default: input.json -> output.json):

    python task2.py predict --model cls --weights weights/cls.pt

CLI flags fall back to sensible defaults — running `python task2.py predict`
with no arguments will use `weights/cls.pt` and overwrite `output.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
MAX_BALLS = 16            # 8-ball pool: 15 object balls + cue ball = 16 max
NUM_CLASSES_CLS = MAX_BALLS + 1   # 0..16 inclusive -> 17 classes
IMG_SIZE = 224
DET_IMG_SIZE = 512
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset helpers                                                             #
# --------------------------------------------------------------------------- #
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Substrings (case-insensitive) of class names that should NOT be counted as
# balls (e.g. table pocket markers, cue sticks, hands, table surface).
NON_BALL_NAME_KEYWORDS = ("dot", "pocket", "stick", "hand", "table",
                          "player", "chalk")

# A Sample carries its own non-ball class-id set so multiple datasets with
# different class orderings can be mixed safely.
Sample = Tuple[Path, Path, frozenset]  # (image, label_or_None, non_ball_ids)


def _parse_class_names(data_yaml: Path) -> List[str]:
    """Tiny parser for the `names:` field of a Roboflow data.yaml.

    Supports both inline list form (`names: ['a', 'b']`) and block form
    (`names:\\n  - a\\n  - b`). Returns [] if not found.
    """
    if not data_yaml.exists():
        return []
    text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    # Inline list: names: ['a', "b", c]
    import re
    m = re.search(r"^\s*names\s*:\s*\[(.*?)\]\s*$", text, re.MULTILINE | re.DOTALL)
    if m:
        items = re.findall(r"['\"]?([^,'\"\[\]]+?)['\"]?(?:,|$)", m.group(1))
        return [s.strip() for s in items if s.strip()]
    # Block list: names:\n  - a\n  - b
    m = re.search(r"^\s*names\s*:\s*$(.*?)(?=^\S|\Z)", text,
                  re.MULTILINE | re.DOTALL)
    if m:
        return [ln.strip().lstrip("-").strip().strip("'\"")
                for ln in m.group(1).splitlines()
                if ln.strip().startswith("-")]
    return []


def _non_ball_ids_for(root: Path) -> frozenset:
    """Auto-detect non-ball class indices from a dataset's data.yaml."""
    names = _parse_class_names(root / "data.yaml")
    if not names:
        return frozenset()
    bad = {i for i, n in enumerate(names)
           if any(kw in n.lower() for kw in NON_BALL_NAME_KEYWORDS)}
    return frozenset(bad)


def _collect_yolo_pairs(root: Path, split: str) -> List[Sample]:
    """Return list of Samples for a YOLO split of one dataset root."""
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.is_dir():
        return []
    exclude = _non_ball_ids_for(root)
    pairs: List[Sample] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            # Image with no annotations -> empty table (0 balls)
            lbl = None  # type: ignore[assignment]
        pairs.append((img, lbl, exclude))  # type: ignore[arg-type]
    return pairs


def _collect_split(roots: List[Path], split: str,
                   fallback: str | None = None) -> List[Sample]:
    """Merge a split across multiple dataset roots."""
    out: List[Sample] = []
    for r in roots:
        part = _collect_yolo_pairs(r, split)
        if not part and fallback:
            part = _collect_yolo_pairs(r, fallback)
        out.extend(part)
    return out


def _read_yolo_label(path: Path | None,
                     exclude: frozenset = frozenset()) -> np.ndarray:
    """Return Nx5 array (cls, cx, cy, w, h) or empty (0,5)."""
    if path is None or not path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows: List[List[float]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(float(parts[0]))
            if cls_id in exclude:
                continue
            rows.append([float(x) for x in parts[:5]])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Classification / regression dataset                                         #
# --------------------------------------------------------------------------- #
class BallCountDataset(Dataset):
    """Returns (image_tensor, count) — used for classification & regression."""

    def __init__(self, pairs: List[Tuple[Path, Path]], train: bool):
        self.pairs = pairs
        self.train = train
        if train:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_path, lbl_path, exclude = self.pairs[i]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        labels = _read_yolo_label(lbl_path, exclude)
        count = min(len(labels), MAX_BALLS)
        return self.tf(rgb), torch.tensor(count, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Detection dataset                                                           #
# --------------------------------------------------------------------------- #
class BallDetectionDataset(Dataset):
    """Returns (image_tensor, target_dict) for torchvision detection API."""

    def __init__(self, pairs: List[Tuple[Path, Path]], train: bool):
        self.pairs = pairs
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_path, lbl_path, exclude = self.pairs[i]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        rgb = cv2.resize(rgb, (DET_IMG_SIZE, DET_IMG_SIZE))
        if self.train and random.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
            flipped = True
        else:
            flipped = False

        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        labels = _read_yolo_label(lbl_path, exclude)
        boxes = []
        for _cls, cx, cy, w, h in labels:
            if flipped:
                cx = 1.0 - cx
            x1 = (cx - w / 2) * DET_IMG_SIZE
            y1 = (cy - h / 2) * DET_IMG_SIZE
            x2 = (cx + w / 2) * DET_IMG_SIZE
            y2 = (cy + h / 2) * DET_IMG_SIZE
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(DET_IMG_SIZE - 1, x2), min(DET_IMG_SIZE - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([i]),
        }
        return img_t, target


def detection_collate(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #
def build_classifier() -> nn.Module:
    """ResNet18 with a 17-way softmax head over 0..MAX_BALLS."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES_CLS)
    return m


def build_regressor() -> nn.Module:
    """ResNet18 with a single-scalar regression head."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 1))
    return m


def build_detector(num_classes: int = 2) -> nn.Module:
    """Faster R-CNN with 2 classes (background + ball)."""
    m = fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return m


# --------------------------------------------------------------------------- #
# Training loops                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    name: str
    accuracy: float        # exact-count accuracy
    mae: float             # mean absolute error on count
    within_one: float      # fraction predicted within ±1
    train_time_sec: float
    eval_time_sec: float


def train_cls(model: nn.Module, dl_tr: DataLoader, dl_va: DataLoader,
              epochs: int, lr: float) -> None:
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        sched.step()
        va_acc = _quick_cls_acc(model, dl_va)
        print(f"  [cls] epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  "
              f"train_acc={correct/total:.3f}  val_acc={va_acc:.3f}")


def train_reg(model: nn.Module, dl_tr: DataLoader, dl_va: DataLoader,
              epochs: int, lr: float) -> None:
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        total, loss_sum = 0, 0.0
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE).float()
            pred = model(x).squeeze(1)
            loss = F.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
            total += x.size(0)
        sched.step()
        va_mae = _quick_reg_mae(model, dl_va)
        print(f"  [reg] epoch {ep+1}/{epochs}  loss={loss_sum/total:.4f}  "
              f"val_mae={va_mae:.3f}")


def train_det(model: nn.Module, dl_tr: DataLoader, epochs: int, lr: float) -> None:
    model.to(DEVICE)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for imgs, targets in dl_tr:
            imgs = [im.to(DEVICE) for im in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); n += 1
        sched.step()
        print(f"  [det] epoch {ep+1}/{epochs}  loss={loss_sum/max(n,1):.4f}")


@torch.no_grad()
def _quick_cls_acc(model: nn.Module, dl: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def _quick_reg_mae(model: nn.Module, dl: DataLoader) -> float:
    model.eval()
    errs, total = 0.0, 0
    for x, y in dl:
        x = x.to(DEVICE)
        pred = model(x).squeeze(1).cpu().numpy()
        pred = np.clip(np.round(pred), 0, MAX_BALLS)
        errs += float(np.abs(pred - y.numpy()).sum())
        total += x.size(0)
    return errs / max(total, 1)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_counter(name: str, predict_fn, pairs: List[Sample],
                     train_time: float) -> EvalResult:
    t0 = time.time()
    preds, gts = [], []
    for img_p, lbl_p, exclude in pairs:
        gt = min(len(_read_yolo_label(lbl_p, exclude)), MAX_BALLS)
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        preds.append(predict_fn(rgb))
        gts.append(gt)
    preds_a = np.asarray(preds); gts_a = np.asarray(gts)
    return EvalResult(
        name=name,
        accuracy=float((preds_a == gts_a).mean()),
        mae=float(np.abs(preds_a - gts_a).mean()),
        within_one=float((np.abs(preds_a - gts_a) <= 1).mean()),
        train_time_sec=train_time,
        eval_time_sec=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# Per-image predictors                                                        #
# --------------------------------------------------------------------------- #
_INFER_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@torch.no_grad()
def predict_cls(model: nn.Module, rgb: np.ndarray) -> int:
    model.eval()
    x = _INFER_TF(rgb).unsqueeze(0).to(DEVICE)
    return int(model(x).argmax(1).item())


@torch.no_grad()
def predict_reg(model: nn.Module, rgb: np.ndarray) -> int:
    model.eval()
    x = _INFER_TF(rgb).unsqueeze(0).to(DEVICE)
    v = float(model(x).squeeze().item())
    return int(np.clip(round(v), 0, MAX_BALLS))


@torch.no_grad()
def predict_det(model: nn.Module, rgb: np.ndarray, score_thr: float = 0.5) -> int:
    model.eval()
    img = cv2.resize(rgb, (DET_IMG_SIZE, DET_IMG_SIZE))
    t = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255.0
    out = model([t])[0]
    return int((out["scores"] >= score_thr).sum().item())


# --------------------------------------------------------------------------- #
# Train command                                                               #
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> None:
    set_seed()
    data_roots = [Path(p) for p in args.data]
    missing = [str(p) for p in data_roots if not p.is_dir()]
    if missing:
        raise SystemExit(f"Dataset root(s) not found: {missing}")

    for r in data_roots:
        names = _parse_class_names(r / "data.yaml")
        excl = _non_ball_ids_for(r)
        excl_names = [names[i] for i in sorted(excl)] if names else []
        print(f"Dataset {r}: classes={names or '?'}  excluded={excl_names}")

    train_pairs = _collect_split(data_roots, "train")
    val_pairs = _collect_split(data_roots, "valid", fallback="val")
    test_pairs = _collect_split(data_roots, "test") or val_pairs

    if not train_pairs:
        raise SystemExit("No training images found across the provided roots")
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  Test: {len(test_pairs)}")

    weights_dir = Path(args.weights_dir); weights_dir.mkdir(exist_ok=True)

    models_to_train = args.models.split(",") if args.models else ["cls", "reg", "det"]
    results: List[EvalResult] = []

    if "cls" in models_to_train:
        print("\n=== Training classification model ===")
        dl_tr = DataLoader(BallCountDataset(train_pairs, train=True),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers)
        dl_va = DataLoader(BallCountDataset(val_pairs, train=False),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers)
        model = build_classifier()
        t0 = time.time()
        train_cls(model, dl_tr, dl_va, epochs=args.epochs, lr=args.lr)
        tt = time.time() - t0
        torch.save(model.state_dict(), weights_dir / "cls.pt")
        results.append(evaluate_counter(
            "cls", lambda rgb, m=model: predict_cls(m, rgb), test_pairs, tt))

    if "reg" in models_to_train:
        print("\n=== Training regression model ===")
        dl_tr = DataLoader(BallCountDataset(train_pairs, train=True),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers)
        dl_va = DataLoader(BallCountDataset(val_pairs, train=False),
                           batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers)
        model = build_regressor()
        t0 = time.time()
        train_reg(model, dl_tr, dl_va, epochs=args.epochs, lr=args.lr)
        tt = time.time() - t0
        torch.save(model.state_dict(), weights_dir / "reg.pt")
        results.append(evaluate_counter(
            "reg", lambda rgb, m=model: predict_reg(m, rgb), test_pairs, tt))

    if "det" in models_to_train:
        print("\n=== Training detection model (this is slow on CPU) ===")
        dl_tr = DataLoader(BallDetectionDataset(train_pairs, train=True),
                           batch_size=max(args.batch_size // 4, 1),
                           shuffle=True, num_workers=args.workers,
                           collate_fn=detection_collate)
        model = build_detector()
        t0 = time.time()
        train_det(model, dl_tr, epochs=max(1, args.epochs // 2), lr=args.lr * 0.1)
        tt = time.time() - t0
        torch.save(model.state_dict(), weights_dir / "det.pt")
        results.append(evaluate_counter(
            "det", lambda rgb, m=model: predict_det(m, rgb), test_pairs, tt))

    # ------------------------------------------------------------------ #
    # Comparison table (extra credit)                                    #
    # ------------------------------------------------------------------ #
    print("\n=== Model comparison on test split ===")
    header = f"{'model':<6} {'acc':>7} {'MAE':>7} {'±1 acc':>8} {'train(s)':>10} {'eval(s)':>9}"
    print(header); print("-" * len(header))
    for r in results:
        print(f"{r.name:<6} {r.accuracy:>7.3f} {r.mae:>7.3f} "
              f"{r.within_one:>8.3f} {r.train_time_sec:>10.1f} {r.eval_time_sec:>9.1f}")

    with open(weights_dir / "comparison.json", "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print(f"\nSaved weights and comparison to {weights_dir}/")


# --------------------------------------------------------------------------- #
# Predict command                                                             #
# --------------------------------------------------------------------------- #
def cmd_predict(args: argparse.Namespace) -> None:
    with open(args.input, "r") as f:
        payload = json.load(f)
    image_paths: List[str] = payload.get("image_path", [])
    if not image_paths:
        raise SystemExit(f"No 'image_path' list in {args.input}")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}. Run training first.")

    if args.model == "cls":
        model = build_classifier()
        predict_fn = predict_cls
    elif args.model == "reg":
        model = build_regressor()
        predict_fn = predict_reg
    elif args.model == "det":
        model = build_detector()
        predict_fn = predict_det
    else:
        raise SystemExit(f"Unknown model: {args.model}")

    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.to(DEVICE).eval()

    base = Path(args.input).parent
    results = []
    for rel in image_paths:
        img_path = (base / rel)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"WARN: could not read {img_path}, predicting 0")
            results.append({"image_path": rel, "num_balls": 0})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n = predict_fn(model, rgb)
        results.append({"image_path": rel, "num_balls": int(n)})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {args.output}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train one or more models")
    pt.add_argument("--data", nargs="+", default=["roboflow"],
                    help="One or more Roboflow YOLO dataset roots to merge")
    pt.add_argument("--models", default="cls,reg,det",
                    help="Comma-separated subset of {cls,reg,det}")
    pt.add_argument("--epochs", type=int, default=8)
    pt.add_argument("--batch-size", type=int, default=32)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--workers", type=int, default=0)
    pt.add_argument("--weights-dir", default="weights")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="Run inference on input.json -> output.json")
    pp.add_argument("--input", default="input.json")
    pp.add_argument("--output", default="output.json")
    pp.add_argument("--model", choices=["cls", "reg", "det"], default="cls")
    pp.add_argument("--weights", default="weights/cls.pt")
    pp.set_defaults(func=cmd_predict)

    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
