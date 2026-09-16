# Computer Vision (VC) Project

Project for the **Computer Vision** course unit (FEUP), focused on detecting, counting and localizing balls on **8-ball pool** tables. The repository contains the main project split into **three tasks** that progress from classical vision methods to deep-learning approaches.

---

## Task overview

| Task | Goal | Approach | Status |
|------|------|----------|--------|
| **Task 1** | Perspective correction (top view), ball and ball-number detection | Classical vision (OpenCV) | Done |
| **Task 2** | **Count** the number of balls on a table | *Deep learning* — CNN (classification, regression, detection) | Done |
| **Task 3** | **Localize** the balls (bounding boxes) and image retrieval of similar tables | YOLO11 vs RT-DETR + retrieval (contrastive/autoencoder) | Done |

---

## Repository structure

```
VC/
├── Project/                  # Task 1 + Task 2 (v1)
│   ├── task1.ipynb           # Task 1 — classical pipeline (top-view + balls + numbers)
│   ├── task2.py              # Task 2 — ball counting (cls/reg/det)
│   ├── Report_Task1.pdf      # Task 1 report
│   ├── balls/  top_view/     # Task 1 generated artifacts
│   ├── development_set/      # 50 labeled images
│   ├── input.json            # list of images for inference
│   ├── output.json           # inference output
│   └── README.md             # detailed Task 2 documentation
├── Proj_Task2/               # Task 2 — final (tuned) version with reports
│   ├── task2.py              # ResNet18 / EfficientNet-B0 / SimpleCNN
│   ├── dataset/              # YOLOv8 data (train/valid/test)
│   ├── weights/              # trained models (*.pth)
│   └── reports/              # metrics, curves and comparisons
├── Proj_Task3/               # Task 3 — detection + retrieval
│   ├── task3.ipynb           # main notebook (YOLO vs RT-DETR detection + retrieval)
│   ├── dataset/              # 1560/195/196 images (train/valid/test)
│   ├── runs/  task3_runs/    # runs, figures and comparisons
│   ├── yolo11s.pt  rtdetr-l.pt  yolo26n.pt
│   └── partition.csv
├── Computer Vision.docx/.odt # course assignment briefs/notes
├── myenv/                    # virtual environment (gitignored)
└── .gitignore
```

---

## Task 1 — Top-view, balls and numbers (classical vision)

Implemented in `Project/task1.ipynb` using **OpenCV** with a fully classical pipeline:

1. **Top-view / rectification** — detection of the table's 4 corners (via `approxPolyDP` with selection of the most "rectangle-like" quadrilateral) and perspective transform to generate the top view (`top_view/`).
2. **Bounding boxes** — ball localization in the rectified image.
3. **Number detection** — identification of each ball's number (from the crops in `balls/`).

Artifacts: `top_view/`, `balls/`, `Report_Task1.pdf`, `VC_Proj_Task1.zip`.

---

## Task 2 — Ball counting with *deep learning*

There are two versions:

- `Project/task2.py` — self-contained script with **three approaches**, including the extra-credit comparison:
  - `cls` — ResNet18 (ImageNet-pretrained) with a 17-way classification head (counts 0–16);
  - `reg` — ResNet18 with a single-scalar regression head (rounded);
  - `det` — Faster R-CNN ResNet50-FPN, count = number of detections.
- `Proj_Task2/task2.py` — final tuned version: **ResNet18, EfficientNet-B0 and SimpleCNN**, up to 100 epochs with *early stopping*, *automatic mixed precision* (AMP), *test-time augmentation* (TTA) and a report harness (`reports/`).

### Results (test split — `Proj_Task2/reports/task2_comparison.csv`)

| architecture | MAE | RMSE | exact acc | ±1 acc | parameters | inference |
|--------------|-----|------|-----------|--------|------------|------------|
| **ResNet18**     | **0.306** | 0.639 | **74.0%** | **95.9%** | 11.2 M | 3.4 ms / img |
| EfficientNet-B0  | 0.337 | 0.693 | 71.4% | 96.4% | 4.0 M | 15.6 ms / img |
| SimpleCNN        | 1.133 | 1.669 | 29.6% | 78.1% | 0.39 M | 0.8 ms / img |

> **Takeaway:** ResNet18 is the most accurate (MAE = 0.31, ~74% exact counts), while SimpleCNN — despite being far faster — falls well short. See `Project/README.md` for full setup, training and inference instructions.

---

## Task 3 — Ball detection + *image retrieval*

Notebook `Proj_Task3/task3.ipynb`, with **two halves**:

### 3.1 Detection (bounding boxes, single class `Ball`)

Two detectors trained from scratch on the **same dataset** (`nc=1`, max 100 epochs with *early stopping*, 640px input) and evaluated on the **same test split**:

| model | family | mAP50-95 | mAP50 | mAP75 | precision | recall | parameters | inference |
|-------|--------|----------|-------|-------|-----------|--------|------------|------------|
| **YOLO11** (`yolo11s`) | one-stage CNN | **0.812** | 0.994 | 0.972 | **0.997** | **0.993** | **9.4 M** | **29 ms** |
| **RT-DETR** (`rtdetr-l`) | Transformer (DETR) | **0.832** | 0.995 | 0.982 | 0.995 | 0.986 | 32.0 M | 55 ms |

> **Takeaway:** RT-DETR is slightly more accurate (mAP50-95 0.83 vs 0.81), but YOLO11 is ~2× faster and ~3× lighter — a much better accuracy/cost trade-off.

### 3.2 *Image retrieval* (similar pool tables)

Without *ground-truth* rankings, a progressive retrieval pipeline was developed:

- **Image-level similarity** — MSE and SSIM baselines, with translation-consistency *sanity checks* (fragile to pixel shifts);
- **Representation-level similarity** — embeddings from a **pretrained ResNet18**;
- **Direct optimization** — **contrastive learning** (margin, positive and negative samples) plus a variant with **same-table views as positives** (dataset `a`/`t`/`f` prefixes);
- **Proxy task** — reconstruction **autoencoder**;
- **Extras** — *cosine similarity*, and **retrieval by ball layout** (YOLO detector centroids + **Chamfer distance**);
- **Quantitative evaluation** — *label-free* metrics: **translation consistency** and *view recall@k*, over 20 test queries.

### 3.3 Two-stage ball classification (extra)

Two-stage pipeline: **YOLO11** localizes the balls → **ResNet18** (trained on 64×64 crops in `ball_crops/`) classifies each ball as **cue (white) / black | stripe | solid**, with *auto-labeling* via color/pattern heuristics. Final model: `task3_runs/best_ball_classifier.pth`.

---

## Environment and dependencies

- **Python 3.10–3.14**
- **PyTorch** + **torchvision** (CPU build is enough; CUDA/GPU recommended for the detector)
- **OpenCV**, **NumPy**, **Matplotlib**, **Pandas**, **scikit-image**, **Ultralytics** (YOLO/RT-DETR)
- Virtual environment in `myenv/` (not versioned).

> **ROCm/AMD note:** the Task 2 and Task 3 scripts include overrides for AMD GPU compatibility via ROCm/HIP (`HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA=0` and a `device_count` patch). They are harmless on CPU/NVIDIA.

For detailed setup instructions, dataset downloads (Roboflow) and train/inference commands, see:
- **Task 2:** `Project/README.md`
- **Docstrings:** `Project/task2.py`, `Proj_Task2/task2.py`, and the notebooks `Project/task1.ipynb` and `Proj_Task3/task3.ipynb`