# VC — Project

This repository contains the two tasks of the VC project.

- **Task 1** — classical computer-vision pipeline (top-view rectification of pool tables).
  See [task1.ipynb](task1.ipynb) and `Report_Task1.pdf`.
- **Task 2** — deep-learning ball counting on 8-ball pool tables.
  See [task2.py](task2.py).

---

## Task 2 — Ball Counting via Deep Learning

A single self-contained PyTorch script ([task2.py](task2.py)) that trains, evaluates and runs inference for three CNN-based ball counters:

| key   | architecture                          | head                                          |
|-------|---------------------------------------|-----------------------------------------------|
| `cls` | ResNet18 (ImageNet-pretrained)        | 17-way classifier over counts `0..16`         |
| `reg` | ResNet18 (ImageNet-pretrained)        | single-scalar regression head, rounded        |
| `det` | Faster R-CNN ResNet50-FPN             | single class "ball" — count = #detections     |

The script also produces a comparison table (accuracy / MAE / ±1 accuracy / training time) — see the extra-credit section below.

### 1. Requirements

- **Python 3.10–3.14** (tested on 3.14)
- A virtual environment is recommended.

Create and activate one at the workspace root (one folder above `Project/`):

```powershell
cd ..                              # go to repo root (the one that contains Project/)
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

Install dependencies (CPU build of PyTorch is enough; switch to a CUDA build if you have an NVIDIA GPU):

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install opencv-python numpy matplotlib
```

Verify:

```powershell
python -c "import torch, torchvision, cv2, numpy; print(torch.__version__, torchvision.__version__, cv2.__version__)"
```

### 2. Download the training datasets

The datasets are **not** included in this repo — you need to fetch them from Roboflow and unzip them inside `Project/`.

#### Primary dataset (required) — `8-ball-pool-l530o`

1. Open https://universe.roboflow.com/bachelorthesis/8-ball-pool-l530o
2. Click **Download Dataset** → format **YOLOv8** → **Download zip to computer**.
3. Unzip into `Project/roboflow/` so the final layout is:

   ```
   Project/roboflow/
       data.yaml
       train/{images,labels}/
       valid/{images,labels}/
       test/{images,labels}/
   ```

#### Secondary dataset (recommended) — `8-ball-pool-15300`

The assignment's primary dataset. Adds many more images and improves accuracy substantially.

1. Open https://universe.roboflow.com/ and search for `8-ball-pool-15300` (or use the link given in the assignment brief).
2. Download in **YOLOv8** format and unzip into `Project/roboflow2/` with the same `train/valid/test` layout.

> You may merge any number of additional YOLOv8-format datasets — see [§ 4](#4-training).

> **Class filtering.** Each dataset's `data.yaml` lists its class names; classes whose names contain `dot`, `pocket`, `stick`, `hand`, `table`, `player`, or `chalk` are auto-excluded from the per-image count (they aren't balls). See `NON_BALL_NAME_KEYWORDS` in [task2.py](task2.py).

### 3. Project layout

```
Project/
├── task2.py                # main script (training + inference + comparison)
├── input.json              # list of images to run inference on
├── output.json             # produced by `predict`
├── README.md               # this file
├── roboflow/               # primary YOLOv8 dataset (you download)
├── roboflow2/              # secondary YOLOv8 dataset (optional, you download)
├── weights/                # produced by `train`: cls.pt, reg.pt, det.pt, comparison.json
├── development_set/        # 50 labelled images shipped with the project
├── example_json/           # reference for the JSON output format
├── balls/, top_view/       # Task 1 artefacts
└── task1.ipynb             # Task 1 notebook
```

### 4. Training

From the `Project/` folder, with the venv active:

**One dataset:**
```powershell
..\myenv\Scripts\python.exe task2.py train --data roboflow --epochs 20
```

**Merging multiple datasets** (recommended — the script handles different class orderings per dataset automatically):
```powershell
..\myenv\Scripts\python.exe task2.py train --data roboflow roboflow2 --epochs 20
```

**Faster smoke test** (skip the slow Faster R-CNN; fine for early iteration):
```powershell
..\myenv\Scripts\python.exe task2.py train --data roboflow roboflow2 --models cls,reg --epochs 10
```

**All `train` flags:**

| flag             | default        | meaning                                          |
|------------------|----------------|--------------------------------------------------|
| `--data`         | `roboflow`     | one or more YOLOv8 dataset roots (space-separated) |
| `--models`       | `cls,reg,det`  | comma-separated subset of `{cls,reg,det}`        |
| `--epochs`       | `8`            | training epochs (detector uses `epochs // 2`)    |
| `--batch-size`   | `32`           | classifier/regressor batch size; detector uses ≤ ¼ of this |
| `--lr`           | `1e-3`         | base learning rate                               |
| `--workers`      | `0`            | DataLoader worker processes                      |
| `--weights-dir`  | `weights`      | output directory for `*.pt` files and `comparison.json` |

> Training overwrites the previous `weights/*.pt` and `weights/comparison.json`. If you want to keep an earlier run, copy `weights/` somewhere first.

> **Time estimates on CPU (8 epochs, ~1100 training images):**
> `cls` ≈ 2 min, `reg` ≈ 2 min, `det` ≈ 30 min. Detection training also benefits the most from a GPU.

### 5. Inference — produce `output.json`

The script reads `input.json` (a JSON object with an `image_path` list of paths **relative to `input.json`'s folder**) and writes `output.json` as a list of `{"image_path": ..., "num_balls": N}`.

```powershell
..\myenv\Scripts\python.exe task2.py predict --model cls --weights weights\cls.pt
```

**All `predict` flags:**

| flag        | default            | meaning                                       |
|-------------|--------------------|-----------------------------------------------|
| `--input`   | `input.json`       | input JSON file                               |
| `--output`  | `output.json`      | output JSON file                              |
| `--model`   | `cls`              | one of `{cls, reg, det}`                      |
| `--weights` | `weights/cls.pt`   | matching weight file for the chosen model     |

To use the regressor instead: `--model reg --weights weights\reg.pt`. For the detector: `--model det --weights weights\det.pt`.

### 6. Extra credit — model comparison

Every training run writes `weights/comparison.json` containing, for each trained model on the **test split**:

- `accuracy` — exact-count accuracy
- `mae` — mean absolute error of the count
- `within_one` — fraction of predictions within ±1 ball
- `train_time_sec`, `eval_time_sec`

The same table is printed at the end of training. Use these numbers in the final report.

### 7. Troubleshooting

- **`Dataset root not found: roboflow`** — you haven't unzipped the Roboflow export into `Project/roboflow/` (or your folder is named differently — pass the correct path with `--data`).
- **`No 'image_path' list in input.json`** — `input.json` must have the shape `{"image_path": [...]}`.
- **`Weights not found: weights/cls.pt`** — you need to run `train` first, or pass `--weights` pointing at an existing `.pt`.
- **PyTorch install fails on Python 3.14** — use the CPU index URL exactly as shown in § 1; the default PyPI index does not yet have 3.14 wheels at the time of writing.
- **Out-of-memory during detection training** — lower `--batch-size` (it is internally divided by 4 for the detector).

