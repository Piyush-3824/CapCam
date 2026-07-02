"""
CapCam YOLOv11s Model Training Script
======================================
Dataset breakdown (verified):
  Train  : 4,227 images
  Valid  :   402 images
  Test   :   195 images
  Total  : 4,824 images

Classes (9):
  AKM, DANGER_POSE, KNIFE, M4, NOT_DANGER_POSE, PKS, RPG, SNIPER, pistol

Model  : YOLOv11s  (yolo11s.pt)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force NVIDIA GPU

from ultralytics import YOLO
import torch
import glob

# ─────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────

DATA_YAML   = r"d:\CapCam Backend\archive\data.yaml"
MODEL_PATH  = r"d:\CapCam Backend\archive\yolo11s.pt"
PROJECT_DIR = r"d:\CapCam Backend\archive\runs"
RUN_NAME    = "capcam_yolo11s"

# ✅ OPTIMIZED FOR RTX 4060 + 16GB RAM
EPOCHS      = 200
BATCH_SIZE  = 16
IMAGE_SIZE  = 640
WORKERS     = 0
DEVICE      = 0 if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
#  PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────

assert os.path.exists(DATA_YAML),  f"[ERROR] data.yaml not found : {DATA_YAML}"
assert os.path.exists(MODEL_PATH), f"[ERROR] Weights not found   : {MODEL_PATH}"

TRAIN_DIR = r"d:\CapCam Backend\archive\train\images"
VALID_DIR = r"d:\CapCam Backend\archive\valid\images"
TEST_DIR  = r"d:\CapCam Backend\archive\test\images"

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def count_images(folder):
    total = 0
    for ext in IMG_EXTS:
        total += len(glob.glob(os.path.join(folder, ext)))
    return total

train_count = count_images(TRAIN_DIR)
valid_count = count_images(VALID_DIR)
test_count  = count_images(TEST_DIR)
total_count = train_count + valid_count + test_count

print("=" * 50)
print("  CapCam YOLOv11s — Training Setup")
print("=" * 50)
print(f"  Train images  : {train_count:,}")
print(f"  Valid images  : {valid_count:,}")
print(f"  Test  images  : {test_count:,}")
print(f"  Total images  : {total_count:,}")
print("-" * 50)
print(f"  Device        : {DEVICE}  (CUDA: {torch.cuda.is_available()})")

if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")

print(f"  Epochs        : {EPOCHS}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"  Image size    : {IMAGE_SIZE}")
print(f"  Steps/epoch   : {train_count // BATCH_SIZE}")
print("=" * 50)
print()

# Safety check
if train_count == 0:
    raise RuntimeError(f"[ERROR] No training images found in: {TRAIN_DIR}")

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────

model = YOLO(MODEL_PATH)
print(f"[INFO] Loaded model : {MODEL_PATH}\n")

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

results = model.train(
    data        = DATA_YAML,
    epochs      = EPOCHS,
    batch       = BATCH_SIZE,
    imgsz       = IMAGE_SIZE,
    device      = DEVICE,
    workers     = WORKERS,
    project     = PROJECT_DIR,
    name        = RUN_NAME,
    exist_ok    = True,

    # ── Augmentation ──────────────────────────
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    degrees     = 5.0,
    translate   = 0.1,
    scale       = 0.5,
    flipud      = 0.0,
    fliplr      = 0.5,
    mosaic      = 1.0,
    mixup       = 0.1,

    # ── Optimiser ─────────────────────────────
    optimizer        = "AdamW",
    lr0              = 0.0008,
    lrf              = 0.01,
    momentum         = 0.937,
    weight_decay     = 0.0005,
    warmup_epochs    = 3,
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.1,

    # ── Early stopping ────────────────────────
    patience    = 50,

    # ── Performance ───────────────────────────
    cache       = False,   # keep False (RAM limit)
    amp         = True,    # faster training on GPU

    # ── Logging ───────────────────────────────
    save        = True,
    save_period = 10,
    verbose     = True,
    plots       = True,
)

print("\n[INFO] Training complete.")
print(f"[INFO] Best weights  : {results.save_dir}\\weights\\best.pt")
print(f"[INFO] Last weights  : {results.save_dir}\\weights\\last.pt")

# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────

print("\n[INFO] Running validation on best weights ...")
best_weights = os.path.join(results.save_dir, "weights", "best.pt")
model_best   = YOLO(best_weights)

metrics = model_best.val(
    data    = DATA_YAML,
    imgsz   = IMAGE_SIZE,
    batch   = BATCH_SIZE,
    device  = DEVICE,
    split   = "val",
    plots   = True,
    verbose = True,
)

print("\n" + "=" * 50)
print("  FINAL RESULTS")
print("=" * 50)
print(f"  mAP50      : {metrics.box.map50:.4f}")
print(f"  mAP50-95   : {metrics.box.map:.4f}")
print(f"  Precision  : {metrics.box.mp:.4f}")
print(f"  Recall     : {metrics.box.mr:.4f}")
print("=" * 50)

